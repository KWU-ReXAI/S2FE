import OpenDartReader
import pandas as pd
import os
import time
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

api_key = 'a4ccf72e53bf597911d0ff504d58c5f09f2029a3'
dart = OpenDartReader(api_key)

kospi_200_file = './data_kr/kospi_200.txt'
kospi_200 = {}

#jump_cnt = 0  # 데이터 뛰어넘을 때 사용하는 변수

with open(kospi_200_file, 'r',encoding='utf-8 sig') as f:
	for line in f:
		code, name, sector = line.strip().split(',')
		kospi_200[code] = name

quarters = {
	'Q1': '11013',
	'Q2': '11012',
	'Q3': '11014',
	'Q4': '11011'
}

years = list(range(2015, 2025))
save_valid_stock = []

for corp_code, corp_name in kospi_200.items():
	if corp_code == 'code': continue
	corp_code = corp_code.zfill(6)
	#jump_cnt += 1

	annual_data_dict = {}
	valid_years = []

	for year in years:
		annual_data = []

		for quarter, code in quarters.items():
			try:
				data = dart.finstate(corp_code, year, reprt_code=code)

				if data is not None:
					data = data[data['fs_nm'] == '재무제표']

					if not data.empty:
						data['Year'] = year
						data['Quarter'] = quarter
						annual_data.append(
							data[['Year', 'Quarter', 'sj_div', 'account_nm', 'thstrm_dt', 'thstrm_amount']]
						)

			except Exception as e:
				print(f"{corp_name} - {year} {quarter} financial data collection error: {e}")
			time.sleep(0.5)

		if annual_data:
			annual_df = pd.concat(annual_data, ignore_index=True)
			annual_data_dict[year] = annual_df
			valid_years.append(year)

	if set(valid_years) == set(years):  # 모든 연도 데이터가 존재하는 경우
		folder_path = f'./data_kr/financial_statements/{corp_code}'
		os.makedirs(folder_path, exist_ok=True)  # 폴더 생성 (필요할 때만)

		for year, annual_df in annual_data_dict.items():
			file_name = f'{folder_path}/{corp_code}_{year}.csv'

			is_data = annual_df[annual_df['sj_div'] == 'IS'].copy()
			is_data['thstrm_amount'] = is_data['thstrm_amount'].replace('-', np.nan)
			is_data['thstrm_amount'] = is_data['thstrm_amount'].replace(',', '', regex=True)
			is_data['thstrm_amount'] = pd.to_numeric(is_data['thstrm_amount'], errors='coerce')
			is_data['thstrm_amount'] = is_data['thstrm_amount'].ffill().bfill()

			q4_data = is_data[is_data['Quarter'] == 'Q4'].copy()
			for index, row in q4_data.iterrows():
				year = row['Year']
				account = row['account_nm']
				sum_q1_to_q3 = is_data[
					(is_data['Year'] == year) &
					(is_data['account_nm'] == account) &
					(is_data['Quarter'].isin(['Q1', 'Q2', 'Q3']))
					]['thstrm_amount'].sum(skipna=True)

				q4_data.at[index, 'thstrm_amount'] = row['thstrm_amount'] - sum_q1_to_q3

			annual_df.loc[is_data.index, 'thstrm_amount'] = is_data['thstrm_amount']
			annual_df.loc[q4_data.index, 'thstrm_amount'] = q4_data['thstrm_amount']
			annual_df['thstrm_amount'] = annual_df['thstrm_amount'].fillna(0)

			##### 분기보고서 최초제출일 구하기 #####
			# 분기보고서 가져오기
			df = dart.list(corp=corp_code, start=f'{year}0401', end=f'{year+1}0415', kind='A', final=False)
			# 정정 보고서 제외 (report_nm 앞에 '[기재정정]'이 있는 행 제거)
			df = df[~df['report_nm'].str.contains(r'^\[기재정정\]|^\[첨부정정\]', regex=True)].reset_index(drop=True)

			annual_df["공시일"] = None
			if (annual_df["Quarter"] == "Q1").any(): annual_df.loc[annual_df["Quarter"] == "Q1", "공시일"] = df.loc[3, "rcept_dt"]
			if (annual_df["Quarter"] == "Q2").any(): annual_df.loc[annual_df["Quarter"] == "Q2", "공시일"] = df.loc[2, "rcept_dt"]
			if (annual_df["Quarter"] == "Q3").any(): annual_df.loc[annual_df["Quarter"] == "Q3", "공시일"] = df.loc[1, "rcept_dt"]
			if (annual_df["Quarter"] == "Q4").any(): annual_df.loc[annual_df["Quarter"] == "Q4", "공시일"] = df.loc[0, "rcept_dt"]

			annual_df = annual_df[['Year','Quarter','공시일','sj_div','account_nm','thstrm_dt','thstrm_amount']]
			annual_df.to_csv(file_name, index=False, encoding='utf-8-sig')
			print(f"{corp_name}'s {year} financial statements saved to {file_name}.")

		# 모든 연도 데이터가 존재하는 기업 저장
		save_valid_stock.append([corp_code, corp_name])
	else:
		print(f"{corp_name} - Data missing for some years, skipping storage.")

# 저장된 기업 목록을 DataFrame으로 변환 후 CSV 파일로 저장
valid_stock_file = './data_kr/symbols.csv'
df_valid_stock = pd.DataFrame(save_valid_stock, columns=['corp_code', 'corp_name'])
df_valid_stock.to_csv(valid_stock_file, index=False, encoding='utf-8-sig')

print(f"Valid stock list saved to {valid_stock_file}.")