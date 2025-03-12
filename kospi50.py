import OpenDartReader
import pandas as pd
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

api_key = ''
dart = OpenDartReader(api_key)

kospi_50_file = './data_kr/kospi.txt'
kospi_50 = {}

with open(kospi_50_file, 'r') as f:
    for line in f:
        code, name, sector = line.strip().split(',')
        kospi_50[code] = name

quarters = {
    'Q1': '11013',  # 1분기
    'Q2': '11012',  # 반기
    'Q3': '11014',  # 3분기
    'Q4': '11011'  # 사업보고서 (4분기)
}
years = range(2015, 2025)

# 10년 KOSPI 기업 재무제표 추출
for corp_code, corp_name in kospi_50.items():
    folder_path = f'./data_kr/financial_statements/{corp_code}'
    os.makedirs(folder_path, exist_ok=True)

    for year in years:
        annual_data = []

        for quarter, code in quarters.items():
            try:
                data = dart.finstate(corp_code, year, reprt_code=code)

                if data is not None:
                    # "재무제표" 데이터만 추출
                    data = data[data['fs_nm'] == '재무제표']

                    if not data.empty:
                        data['Year'] = year
                        data['Quarter'] = quarter
                        annual_data.append(
                            data[['Year', 'Quarter', 'sj_div', 'account_nm', 'thstrm_dt', 'thstrm_amount']])

            except Exception as e:
                print(f"{corp_name} - {year} {quarter} financial data collection error: {e}")
            time.sleep(0.5)

        if annual_data:
            annual_df = pd.concat(annual_data, ignore_index=True)
            file_name = f'{folder_path}/{corp_code}_{year}.csv'
            # 'sj_div'가 'IS'인 데이터 필터링
            is_data = annual_df[annual_df['sj_div'] == 'IS'].copy()

            # 'thstrm_amount' 컬럼에서 '-'를 NaN으로 처리
            is_data['thstrm_amount'] = is_data['thstrm_amount'].replace('-', None)

            # 'thstrm_amount'를 숫자형으로 변환
            is_data['thstrm_amount'] = is_data['thstrm_amount'].replace(',', '', regex=True).astype(float)

            # 결측치(NaN)를 이전 또는 다음 분기의 값으로 대체
            is_data['thstrm_amount'] = is_data['thstrm_amount'].ffill().bfill()

            # Q4에 대해 Q1, Q2, Q3 값을 뺀 값으로 대체
            q4_data = is_data[is_data['Quarter'] == 'Q4'].copy()
            for index, row in q4_data.iterrows():
                year = row['Year']
                account = row['account_nm']
                # 해당 연도, 계정의 Q1~Q3 값 합산
                sum_q1_to_q3 = is_data[(is_data['Year'] == year) &
                                       (is_data['account_nm'] == account) &
                                       (is_data['Quarter'].isin(['Q1', 'Q2', 'Q3']))
                                       ]['thstrm_amount'].sum()
                # Q4에서 Q1~Q3 누적값을 뺀 값을 대체
                q4_data.at[index, 'thstrm_amount'] = row['thstrm_amount'] - sum_q1_to_q3

            # 원본 데이터의 Q4 값 업데이트
            annual_df.loc[is_data.index, 'thstrm_amount'] = is_data['thstrm_amount']
            annual_df.loc[q4_data.index, 'thstrm_amount'] = q4_data['thstrm_amount']

            annual_df['thstrm_amount'] = annual_df['thstrm_amount'].replace(',', '', regex=True).astype(float)

            annual_df.to_csv(file_name, index=False, encoding='utf-8-sig')
            print(f"{corp_name}'s {year} financial statements saved to {file_name}.")
        else:
            print(f"{corp_name} - No financial data for the year {year}.")
