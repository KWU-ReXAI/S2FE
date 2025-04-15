import OpenDartReader
import pandas as pd
import os
import time
import numpy as np
import re

# 현재 파일이 있는 디렉토리로 작업 디렉토리 변경
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# OpenDartReader API 설정
api_key = 'a4ccf72e53bf597911d0ff504d58c5f09f2029a3'
dart = OpenDartReader(api_key)

# kospi_200.txt 파일 읽기 (기업 코드, 이름, 섹터)
kospi_200_file = './data_kr/kospi_200.txt'
kospi_200 = {}
with open(kospi_200_file, 'r', encoding='utf-8 sig') as f:
    for line in f:
        code, name, sector = line.strip().split(',')
        kospi_200[code] = name

# 분기별 reprt_code 설정 (재무제표 데이터 요청 시 사용)
quarters = {
    'Q1': '11013',  # 1분기보고서
    'Q2': '11012',  # 반기보고서
    'Q3': '11014',  # 3분기보고서
    'Q4': '11011'   # 사업보고서
}

# 분기별 제목 필터용 매핑 (dart.list에서 report_nm 필터링시 활용)
# Q1, Q3의 경우 "분기보고서"가 사용되며, 제목 내 괄호에 월 정보가 포함됩니다.
# Q2는 "반기보고서", Q4는 "사업보고서" 형태로 가정합니다.
quarter_name_map = {
    'Q1': '분기보고서',
    'Q2': '반기보고서',
    'Q3': '분기보고서',
    'Q4': '사업보고서'
}

def get_quarter_from_report(report_nm, year):
    """
    보고서 제목(report_nm)에서 "(연도.MM)" 형태의 정보를 추출하여
    월이 '03'이면 'Q1', '09'이면 'Q3'을 반환합니다.
    해당 형태가 없으면 None을 반환합니다.
    """
    pattern = r'\(\s*' + str(year) + r'\.(\d{2})\s*\)'
    match = re.search(pattern, report_nm)
    if match:
        month = match.group(1)
        if month == '03':
            return 'Q1'
        elif month == '09':
            return 'Q3'
    return None

years = list(range(2015, 2025))
save_valid_stock = []

for corp_code, corp_name in kospi_200.items():
    if corp_code == 'code':  # 파일 헤더 등 제외
        continue
    #corp_code = corp_code.zfill(6)

    annual_data_dict = {}
    valid_years = []

    for year in years:
        annual_data = []

        # (1) 해당 연도의 공시 목록 수집 (dart.list 호출)
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        try:
            disclosure_list = dart.list(corp=corp_code, start=start_date, end=end_date)
        except Exception as e:
            print(f"{corp_name} - {year}년도 공시 목록 조회 오류: {e}")
            disclosure_list = pd.DataFrame()

        # (2) 분기별 재무제표 데이터 수집
        for quarter, reprt_code in quarters.items():
            try:
                data = dart.finstate(corp_code, year, reprt_code=reprt_code)
                if data is not None:
                    data = data[data['fs_nm'] == '재무제표']
                    if not data.empty:
                        # (3) 공시일 추출 로직
                        disclosure_date = None
                        if not disclosure_list.empty:
                            # report_nm에 해당 분기 필터 키워드가 포함된 행을 선택
                            matching_disclosure = disclosure_list[
                                disclosure_list['report_nm'].str.contains(quarter_name_map[quarter], na=False)
                            ]
                            # Q1, Q3의 경우 정규표현식을 사용하여 괄호 내 월 정보를 추출합니다.
                            if quarter in ['Q1', 'Q3']:
                                for idx, row in matching_disclosure.iterrows():
                                    extracted_quarter = get_quarter_from_report(row['report_nm'], year)
                                    if extracted_quarter == quarter:
                                        disclosure_date = row['rcept_dt']
                                        break
                            else:
                                # Q2, Q4의 경우 키워드만으로 매칭 (여러 건이 있으면 첫 번째 데이터를 사용)
                                if not matching_disclosure.empty:
                                    disclosure_date = matching_disclosure.iloc[0]['rcept_dt']

                        # (4) 공시일, 연도, 분기 정보를 데이터에 추가
                        data['공시일'] = disclosure_date
                        data['Year'] = year
                        data['Quarter'] = quarter

                        # 필요한 컬럼만 선택하여 리스트에 추가
                        data = data[['Year', 'Quarter', '공시일', 'sj_div', 'account_nm', 'thstrm_dt', 'thstrm_amount']]
                        annual_data.append(data)
            except Exception as e:
                print(f"{corp_name} - {year} {quarter} 재무제표 수집 오류: {e}")
            time.sleep(0.5)

        if annual_data:
            annual_df = pd.concat(annual_data, ignore_index=True)
            annual_data_dict[year] = annual_df
            valid_years.append(year)

    # 모든 연도의 데이터가 수집된 경우에만 저장
    if set(valid_years) == set(years):
        folder_path = f'./data_kr/financial_statements/{corp_code}'
        os.makedirs(folder_path, exist_ok=True)

        for year, annual_df in annual_data_dict.items():
            file_name = f'{folder_path}/{corp_code}_{year}.csv'

            # 손익계산서(IS) 데이터 전처리
            is_data = annual_df[annual_df['sj_div'] == 'IS'].copy()
            is_data['thstrm_amount'] = is_data['thstrm_amount'].replace('-', np.nan)
            is_data['thstrm_amount'] = is_data['thstrm_amount'].replace(',', '', regex=True)
            is_data['thstrm_amount'] = pd.to_numeric(is_data['thstrm_amount'], errors='coerce')
            is_data['thstrm_amount'] = is_data['thstrm_amount'].ffill().bfill()

            # Q4 데이터 보정: Q1~Q3의 합을 제외
            q4_data = is_data[is_data['Quarter'] == 'Q4'].copy()
            for index, row in q4_data.iterrows():
                this_year = row['Year']
                account = row['account_nm']
                sum_q1_to_q3 = is_data[
                    (is_data['Year'] == this_year) &
                    (is_data['account_nm'] == account) &
                    (is_data['Quarter'].isin(['Q1','Q2','Q3']))
                ]['thstrm_amount'].sum(skipna=True)
                q4_data.at[index, 'thstrm_amount'] = row['thstrm_amount'] - sum_q1_to_q3

            # 값 업데이트 후 결측치는 0으로 채움
            annual_df.loc[is_data.index, 'thstrm_amount'] = is_data['thstrm_amount']
            annual_df.loc[q4_data.index, 'thstrm_amount'] = q4_data['thstrm_amount']
            annual_df['thstrm_amount'] = annual_df['thstrm_amount'].fillna(0)

            annual_df.to_csv(file_name, index=False, encoding='utf-8-sig')
            print(f"{corp_name}의 {year}년도 재무제표가 {file_name}에 저장되었습니다.")

        save_valid_stock.append([corp_code, corp_name])
    else:
        print(f"{corp_name} - 일부 연도의 데이터가 누락되어 저장하지 않습니다.")

# 저장된 기업 목록을 DataFrame으로 변환 후 CSV 파일로 저장
valid_stock_file = './data_kr/symbol_saved.csv'
df_valid_stock = pd.DataFrame(save_valid_stock, columns=['corp_code', 'corp_name'])
df_valid_stock.to_csv(valid_stock_file, index=False, encoding='utf-8-sig')

print(f"유효한 주식 목록이 {valid_stock_file}에 저장되었습니다.")
