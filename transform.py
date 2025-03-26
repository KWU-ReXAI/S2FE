from FinDataLoader import FinDataLoader
import pandas as pd
import os

# 결과물 저장 폴더 생성
output_folder = "./data_kr/merged"
os.makedirs(output_folder, exist_ok=True)

# 재무제표 데이터 로더 초기화
data = FinDataLoader("./data_kr/financial_statements")

# symbol.csv 파일에서 기업 정보(코드, 이름, 섹터)를 읽어옵니다.
symbol_file = "./data_kr/symbol.csv"
stock_info = {}
with open(symbol_file, 'r', encoding='utf-8-sig') as f:
    header = f.readline()  # 첫 줄은 헤더이므로 건너뜁니다.
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            code, name, sector = parts[0], parts[1], parts[2]
            stock_info[code.zfill(6)] = {"name": name, "sector": sector}

# 사용할 분기 목록 (영어로 표기)
quarter_list = ["Q1", "Q2", "Q3", "Q4"]

# 각 기업별 재무제표 데이터를 병합하여 저장합니다.
for code in data.stock_list.keys():
    stock_data = []
    # 기업 정보(이름, 섹터)를 symbol 파일에서 가져옵니다.
    info = stock_info.get(code, {"name": None, "sector": None})

    for year in range(2015, 2025):
        for quarter in quarter_list:
            df_fs = data.get_statement(code, year, quarter)
            if df_fs.empty:
                continue
            # 재무데이터 중 account_nm과 thstrm_amount 컬럼만 선택하고,
            # account_nm을 인덱스로 설정 후 전치하여 각 계정이 컬럼이 되도록 합니다.
            df_pivot = df_fs[['account_nm', 'thstrm_amount']].set_index('account_nm').T

            # 연도, 분기, code, name, sector 컬럼 추가 (모든 컬럼명을 영어로)
            df_pivot['year'] = year
            df_pivot['quarter'] = quarter
            df_pivot['code'] = code
            df_pivot['name'] = info["name"]
            df_pivot['sector'] = info["sector"]

            stock_data.append(df_pivot)

    if stock_data:
        df_stock = pd.concat(stock_data, ignore_index=True)
        # 컬럼 순서를 재정렬 (code, name, sector, year, quarter가 앞쪽에 오도록)
        base_cols = ['code', 'name', 'sector', 'year', 'quarter']
        remaining_cols = [col for col in df_stock.columns if col not in base_cols]
        df_stock = df_stock[base_cols + remaining_cols]

        # 결과 CSV 파일 저장
        output_file = f"{output_folder}/{code}.csv"
        df_stock.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"기업 {code} ({info['name']})의 재무데이터가 {output_file}에 저장되었습니다.")
    else:
        print(f"기업 {code} ({info['name']})의 병합할 재무데이터가 없습니다.")
