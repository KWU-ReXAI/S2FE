from FinDataLoader import FinDataLoader
import pandas as pd
import os


def load_stock_info(symbol_file):
    """
    symbol 파일에서 기업 정보를 읽어옵니다.
    파일의 각 줄은 code, name, sector 순으로 구성되어 있다고 가정합니다.
    """
    stock_info = {}
    with open(symbol_file, 'r', encoding='utf-8-sig') as f:
        header = f.readline()  # 헤더 건너뛰기
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                code, name, sector = parts[0], parts[1], parts[2]
                stock_info[code] = {"name": name, "sector": sector}
    return stock_info


def merge_financial_statements(data_loader, stock_info, start_year, end_year, quarter_list):
    """
    각 기업별 재무제표 데이터를 병합하여 딕셔너리 형태로 반환합니다.
    key는 기업 코드, value는 해당 기업의 병합된 DataFrame입니다.
    """
    merged_data = {}
    for code in data_loader.stock_list.keys():
        stock_data = []
        info = stock_info.get(code, {"name": None, "sector": None})

        for year in range(start_year, end_year + 1):
            for quarter in quarter_list:
                df_fs = data_loader.get_statement(code, year, quarter)
                if df_fs.empty:
                    continue
                # account_nm과 thstrm_amount 컬럼만 선택하고, 계정을 인덱스로 설정한 후 전치
                df_pivot = df_fs[['account_nm', 'thstrm_amount']].set_index('account_nm').T

                # 추가 정보(연도, 분기, 기업 코드, 이름, 섹터) 추가
                df_pivot['year'] = year
                df_pivot['quarter'] = quarter
                df_pivot['code'] = code
                df_pivot['name'] = info["name"]
                df_pivot['sector'] = info["sector"]

                stock_data.append(df_pivot)

        if stock_data:
            merged_df = pd.concat(stock_data, ignore_index=True)
            # 기본 컬럼들을 앞쪽으로 배치
            base_cols = ['code', 'name', 'sector', 'year', 'quarter']
            remaining_cols = [col for col in merged_df.columns if col not in base_cols]
            merged_df = merged_df[base_cols + remaining_cols]
            merged_data[code] = merged_df
    return merged_data


def save_merged_data(merged_data, output_folder):
    """
    병합된 재무제표 데이터를 지정한 폴더에 CSV 파일로 저장합니다.
    파일명은 {기업코드}.csv 형식입니다.
    """
    os.makedirs(output_folder, exist_ok=True)
    for code, df_stock in merged_data.items():
        output_file = os.path.join(output_folder, f"{code}.csv")
        df_stock.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"기업 {code}의 재무데이터가 {output_file}에 저장되었습니다.")


def main():
    # 기본 폴더 경로 설정
    base_folder = "./data_kr"
    symbol_file = os.path.join(base_folder, "symbol.csv")
    output_folder = os.path.join(base_folder, "/k_features")

    # 재무제표 데이터 로더 초기화
    data_loader = FinDataLoader(os.path.join(base_folder, "financial_statements"))

    # symbol 파일로부터 기업 정보 로드
    stock_info = load_stock_info(symbol_file)

    # 사용할 분기 목록 (영어 표기)
    quarter_list = ["Q1", "Q2", "Q3", "Q4"]

    # 병합할 연도 범위 설정
    start_year = 2015
    end_year = 2024

    # 기업별 재무제표 데이터 병합
    merged_data = merge_financial_statements(data_loader, stock_info, start_year, end_year, quarter_list)

    # 병합된 데이터를 CSV 파일로 저장
    save_merged_data(merged_data, output_folder)


if __name__ == "__main__":
    main()
