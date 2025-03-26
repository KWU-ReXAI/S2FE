import pandas as pd
import os
import glob

# 1) 위에서 정의한 컬럼 목록과 처리 함수
COLUMNS_TO_DROP = [
    "당기손익-공정가치측정금융자산",
    "기타포괄손익-공정가치측정금융자산",
    "보험계약자산",
    "보험계약부채",
    "파생상품자산",
    "파생상품부채",
    "이자수익",
    "이자비용",
    "영업이익(손실)",
    "영업비용",
    "예수부채",
    "순이자손익",
    "파생상품관련손익",
    "순수수료손익",
    "차입부채",
    "상각후원가측정금융자산"
]


def process_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    return df


def merge_date_regression():
    merged_folder = "./data_kr/merged"
    output_folder = "./data_kr/date_regression"
    os.makedirs(output_folder, exist_ok=True)

    file_paths = glob.glob(os.path.join(merged_folder, "*.csv"))
    if not file_paths:
        print("merged 폴더 내 CSV 파일이 없습니다.")
        return

    all_data = []
    for file in file_paths:
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')
        except Exception as e:
            print(f"{file} 파일을 읽는 중 오류 발생: {e}")
            continue

        # 여기서 drop & 컬럼 순서 재정렬
        df = process_columns(df)

        if 'year' in df.columns and 'quarter' in df.columns:
            all_data.append(df)
        else:
            print(f"{file} 파일에 'year' 또는 'quarter' 컬럼이 없습니다.")

    if not all_data:
        print("병합할 데이터가 없습니다.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # 그룹화 이전에 drop & 재정렬을 또 하고 싶다면 이곳에서도 가능
    # combined_df = process_columns(combined_df)

    groups = combined_df.groupby(['year', 'quarter'])
    for (year, quarter), group in groups:
        # 그룹별로도 drop & 재정렬을 적용할 수 있음
        group = process_columns(group)

        output_file = os.path.join(output_folder, f"{year}_{quarter}.csv")
        group.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"연도 {year}, 분기 {quarter} 데이터가 {output_file}에 저장되었습니다.")


def save_by_sector():
    merged_folder = "./data_kr/merged"
    output_base = "./data_kr/sector"
    os.makedirs(output_base, exist_ok=True)

    file_paths = glob.glob(os.path.join(merged_folder, "*.csv"))
    if not file_paths:
        print("merged 폴더 내 CSV 파일이 없습니다.")
        return

    for file in file_paths:
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')
        except Exception as e:
            print(f"{file} 파일을 읽는 중 오류 발생: {e}")
            continue

        if df.empty:
            print(f"{file} 파일에 데이터가 없습니다.")
            continue

        if 'sector' not in df.columns or 'code' not in df.columns:
            print(f"{file} 파일에 'sector' 또는 'code' 컬럼이 없습니다.")
            continue

        sector = df.iloc[0]['sector']
        code = df.iloc[0]['code']
        code_str = str(code)
        # 6자리 숫자 형식(앞에 0 포함)으로 변환합니다.
        if len(code_str) < 6:
            code_str = code_str.zfill(6)

        # 섹터별 폴더 생성 후 파일 그대로 저장합니다.
        sector_folder = os.path.join(output_base, str(sector))
        os.makedirs(sector_folder, exist_ok=True)
        output_file = os.path.join(sector_folder, f"{code_str}.csv")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"기업 코드 {code_str}의 데이터가 섹터 '{sector}' 폴더의 {output_file}에 저장되었습니다.")


def filter_all_files_by_sector():
    # 입력 디렉토리와 출력 기본 디렉토리 설정
    input_dir = "./data_kr/date_regression"
    output_dir_base = "./data_kr/date_sector"

    # 입력 디렉토리 내의 모든 CSV 파일 검색
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    # 각 CSV 파일에 대해 처리
    for file_path in csv_files:
        file_name = os.path.basename(file_path)  # 예: "2023_Q1.csv"
        df = pd.read_csv(file_path)

        # 'sector'별로 그룹화하여 각 그룹의 데이터를 저장
        for sector, group in df.groupby('sector'):
            # 해당 sector에 해당하는 출력 디렉토리 생성
            sector_output_dir = os.path.join(output_dir_base, str(sector))
            os.makedirs(sector_output_dir, exist_ok=True)

            # 출력 파일 경로 설정
            output_file = os.path.join(sector_output_dir, file_name)
            group.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"'{output_file}'에 {sector} sector 데이터가 저장되었습니다.")


def save_sector_codes():
    # symbol.csv 파일 읽기
    df = pd.read_csv('./data_kr/symbol.csv')

    # sector 기준으로 그룹화
    for sector in df['sector'].unique():
        # 해당 섹터의 데이터만 추출
        sector_df = df[df['sector'] == sector]

        # 'code' 컬럼만 저장할 데이터프레임
        sector_code_df = sector_df[['code']]

        # 섹터별 디렉토리 생성
        output_dir = f'./data_kr/date_sector/{sector}'
        os.makedirs(output_dir, exist_ok=True)

        # CSV 파일로 저장 (인덱스 제외)
        output_file = f'{output_dir}/sector_code.csv'
        sector_code_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"섹터 '{sector}'에 속하는 code {len(sector_code_df)}개를 '{output_file}'에 저장했습니다.")


def print_csv_shapes(folder_path):
    """
    주어진 폴더 내의 모든 .csv 파일의 shape을 출력하는 함수입니다.

    Parameters:
    folder_path (str): .csv 파일들이 있는 폴더의 경로입니다.
    """
    # 폴더 내의 모든 파일을 확인합니다.
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            try:
                df = pd.read_csv(file_path)
                print(f"{file_name}의 shape: {df.shape}")
            except Exception as e:
                print(f"{file_name} 파일을 읽는 중 에러 발생: {e}")


import pandas as pd


def compare_code_and_columns(file1, file2):
    """
    두 CSV 파일을 불러와서 'code' 컬럼 값을 비교하고, 컬럼 리스트의 차이를 확인하는 함수입니다.

    Parameters:
        file1 (str): 첫 번째 CSV 파일 경로.
        file2 (str): 두 번째 CSV 파일 경로.
    """
    # 파일 불러오기
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    except Exception as e:
        print(f"파일을 불러오는 도중 오류가 발생했습니다: {e}")
        return

    # 'code' 컬럼 존재 여부 확인
    if 'code' not in df1.columns:
        print(f"첫 번째 파일({file1})에 'code' 컬럼이 존재하지 않습니다.")
    if 'code' not in df2.columns:
        print(f"두 번째 파일({file2})에 'code' 컬럼이 존재하지 않습니다.")

    # 두 파일 모두 'code' 컬럼이 있는 경우 비교 수행
    if 'code' in df1.columns and 'code' in df2.columns:
        code_set1 = set(df1['code'])
        code_set2 = set(df2['code'])

        diff1 = code_set1 - code_set2
        diff2 = code_set2 - code_set1

        print("첫 번째 파일에만 있는 'code' 값:")
        print(diff1)
        print("\n두 번째 파일에만 있는 'code' 값:")
        print(diff2)

    # 컬럼 리스트 비교
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    cols_only_in_file1 = cols1 - cols2
    cols_only_in_file2 = cols2 - cols1

    print("\n첫 번째 파일에만 있는 컬럼:")
    print(cols_only_in_file1)
    print("\n두 번째 파일에만 있는 컬럼:")
    print(cols_only_in_file2)


def remove_specific_codes(file_path):
    # 제거할 code 값들
    codes_to_remove = {68870, 86790, 71050, 6800, 4990, 30, 88350, 810, 24110, 138930,
                       5940, 32830, 5830, 29780, 105560, 2270, 16360, 37620, 3450, 55550}

    # CSV 파일 읽기
    df = pd.read_csv(file_path)

    # code 컬럼이 codes_to_remove에 포함되지 않은 행만 남김
    df_filtered = df[~df['code'].isin(codes_to_remove)]

    # 다시 같은 파일로 저장
    df_filtered.to_csv(file_path, index=False)

    print(f"Filtered data saved to {file_path}")

# 사용 예시
# compare_code_and_columns("path/to/first_file.csv", "path/to/second_file.csv")

# 198: symbol
#12 18 13 5 29 5 7 1 13 22 33 14 4 2 1
if __name__ == "__main__":
    save_sector_codes()