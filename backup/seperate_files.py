import pandas as pd
import os
import glob
from pandas.tseries.offsets import DateOffset

# 1) 위에서 정의한 컬럼 목록과 처리 함수
'''COLUMNS_TO_DROP = [
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
]'''

COLUMNS_TO_DROP = []

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
    codes_to_remove = ['4990', '5830', '5940', '6800', '16360', '24110', '29780', '32830', '37620', '55550', '68870', '71050', '86790', '88350', '105560', '138930']
    # CSV 파일 읽기
    df = pd.read_csv(file_path)

    # code 컬럼이 codes_to_remove에 포함되지 않은 행만 남김
    df_filtered = df[~df['code'].isin(codes_to_remove)]

    # 다시 같은 파일로 저장
    df_filtered.to_csv(file_path, index=False)

    print(f"Filtered data saved to {file_path}")

def seperate_comma():
    input_path = './data_kr/symbol.csv'

    # 출력 파일 경로
    output_path = './data_kr/kospi_200.txt'

    # CSV 파일 읽기
    symbol_df = pd.read_csv(input_path)

    # code 컬럼을 문자열로 변환 후 zfill(6) 처리
    symbol_df['code'] = symbol_df['code'].astype(str).str.zfill(6)

    # ,로 구분된 텍스트 파일로 저장
    symbol_df.to_csv(output_path, index=False, sep=',')

    print(f"파일이 저장되었습니다: {output_path}")

def saveUpjongtoSymbol():
    df_data = pd.read_csv('./data_kr/20201002_업종.csv', encoding='cp949')
    df_symbol = pd.read_csv('./data_kr/symbol.csv', encoding='utf-8-sig')

    # 종목코드 문자열 변환
    df_data['종목코드'] = df_data['종목코드'].astype(str).str.zfill(6)
    df_symbol['code'] = df_symbol['code'].astype(str).str.zfill(6)

    # 종목코드 기준으로 병합하여 sector 추가
    df_updated = pd.merge(df_symbol, df_data[['종목코드', '업종명']],
                          how='left', left_on='code', right_on='종목코드')

    # 기존 sector 컬럼 삭제하고, 새로 가져온 '업종명'을 sector로 사용
    df_updated.drop(columns=['sector', '종목코드'], inplace=True)
    df_updated.rename(columns={'업종명': 'sector'}, inplace=True)

    # 결과 저장
    df_updated.to_csv('./data_kr/symbol_upjong.csv', index=False)

def saveSectortoSymbol():
    # symbol.csv 불러오기
    symbol_path = './data_kr/symbol.csv'
    sector_folder = './data_kr/섹터정보'
    output_path = './data_kr/symbol_sector.csv'
    df_symbol = pd.read_csv(symbol_path, encoding='utf-8-sig')
    df_symbol['code'] = df_symbol['code'].astype(str).str.zfill(6)
    df_symbol['sector'] = None  # sector 열 초기화

    # 섹터 폴더 내 모든 csv 파일 순회
    for filename in os.listdir(sector_folder):
        if filename.endswith('.csv'):
            sector_name = os.path.splitext(filename)[0]  # 파일명 (확장자 제거) = 섹터명
            sector_file_path = os.path.join(sector_folder, filename)

            # 섹터 파일 읽기
            df_sector = pd.read_csv(sector_file_path, encoding='utf-8-sig')
            if '종목코드' not in df_sector.columns:
                continue  # 종목코드 컬럼이 없으면 스킵

            df_sector['종목코드'] = df_sector['종목코드'].astype(str).str.zfill(6)

            # 종목코드 매칭되는 symbol에 섹터명 넣기
            df_symbol.loc[df_symbol['code'].isin(df_sector['종목코드']), 'sector'] = sector_name

    # 결과 저장
    df_symbol.to_csv(output_path, index=False, encoding='utf-8-sig')


def removeInvalidSymbolsAndFiles(folder_path, symbol_path, expected_rows=37, encoding='utf-8-sig'):
    # symbol.csv 불러오기
    df_symbol = pd.read_csv(symbol_path, encoding=encoding)
    df_symbol['code'] = df_symbol['code'].astype(str).str.zfill(6)

    invalid_codes = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                if df.shape[0] != expected_rows:
                    code = os.path.splitext(filename)[0].zfill(6)
                    invalid_codes.append(code)
                    os.remove(filepath)  # 파일 삭제
                    print(f"[삭제됨] {filename} (행 개수: {df.shape[0]})")
            except Exception as e:
                print(f"[오류] {filename} 파일을 읽는 중 오류 발생: {e}")
                code = os.path.splitext(filename)[0].zfill(6)
                invalid_codes.append(code)
                try:
                    os.remove(filepath)
                    print(f"[삭제됨] 오류난 파일 {filename}")
                except:
                    print(f"[경고] {filename} 삭제 실패")

    # symbol.csv에서 코드 삭제
    before_count = df_symbol.shape[0]
    df_symbol = df_symbol[~df_symbol['code'].isin(invalid_codes)]
    after_count = df_symbol.shape[0]

    # 저장
    df_symbol.to_csv(symbol_path, index=False, encoding=encoding)

    print(f"\n총 {before_count - after_count}개의 종목 코드가 symbol.csv에서 삭제되었습니다.")
    if invalid_codes:
        print("삭제된 종목코드 목록:", invalid_codes)
    else:
        print("모든 CSV 파일이 유효한 37개의 행을 가지고 있습니다.")


def compareFolderAndSymbol(folder_path, symbol_path, encoding='utf-8-sig'):
    # 폴더 내 csv 파일 이름들 → 종목코드 리스트
    folder_codes = [
        os.path.splitext(f)[0].zfill(6)
        for f in os.listdir(folder_path)
        if f.endswith('.csv')
    ]

    # symbol.csv의 종목코드 리스트
    df_symbol = pd.read_csv(symbol_path, encoding=encoding)
    symbol_codes = df_symbol['code'].astype(str).str.zfill(6).tolist()

    # 비교
    only_in_folder = sorted(set(folder_codes) - set(symbol_codes))
    only_in_symbol = sorted(set(symbol_codes) - set(folder_codes))
    in_both = sorted(set(folder_codes) & set(symbol_codes))

    print(f"✅ 폴더에만 있는 코드 ({len(only_in_folder)}개): {only_in_folder}")
    print(f"✅ symbol.csv에만 있는 코드 ({len(only_in_symbol)}개): {only_in_symbol}")
    print(f"✅ 둘 다에 있는 코드 ({len(in_both)}개): {in_both[:10]}{' ...' if len(in_both) > 10 else ''}")  # 일부만 출력

    return {
        "only_in_folder": only_in_folder,
        "only_in_symbol": only_in_symbol,
        "in_both": in_both
    }

def cleanSymbolWithFolder(folder_path, symbol_path, encoding='utf-8-sig'):
    # 폴더 내 파일 이름들에서 종목코드 추출
    folder_codes = [
        os.path.splitext(f)[0].zfill(6)
        for f in os.listdir(folder_path)
        if f.endswith('.csv')
    ]

    # symbol.csv 로드
    df_symbol = pd.read_csv(symbol_path, encoding=encoding)
    df_symbol['code'] = df_symbol['code'].astype(str).str.zfill(6)

    # 비교: symbol.csv에는 있지만 폴더에 없는 코드
    symbol_codes = df_symbol['code'].tolist()
    only_in_symbol = sorted(set(symbol_codes) - set(folder_codes))

    # 해당 코드 제거
    before_count = len(df_symbol)
    df_symbol = df_symbol[~df_symbol['code'].isin(only_in_symbol)]
    after_count = len(df_symbol)

    # 저장
    df_symbol.to_csv(symbol_path, index=False, encoding=encoding)

    print(f"총 {before_count - after_count}개의 종목이 symbol.csv에서 삭제되었습니다.")
    if only_in_symbol:
        print("삭제된 종목코드:", only_in_symbol)
    else:
        print("symbol.csv와 폴더 내 파일명이 완전히 일치합니다.")

def add_disclosure_date(file_path, save_path=None):
    """
    CSV 파일 경로를 받아 deadline과 quarter 열 기준으로 disclosure_date를 계산하여 저장합니다.

    :param file_path: str, 입력 CSV 파일 경로
    :param save_path: str, 결과를 저장할 경로 (None이면 기존 경로에 덮어쓰기)
    """
    df = pd.read_csv(file_path)

    # datetime 형식으로 변환
    df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')

    # disclosure_date 열 추가
    df['disclosure_date'] = df.apply(
        lambda row: row['deadline'] + pd.Timedelta(days=90) if row['quarter'] == 'Q4'
        else row['deadline'] + pd.Timedelta(days=45),
        axis=1
    )

    # 저장 경로 설정
    if save_path is None:
        save_path = file_path

    df.to_csv(save_path, index=False)
    print(f"저장 완료: {save_path}")

def merge_LLM_date_regression(sector=" "):
    merged_folder = f"../preprocessed_data/llm/predict/{sector}"
    output_folder = f"../preprocessed_data/llm/date_regression/{sector}"
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


def merge_all_sectors_to_date_regression(base_merged_folder="../preprocessed_data/llm/predict",
                                          output_folder="../preprocessed_data/llm/date_regression/cluster_1"):
    os.makedirs(output_folder, exist_ok=True)

    # 모든 섹터 폴더의 모든 CSV 수집
    all_csv_files = glob.glob(os.path.join(base_merged_folder, "*", "*.csv"))
    if not all_csv_files:
        print("병합할 CSV 파일이 없습니다.")
        return

    all_data = []
    for file in all_csv_files:
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')
            df = process_columns(df)
            if 'year' in df.columns and 'quarter' in df.columns:
                all_data.append(df)
            else:
                print(f"{file} → 'year' 또는 'quarter' 컬럼이 없습니다.")
        except Exception as e:
            print(f"{file} → 파일 읽는 중 오류 발생: {e}")

    if not all_data:
        print("유효한 데이터가 없습니다.")
        return

    # 모든 섹터의 데이터 하나로 병합
    combined_df = pd.concat(all_data, ignore_index=True)

    # 연도/분기별로 그룹화하여 저장
    groups = combined_df.groupby(['year', 'quarter'])
    for (year, quarter), group in groups:
        group = process_columns(group)
        output_file = os.path.join(output_folder, f"{year}_{quarter}.csv")
        group.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 연도 {year}, 분기 {quarter} 데이터가 {output_file}에 저장되었습니다.")
if __name__ == "__main__":
    merge_all_sectors_to_date_regression()
"""
# 198: symbol
#12 18 13 5 29 5 7 1 13 22 33 14 4 2 1
if __name__ == "__main__":
    compareFolderAndSymbol(
        folder_path='./backup/data_kr_sector/k_features/최대주주변동',
        symbol_path='./data_kr/symbol.csv'
    )

    cleanSymbolWithFolder(
        folder_path='./data_kr/merged',
        symbol_path='./data_kr/symbol.csv'
    )
    removeInvalidSymbolsAndFiles(
        folder_path='./data_kr/merged',
        symbol_path='./data_kr/symbol.csv',
        expected_rows=37
    )
    seperate_comma()

    merge_date_regression()
    save_by_sector()
    filter_all_files_by_sector()
    save_sector_codes()"""

