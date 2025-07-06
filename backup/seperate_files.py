import pandas as pd
import os
import glob
import re

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
        group = group.sort_values(by="code")
        output_file = os.path.join(output_folder, f"{year}_{quarter}.csv")
        group.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 연도 {year}, 분기 {quarter} 데이터가 {output_file}에 저장되었습니다.")

import os
import chardet

def detect_csv_encodings(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read(10000)  # 처음 10KB만 샘플로 사용
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']
                    confidence = result['confidence']
                    print(f"{filename}: encoding = {encoding}, confidence = {confidence:.2f}")
            except Exception as e:
                print(f"{filename}: 에러 발생 - {e}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.dates as mdates  # 날짜 형식 지정을 위해 추가
import os


def set_korean_font():
    """ 운영체제에 맞는 한글 폰트를 설정합니다. """
    system_name = os.name

    if system_name == 'nt':  # Windows
        font_family = 'Malgun Gothic'
    elif system_name == 'darwin':  # Mac OS
        font_family = 'AppleGothic'
    else:  # Linux
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        if os.path.exists(font_path):
            font_family = 'NanumGothic'
        else:
            font_family = 'sans-serif'

    plt.rc('font', family=font_family)
    plt.rc('axes', unicode_minus=False)


def analyze_and_plot_performance(model_a_path: str, model_b_path: str, output_path: str, model_a_name, model_b_name):
    """
    두 모델의 성과를 분석하고, 누적수익률과 월별 성과차이를 시각화하여 파일로 저장합니다.
    """
    try:
        df_a = pd.read_csv(model_a_path)
        df_b = pd.read_csv(model_b_path)

        for df in [df_a, df_b]:
            df['date'] = pd.to_datetime(df['date'])
            if 'return' not in df.columns:
                raise KeyError(f"파일에 'return' 컬럼이 없습니다. 실제 컬럼: {df.columns.tolist()}")

        df_a = df_a.rename(columns={'return': 'return_a'}).set_index('date')
        df_b = df_b.rename(columns={'return': 'return_b'}).set_index('date')

        merged_df = pd.merge(df_a[['return_a']], df_b[['return_b']], on='date', how='outer').fillna(0)

        merged_df['cumulative_a'] = (1 + merged_df['return_a']).cumprod() - 1
        merged_df['cumulative_b'] = (1 + merged_df['return_b']).cumprod() - 1
        merged_df['monthly_difference'] = (merged_df['return_a'] - merged_df['return_b']) * 100

        set_korean_font()
        fig, axes = plt.subplots(2, 1, figsize=(15, 14), sharex=True)
        fig.suptitle('모델 A vs 모델 B 성과 분석', fontsize=20, y=0.95)

        # --- 첫 번째 그래프: 누적 수익률 추이 ---
        ax1 = axes[0]
        ax1.plot(merged_df.index, merged_df['cumulative_a'], label=f'{model_a_name} 누적 수익률', color='crimson', linewidth=2)
        ax1.plot(merged_df.index, merged_df['cumulative_b'], label=f'{model_b_name} 누적 수익률', color='royalblue', linewidth=2)
        ax1.set_title('모델별 누적 수익률 추이', fontsize=16)
        ax1.set_ylabel('누적 수익률', fontsize=12)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax1.legend()
        ax1.grid(True, linestyle='--', linewidth=0.5)

        # --- 두 번째 그래프: 월별 성과 차이 ---
        ax2 = axes[1]
        max_diff_date = merged_df['monthly_difference'].idxmax()
        min_diff_date = merged_df['monthly_difference'].idxmin()

        ax2.plot(merged_df.index, merged_df['monthly_difference'], label=f'월별 성과 차이 ({model_a_name} - {model_b_name})', color='green')
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax2.scatter(max_diff_date, merged_df.loc[max_diff_date, 'monthly_difference'], color='red', s=80, zorder=5,
                    label=f'{model_a_name} 최대 우위')
        ax2.scatter(min_diff_date, merged_df.loc[min_diff_date, 'monthly_difference'], color='blue', s=80, zorder=5,
                    label=f'{model_b_name} 최대 우위')
        ax2.set_title('월별 수익률 차이', fontsize=16)
        ax2.set_xlabel('날짜', fontsize=12)
        ax2.set_ylabel('초과 수익률 (%p)', fontsize=12)
        ax2.legend()
        ax2.grid(True, linestyle='--', linewidth=0.5)

        # --- [수정된 부분] x축 날짜 형식 지정 ---
        # 날짜 포맷을 'YYYY-MM' 형식으로 지정합니다.
        date_format = mdates.DateFormatter('%Y-%m')
        ax2.xaxis.set_major_formatter(date_format)

        # x축 라벨이 겹치지 않도록 30도 회전시킵니다.
        plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
        ax1.xaxis.set_major_formatter(date_format)

        # x축 라벨이 겹치지 않도록 30도 회전시킵니다.
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
        # --- [수정 끝] ---

        # 레이아웃을 조절하여 라벨이 잘리지 않도록 합니다.
        fig.tight_layout(pad=2.5)

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"'{output_dir}' 디렉터리를 생성했습니다.")

        plt.savefig(output_path, dpi=300)
        print(f"\n[성공] 분석 그래프를 '{output_path}' 경로에 저장했습니다.")

        #plt.show()

    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요.\n{e}")
    except KeyError as e:
        print(f"오류: {e}")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def plot_kospi200_chart(csv_path: str):
    """
    주어진 CSV 파일 경로를 사용하여 KOSPI 200 지수 차트를 생성하고 표시합니다.
    x축은 매년 4분기 시작일을 기준으로 '{연도}_Q4' 형식으로 표시됩니다.
    'KS200.csv' 파일의 '날짜', '종가' 컬럼명에 맞춰 수정되었으며, 범례가 제거되었습니다.
    x축 범위는 2015년 4분기부터 2024년 4분기까지로 제한됩니다.
    x축/y축 라벨과 눈금 값 모두 크고 굵게 표시됩니다.

    Args:
        csv_path (str): KOSPI 200 데이터가 포함된 CSV 파일의 경로.
    """
    try:
        # 한글 폰트 설정
        try:
            plt.rc('font', family='Malgun Gothic')  # Windows
        except:
            try:
                plt.rc('font', family='AppleGothic')  # macOS
            except:
                font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
                if fm.findfont(fm.FontProperties(fname=font_path)):
                    plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
        plt.rcParams['axes.unicode_minus'] = False

        # 1. CSV 파일 읽기 및 데이터 필터링
        df = pd.read_csv(csv_path, parse_dates=['날짜'], index_col='날짜')
        df.sort_index(inplace=True)
        start_date = '2015-10-01'
        end_date = '2024-10-01'
        df_filtered = df.loc[start_date:end_date].copy()

        if df_filtered.empty:
            print(f"오류: {start_date}부터 {end_date}까지의 데이터가 파일에 없습니다.")
            return

        price_column = '종가'
        if price_column not in df_filtered.columns:
            raise ValueError(f"차트를 그릴 가격 데이터 컬럼('{price_column}')을 찾을 수 없습니다.")

        # 2. 그래프 생성
        fig, ax = plt.subplots(figsize=(17, 8))  # 세로 길이를 조금 늘려 공간 확보
        ax.plot(df_filtered.index, df_filtered[price_column], color='royalblue',linewidth=2.5)

        # 3. x축 눈금 및 레이블 설정
        years = range(2015, 2025)
        xticks = [pd.Timestamp(f'{year}-10-01') for year in years]
        xtick_labels = [f'{year-2000}/10' for year in years]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, ha='center',fontsize=8)

        # 4. x축 범위 명시적 설정
        ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
        ax.set_ylim(160,450)

        # 5. 그래프 스타일 및 정보 추가
        #ax.set_title('KOSPI 200 지수 (2015_Q4 - 2024_Q4)', fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('KOSPI200 Index', fontsize=25, fontweight='bold')
        ax.set_xlabel('Date', fontsize=25, fontweight='bold')

        # 6. x축과 y축 눈금 값 스타일 변경 (수정된 부분)
        # 글씨 크기를 12로, 굵게(bold) 설정
        plt.setp(ax.get_xticklabels(), fontsize=20, fontweight='bold')
        plt.setp(ax.get_yticklabels(), fontsize=20, fontweight='bold')


        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # 'color'를 'facecolor'로 변경하여 면과 테두리 색을 분리
        ax.axvspan('2020-10-01', '2021-10-01', facecolor='skyblue', alpha=0.3, edgecolor='black', linewidth=3, zorder=-4)
        ax.axvspan('2021-10-01', '2022-10-01', facecolor='lightgreen', alpha=0.3, edgecolor='black', linewidth=3,
                   zorder=-3)
        ax.axvspan('2022-10-01', '2023-10-01', facecolor='yellow', alpha=0.3, edgecolor='black', linewidth=3, zorder=-2)
        ax.axvspan('2023-10-01', '2024-10-01', facecolor='lightpink', alpha=0.3, edgecolor='black', linewidth=3,
                   zorder=-1)
        # y축 400~450 & x축 2016-10-01 ~ 2020-10-01 영역을 투명 회색으로 표시
        ax.fill_between([pd.to_datetime('2015-10-01'), pd.to_datetime('2020-10-01')], 400, 450, color='lightgray', alpha=1,
                        zorder=-3.5)
        ax.fill_between([pd.to_datetime('2016-10-01'), pd.to_datetime('2021-10-01')], 350, 400, color='lightgray', alpha=1,
                        zorder=-2.5)
        #ax.fill_between([pd.to_datetime('2018-10-01'), pd.to_datetime('2022-10-01')], 300, 350, color='lightgray', alpha=1,zorder=-1.5)
        #ax.fill_between([pd.to_datetime('2019-10-01'), pd.to_datetime('2023-10-01')], 250, 300, color='lightgray', alpha=1,zorder=-0.5)
        ax.grid(True, linestyle='-', alpha=0.6)

        # 레이아웃을 조정하여 라벨이 잘리지 않도록 합니다.
        plt.tight_layout()
        plt.show();

    except FileNotFoundError:
        print(f"오류: '{csv_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    except KeyError as e:
        print(f"오류: CSV 파일에서 필요한 컬럼({e})을 찾을 수 없습니다. 파일 내용을 확인해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")


import pandas as pd
import glob
import os


def analyze_merged_csv(folder_path):
    """
    지정된 폴더의 모든 CSV 파일을 세로로 병합하고, 열 목록을 출력합니다.
    이후 결측치가 50%가 넘는 열과 해당 열의 결측치 개수를 출력합니다.

    Args:
        folder_path (str): CSV 파일들이 위치한 폴더의 경로입니다.
                           예: './data_kr/merged/'
    """
    # 1. 폴더 경로를 사용하여 모든 CSV 파일의 전체 경로를 리스트로 가져옵니다.
    all_csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not all_csv_files:
        print(f"'{folder_path}' 폴더에서 CSV 파일을 찾을 수 없습니다. 🤷")
        return

    # 2. 찾은 모든 CSV 파일을 순서대로 읽어 데이터프레임 리스트를 만듭니다.
    df_list = [pd.read_csv(file) for file in all_csv_files]
    print(f"총 {len(df_list)}개의 CSV 파일을 찾았습니다.")

    # 3. 데이터프레임들을 세로 방향(axis=0)으로 모두 합칩니다.
    # ignore_index=True는 기존 파일들의 인덱스를 무시하고 새로 인덱스를 부여합니다.
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    print("✅ 모든 CSV 파일을 성공적으로 병합했습니다!")

    print("-" * 50)  # 구분을 위한 라인

    # 4. 병합된 데이터프레임의 전체 열 목록을 출력합니다.
    print("📋 병합 후 전체 열 목록입니다:")
    print(merged_df.columns.tolist())

    print("-" * 50)  # 구분을 위한 라인

    # 5. 결측치가 50%가 넘는 열을 찾아 결측치 정보와 함께 출력합니다.
    print("🔍 결측치가 50% 이상인 열 목록입니다:")

    # 전체 행의 개수를 구합니다.
    total_rows = len(merged_df)
    # 각 열의 결측치 개수를 계산합니다.
    missing_values = merged_df.isnull().sum()

    # 결측치가 50%를 넘는 열이 있는지 확인하기 위한 플래그
    found_missing_columns = False

    for column_name, missing_count in missing_values.items():
        if missing_count == 0:
            continue

        # 결측치 비율을 계산합니다.
        missing_percentage = (missing_count / total_rows) * 100

        # 비율이 50%를 초과하는 경우 해당 열의 정보를 출력합니다.
        if missing_percentage > 50:
            print(f"  - 열 이름: '{column_name}'")
            print(f"    - 결측치 개수: {missing_count}개")
            print(f"    - 결측치 비율: {missing_percentage:.2f}%")
            found_missing_columns = True

    if not found_missing_columns:
        print("결측치가 50% 이상인 열이 없습니다. ✨")


# --- 함수 실행 ---
# 아래 변수에 분석하고 싶은 CSV 파일들이 있는 폴더 경로를 지정해주세요.
TARGET_FOLDER_PATH = './data_kr/merged/'

# 함수를 호출하여 분석을 시작합니다.
analyze_merged_csv(TARGET_FOLDER_PATH)