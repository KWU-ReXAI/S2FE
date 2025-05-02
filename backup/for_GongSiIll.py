import os
import pandas as pd
from datetime import datetime
import glob

def fill_disclosure_date(row):
    """
    disclosure_date 열이 비어있는 경우, 분기와 연도에 따라 기본값을 채워준다.
    Q1: {year}-05-16
    Q2: {year}-08-12
    Q3: {year}-11-13
    Q4: {year-1}-03-31
    """
    # disclosure_date이 결측인 경우 (NaN 또는 빈 문자열)
    if pd.isna(row['disclosure_date']) or row['disclosure_date'] == '':
        year = int(row['year'])
        quarter = row['quarter']
        if quarter == 'Q1':
            return f"{year}-05-16"
        elif quarter == 'Q2':
            return f"{year}-08-12"
        elif quarter == 'Q3':
            return f"{year}-11-13"
        elif quarter == 'Q4':
            return f"{year - 1}-03-31"
    # disclosure_date이 존재하면 그대로 반환
    return row['disclosure_date']


def adjust_q4_disclosure_date(row, mapping):
    """
    분기가 Q4인 행에 대해, 다음 연도(Q4)의 disclosure_date 값을 적용한다.
    만약 다음 연도의 Q4 값이 없으면 현재의 disclosure_date 날짜의 연도 값에 1을 더해서 반환한다.
    """
    if row['quarter'] == 'Q4':
        next_year = int(row['year']) + 1
        key = (next_year, 'Q4')
        if key in mapping:
            return mapping[key]
        else:
            # 기존 disclosure_date을 datetime으로 파싱한 후 연도만 +1 적용
            try:
                dt = datetime.strptime(row['disclosure_date'], "%Y-%m-%d")
                return f"{dt.year + 1}-{dt.month:02d}-{dt.day:02d}"
            except Exception as e:
                return row['disclosure_date']
    # Q4가 아니면 그대로 반환
    return row['disclosure_date']


def process_file(file_path, output_folder):
    """
    한 파일을 읽어들여 결측치 채우기 및 Q4 보정을 진행한 후 지정한 폴더에 저장한다.
    """
    # CSV 파일 읽기 (year는 정수형, quarter와 disclosure_date은 문자열로)
    df = pd.read_csv(file_path, dtype={'year': int, 'quarter': str, 'disclosure_date': str})

    # 1단계: disclosure_date의 결측값을 기본 날짜로 채운다.
    df['disclosure_date'] = df.apply(fill_disclosure_date, axis=1)

    # 모든 행에 대해 (year, quarter)를 key로 하는 mapping 생성
    mapping = {(row['year'], row['quarter']): row['disclosure_date'] for _, row in df.iterrows()}

    # 2단계: 분기가 Q4인 경우, 다음 연도의 Q4 disclosure_date 값을 가져오거나 없으면 연도만 +1.
    df['disclosure_date'] = df.apply(lambda row: adjust_q4_disclosure_date(row, mapping), axis=1)

    # 출력 폴더가 존재하지 않으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # 저장할 파일 경로 구성 (파일명은 원본과 동일)
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_folder, filename)

    # 보정된 데이터를 CSV 파일로 저장
    df.to_csv(output_path, index=False)
    print(f"파일이 저장되었습니다: {output_path}")


def process_all_files(input_folder, output_folder):
    """
    입력 폴더 내의 모든 CSV 파일을 순회하며 각각 처리한다.
    """
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(input_folder, file)
            process_file(file_path, output_folder)


import pandas as pd
import os
from glob import glob

def fix_q4_disclosure_year(folder='./data_kr/merged'):
    for filepath in glob(f"{folder}/*.csv"):
        df = pd.read_csv(filepath)
        code = os.path.basename(filepath).split(".")[0]

        # Q4 조건 필터
        mask = df['quarter'] == 'Q4'

        for idx in df[mask].index:
            try:
                declared_year = int(str(df.loc[idx, 'disclosure_date'])[:4])
                correct_year = int(df.loc[idx, 'year']) + 1

                if declared_year != correct_year:
                    df.loc[idx, 'disclosure_date'] = f"{correct_year}{df.loc[idx, 'disclosure_date'][4:]}"
            except Exception as e:
                print(f"{code} - 오류 발생 (index {idx}): {e}")

        # 저장
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"{code} - Q4 연도 보정 완료")


# 사용 예시:
input_folder = "./data_kr/merged_origin"  # 원본 CSV 파일들이 있는 폴더
output_folder = "./data_kr/merged"  # 보정된 파일을 저장할 폴더
process_all_files(input_folder, output_folder)
fix_q4_disclosure_year()
