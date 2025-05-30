import os
import pandas as pd
import numpy as np
from datetime import datetime


def split_and_save(input_path: str, output_dir: str) -> None:
    """
    1) input_path의 CSV를 읽어서,
    2) 'code' 컬럼에 zfill(6) 적용,
    3) 각 code별로 분할한 뒤 'score' 컬럼(0~10 랜덤값) 추가,
    4) output_dir/{code}.csv 로 저장
    """
    # 결과 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # CSV 읽기
    df = pd.read_csv(input_path, encoding='utf-8-sig')

    # code 컬럼 zfill(6)
    df['code'] = df['code'].astype(str).str.zfill(6)

    # code별 그룹으로 저장
    for code_value, group in df.groupby('code'):
        sub_df = group.copy()
        # score 컬럼 추가 (실수형)
        sub_df['score'] = np.random.randint(0, 11, size=len(sub_df))
        # 만약 정수형 점수를 원하면 아래를 사용하세요:
        # sub_df['score'] = np.random.randint(0, 11, size=len(sub_df))

        out_path = os.path.join(output_dir, f"{code_value}.csv")
        sub_df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"분할 저장 완료: '{output_dir}' 폴더에 {len(df['code'].unique())}개의 파일이 생성되었습니다.")

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

def find_filtering_by_date(file_path: str, date_str: str) -> str:
    """
    file_path의 CSV를 읽어서,
    filtering 열의 'after:YYYY-MM-DD before:YYYY-MM-DD' 구간 중
    date_str(예: '2021-03-20')이 속하는 구간 문자열을 반환합니다.
    조건에 맞는 구간이 없으면 None을 반환합니다.
    """
    # CSV 읽기
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    # filtering에서 after, before 날짜 추출
    intervals = df['filtering'].str.extract(
        r'after:(?P<after>\d{4}-\d{2}-\d{2})\s+before:(?P<before>\d{4}-\d{2}-\d{2})'
    )
    # datetime으로 변환
    intervals['after'] = pd.to_datetime(intervals['after'])
    intervals['before'] = pd.to_datetime(intervals['before'])

    # 찾고 싶은 날짜
    target = pd.to_datetime(date_str)

    # 구간에 포함되는 행 찾기
    mask = (intervals['after'] <= target) & (target <= intervals['before'])
    if mask.any():
        # 첫 번째 일치하는 filtering 값을 반환
        return df.loc[mask, 'filtering'].iloc[0]
    else:
        return None

def get_rows_by_date_range(file_path: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """
    1) file_path의 CSV를 읽어서,
    2) filtering 열에서 after, before 날짜를 추출하고,
    3) 주어진 start_date_str ~ end_date_str 범위와 겹치는 모든 행을 반환합니다.
    """
    # CSV 읽기
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    # filtering에서 after, before 날짜 추출
    intervals = df['filtering'].str.extract(
        r'after:(?P<after>\d{4}-\d{2}-\d{2})\s+before:(?P<before>\d{4}-\d{2}-\d{2})'
    )
    intervals['after'] = pd.to_datetime(intervals['after'])
    intervals['before'] = pd.to_datetime(intervals['before'])

    # 문자열을 datetime으로 변환
    start_dt = pd.to_datetime(start_date_str)
    end_dt   = pd.to_datetime(end_date_str)

    # 두 구간이 겹치는 조건:
    # interval.after <= end_dt  AND  interval.before >= start_dt
    mask = (intervals['after'] < end_dt) & (intervals['before'] >= start_dt)

    # 겹치는 행 반환
    return df.loc[mask].reset_index(drop=True)
if __name__ == "__main__":
    # 예시 사용
    file_path      = '../result/experiment/003490.csv'
    start_date_str = '2021-03-20'
    end_date_str   = '2021-07-15'

    result_df = get_rows_by_date_range(file_path, start_date_str, end_date_str)
    if not result_df.empty:
        print(f"'{start_date_str}'부터 '{end_date_str}' 사이에 해당하는 행들:")
        print(result_df)
    else:
        print("해당 기간과 겹치는 행을 찾지 못했습니다.")