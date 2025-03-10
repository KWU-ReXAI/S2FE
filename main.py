'''
import os
import pandas as pd

# CSV 파일 경로
csv_path = "./data_kr/symbol.csv"

# 폴더 경로
folder_path = "./data_kr/merged"

# CSV 파일 읽기
df = pd.read_csv(csv_path)

# code 컬럼 값을 6자리 문자열로 변환 (숫자인 경우 앞에 0을 채움)
valid_codes = set(df["code"].astype(str).str.zfill(6))

# 파일 리스트 가져오기
if os.path.exists(folder_path):
    for file_name in os.listdir(folder_path):
        file_full_path = os.path.join(folder_path, file_name)

        # 파일명이 "6자리 숫자.csv" 형식인지 확인
        if file_name.endswith(".csv") and len(file_name) == 10 and file_name[:6].isdigit():
            code = file_name[:6]  # 파일명의 코드 부분 추출

            if code not in valid_codes:
                print(f"Deleting file: {file_full_path}")
                os.remove(file_full_path)  # 파일 삭제

import os
import pandas as pd

# 데이터가 저장된 폴더 경로
merged_folder = "./data_kr/merged"
output_folder = "./data_kr/sector"
symbol_file = "./data_kr/symbol.csv"
date_sector_folder = "./data_kr/date_sector"

# symbol.csv 로드
symbol_df = pd.read_csv(symbol_file)
symbol_df["code"] = symbol_df["code"].astype(str).str.zfill(6)  # 코드 값을 6자리로 변환

# sector 별 코드 리스트 저장
for sector, group in symbol_df.groupby("sector"):
    sector_folder = os.path.join(date_sector_folder, sector)
    os.makedirs(sector_folder, exist_ok=True)
    output_file = os.path.join(sector_folder, "sector_code.csv")
    group[["code"]].to_csv(output_file, index=False, encoding='utf-8-sig')

# 모든 CSV 파일 확인
for file_name in os.listdir(merged_folder):
    if file_name.endswith(".csv"):  # CSV 파일만 처리
        file_path = os.path.join(merged_folder, file_name)
        df = pd.read_csv(file_path)

        # sector, code 컬럼이 있는지 확인
        if "sector" in df.columns and "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)  # 코드 값을 6자리로 변환

            for (sector, code), group in df.groupby(["sector", "code"]):
                sector_folder = os.path.join(output_folder, sector)
                os.makedirs(sector_folder, exist_ok=True)
                output_file = os.path.join(sector_folder, f"{code}.csv")

                # 파일이 이미 존재하면 추가 (append)
                if os.path.exists(output_file):
                    group.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
                else:
                    group.to_csv(output_file, mode='w', header=True, index=False, encoding='utf-8-sig')

print("Data successfully split and saved by sector and company.")


import os
import pandas as pd
import numpy as np
from datamanager import DataManager

# 사용할 phase_list 정의
phase_list = {"p1": [0, 12, 16, 19],
              "p2": [4, 16, 20, 23],
              "p3": [8, 20, 24, 27],
              "p4": [12, 24, 28, 31],
              "p5": [16, 28, 32, 35],
              }

# DataManager 인스턴스 생성
features_n = 6  # 사용할 feature 개수
sector_list = ['Chemistry','Medicine_Production', 'Food', 'etc', 'Distribution',
                'IT_Service', 'Electronics', 'Communication',
                'Transportation', 'Metal', 'Machines']
dm = DataManager(features_n)
dm.create_date_list()

file = pd.read_csv(f"./data_kr/date_sector/IT_Service/2015_Q4.csv")
# 모든 phase에 대해 data_phase 함수 실행

for sector in sector_list:
    print(f"\nSector: {sector}")
    for phase in phase_list.keys():
        print(f"\nTesting phase: {phase}")

        # data_phase 실행 (pandas 형식)
        try:
            train_data1, valid_data1, test_data1 = dm.data_phase(sector, phase, pandas_format=True)

            print(f"Train Data Shape ({phase}): {train_data1.shape}")
            print(f"Validation Data Shape ({phase}): {valid_data1.shape}")
            print(f"Test Data Shape ({phase}): {test_data1.shape}")

        except Exception as e:
            print(f"Error occurred in phase {phase}: {e}")

        # data_phase 실행 (numpy 형식)
        try:
            train_data, valid_data, test_data = dm.data_phase(sector, phase, pandas_format=False)

            print(f"Train Data Shape ({phase}): {train_data.shape}")
            print(f"Validation Data Shape ({phase}): {valid_data.shape}")
            print(f"Test Data Shape ({phase}): {test_data.shape}")

        except Exception as e:
            print(f"Error occurred in phase {phase}: {e}")

'''

import os
import pandas as pd

# 디렉토리 경로 설정
directory_path = "./data_kr/clustered_data/ALL/"

# 디렉토리 내의 모든 CSV 파일 목록 가져오기
csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]

# 각 파일의 shape 출력
for file in csv_files:
    file_path = os.path.join(directory_path, file)
    df = pd.read_csv(file_path)  # CSV 파일 읽기
    print(f"{file}: {df.shape}")  # 파일명과 shape 출력
