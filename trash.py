import os
import csv
import pandas as pd

# CSV 파일이 저장된 폴더 경로를 입력하세요.
folder_path = "./data_kr/date_regression/"  # 예: "./data_folder"

# 폴더 내 모든 파일 탐색
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # CSV 파일만 선택
        file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_csv(file_path)  # CSV 파일 읽기
            print(f"{file_name}: {df.shape}")  # 파일 이름과 shape 출력
        except Exception as e:
            print(f"Error reading {file_name}: {e}")  # 오류 발생 시 출력

'''
# 지정된 폴더 내의 모든 CSV 파일에 대해 처리합니다.
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(f"{filename} 파일 처리 중...")

        # 파일을 읽어 들입니다.
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            print(f"{filename} 파일은 비어있습니다.")
            continue

        # 첫 번째 행(헤더)의 열 개수를 기준으로 합니다.
        header = rows[0]
        header_length = len(header)
        cleaned_rows = [header]

        # 각 데이터 행에서 헤더보다 많은 열이 있다면, 헤더 길이에 맞게 잘라냅니다.
        for row in rows[1:]:
            if len(row) > header_length:
                row = row[:header_length]
            cleaned_rows.append(row)

        # 변경된 내용을 동일한 파일에 다시 저장합니다.
        with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(cleaned_rows)

        print(f"{filename} 파일 처리가 완료되었습니다.")
'''