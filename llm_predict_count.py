import pandas as pd
import os

# 설정
base_dir = r"C:\DL\S3CE\preprocessed_data\llm\score_result"
sectors = ['산업재', '정보기술']

# 전체 누적 합계
grand_total_rows = 0
grand_zero_rows = 0

for sector in sectors:
    folder_path = os.path.join(base_dir, f"{sector}_score_result")
    sector_total = 0
    sector_zero = 0

    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(folder_path, file)

        try:
            df = pd.read_csv(file_path)
            total = len(df)
            zero_count = (df['score_sum'] == 0).sum()

            sector_total += total
            sector_zero += zero_count
        except Exception as e:
            print(f"[ERROR] {sector} - {file} 처리 실패: {e}")

    grand_total_rows += sector_total
    grand_zero_rows += sector_zero

    rate = (sector_zero / sector_total * 100) if sector_total else 0
    print(f"\n===== {sector} 예측 행 요약 =====")
    print(f"총 행 수            : {sector_total:,}개")
    print(f"score_sum = 0 행 수 : {sector_zero:,}개")
    print(f"0 비율              : {rate:.2f}%")

# 전체 요약
grand_rate = (grand_zero_rows / grand_total_rows * 100) if grand_total_rows else 0
print(f"\n✅ 전체(산업재+정보기술) 요약")
print(f"총 행 수            : {grand_total_rows:,}개")
print(f"score_sum = 0 행 수 : {grand_zero_rows:,}개")
print(f"0 비율              : {grand_rate:.2f}%")
