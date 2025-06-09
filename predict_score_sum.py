import pandas as pd
import os

sectors = ['산업재', '정보기술']

for sector in sectors:
    base_path = fr"C:\DL\S3CE\preprocessed_data\llm\predict2\{sector}"
    save_root = fr"C:\DL\S3CE\preprocessed_data\llm\confusion_matrix\score_result\{sector}_score_result"
    os.makedirs(save_root, exist_ok=True)

    for code in os.listdir(base_path):
        code_folder = os.path.join(base_path, code)
        if os.path.isdir(code_folder):
            csv_path = os.path.join(code_folder, f"{code}.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df['upload_dt'] = pd.to_datetime(df['upload_dt'])

                    # 그룹별 집계: 시작일, 종료일, score 합계
                    monthly_scores = df.groupby(['year', 'quarter', 'month']).agg(
                        start_date=('upload_dt', 'min'),
                        end_date=('upload_dt', 'max'),
                        score_sum=('score', 'sum')
                    ).reset_index()

                    # 저장
                    save_path = os.path.join(save_root, f"{code}.csv")
                    monthly_scores.to_csv(save_path, index=False)
                    print(f"{sector} - {code} 저장 완료 → {save_path}")
                except Exception as e:
                    print(f"{sector} - {code} 처리 중 오류 발생: {e}")
