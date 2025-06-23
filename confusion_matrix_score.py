import pandas as pd
import os
from datetime import timedelta

# 디렉토리 설정
base_dir = r"C:\DL\S3CE"
sectors = ['산업재', '정보기술']
score_root_template = os.path.join(base_dir, r"preprocessed_data\llm\confusion_matrix\score_result\{}_score_result")
price_path = os.path.join(base_dir, r"data_kr\price")
confusion_save_template = os.path.join(base_dir, r"preprocessed_data\llm\confusion_matrix\confusion_matrix_score\{}")

# 섹터별 반복
for sector in sectors:
    score_root = score_root_template.format(sector)
    save_root = confusion_save_template.format(sector)
    os.makedirs(save_root, exist_ok=True)

    for file in os.listdir(score_root):
        if not file.endswith(".csv"):
            continue

        code = file.replace(".csv", "")
        try:
            df_score = pd.read_csv(os.path.join(score_root, file))
            df_score['start_date'] = pd.to_datetime(df_score['start_date'], errors='coerce')
            df_score['end_date'] = pd.to_datetime(df_score['end_date'], errors='coerce')
            df_score = df_score.dropna(subset=['start_date', 'end_date'])

            # 가격 데이터 로드
            price_file = os.path.join(price_path, f"{code}.csv")
            if not os.path.exists(price_file):
                continue
            df_price = pd.read_csv(price_file)
            df_price['날짜'] = pd.to_datetime(df_price['날짜'], errors='coerce')
            df_price = df_price.dropna(subset=['날짜', '시가', '종가'])

        except Exception as e:
            print(f"[ERROR] {sector} - {code} 파일 처리 실패: {e}")
            continue

        results = []

        for _, row in df_score.iterrows():
            trade_start = row['start_date']
            trade_end = row['end_date']
            score_val = row['score_sum']

            # 시가 (거래 시작일)
            start_row = df_price[df_price['날짜'] >= trade_start]
            if start_row.empty:
                continue
            open_price = start_row.iloc[0]['시가']

            # 종가 (거래 종료일)
            end_row = df_price[df_price['날짜'] <= trade_end]
            if end_row.empty:
                continue
            close_price = end_row.iloc[-1]['종가']

            if open_price == 0:
                continue

            # 변화율
            change_ratio = (close_price - open_price) / open_price

            # 혼동 레이블 계산
            if score_val == 0:
                confusion = ''
            else:
                pred_sign = 'P' if score_val > 0 else 'N'
                result_sign = 'P' if change_ratio >= 0 else 'N'
                confusion = pred_sign + result_sign

            results.append({
                'trade_start': trade_start.date(),
                'trade_end': trade_end.date(),
                'score': score_val,
                'open_price': open_price,
                'close_price': close_price,
                'change_ratio': round(change_ratio, 4),
                'confusion_label': confusion
            })

        if results:
            df_result = pd.DataFrame(results)
            save_path = os.path.join(save_root, f"{code}_confusion_matrix_score.csv")
            df_result.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"[DONE] {sector} - {code} 저장 완료 → {save_path}")
