import pandas as pd
import os
from datetime import timedelta

def get_disclosure_date(year, quarter, code):
    df = pd.read_csv(f"./data_kr/date_regression/{year}_{quarter}.csv")
    row = df[df['code'].astype(str).str.zfill(6) == code]
    if row.empty:
        return None
    return pd.to_datetime(row.iloc[0]['disclosure_date'])

def get_price(df_price, target_date, kind="open"):
    target_date = pd.to_datetime(target_date)
    if kind == "open":
        while target_date <= df_price['날짜'].max():
            match = df_price[df_price['날짜'] == target_date]
            if not match.empty:
                return match.iloc[0]['시가']
            target_date += timedelta(days=1)
    elif kind == "close":
        while target_date >= df_price['날짜'].min():
            match = df_price[df_price['날짜'] == target_date]
            if not match.empty:
                return match.iloc[0]['종가']
            target_date -= timedelta(days=1)
    return None

def evaluate(pred, y):
    if pd.isna(pred) or pd.isna(y):
        return "excluded"
    pred = pred.strip().lower()
    if pred == 'up':
        return "true" if y > 0 else "false"
    elif pred == 'down':
        return "true" if y < 0 else "false"
    elif pred == 'irrelevant':
        return "excluded"
    return "excluded"

#  경로 설정
llm_paths = {
    "산업재": "./preprocessed_data/llm/predict/산업재/",
    "정보기술": "./preprocessed_data/llm/predict/정보기술/"
}
quarters_path = "./data_kr/date_regression/"
price_path = "./data_kr/price/"
quarters = sorted(os.listdir(quarters_path))

os.makedirs("./preprocessed_data/llm_cmp_result", exist_ok=True)

#  산업별 반복
for sector_name, llm_path in llm_paths.items():
    all_results = []
    summary = []

    for file in os.listdir(llm_path):
        if not file.endswith(".csv"):
            continue

        code = file.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(llm_path, file), encoding='utf-8-sig')
            price_df = pd.read_csv(os.path.join(price_path, f"{code}.csv"), encoding='utf-8-sig')
            price_df['날짜'] = pd.to_datetime(price_df['날짜'])
        except:
            continue

        if 'name' not in df.columns:
            continue

        per_stock_results = []
        for _, row in df.iterrows():
            try:
                year = int(row['year'])
                quarter = row['quarter']
                pred = row['prediction']
                name = row['name']
                disclosure_date = get_disclosure_date(year, quarter, code)
                if disclosure_date is None:
                    continue

                qkey = f"{year}_{quarter}.csv"
                if qkey not in quarters:
                    continue
                idx = quarters.index(qkey)
                if idx + 1 >= len(quarters):
                    continue

                next_q = pd.read_csv(os.path.join(quarters_path, quarters[idx + 1]))
                next_row = next_q[next_q['code'].astype(str).str.zfill(6) == code]
                if next_row.empty:
                    continue
                next_disclosure = pd.to_datetime(next_row.iloc[0]['disclosure_date'])
                end_date = next_disclosure - timedelta(days=1)

                O = get_price(price_df, disclosure_date, kind="open")
                C = get_price(price_df, end_date, kind="close")
                if O is None or C is None or O == 0:
                    continue

                y = (C - O) / O
                judgment = evaluate(pred, y)
                judgment = str(judgment)

                per_stock_results.append({
                    "종목명": name,
                    "code": code,
                    "year": year,
                    "quarter": quarter,
                    "공시일": disclosure_date.date(),
                    "시가": O,
                    "종가": C,
                    "수익률": round(y, 4),
                    "LLM 예측": pred,
                    "판단": str(judgment)
                })
            except:
                continue

        if per_stock_results:
            df_stock = pd.DataFrame(per_stock_results)
            true_count = (df_stock["판단"] == "true").sum()
            excluded_count = (df_stock["판단"] == "excluded").sum()
            total_valid = len(df_stock[df_stock["판단"] != "excluded"])
            accuracy = round((true_count / total_valid) * 100, 2) if total_valid else 0

            summary.append({
                "종목명": df_stock["종목명"].iloc[0],
                "code": code,
                "true": true_count,
                "excluded": excluded_count,
                "total": total_valid,
                "정답률(%)": accuracy
            })

            all_results.extend(per_stock_results)

    # 전체 평균 정확도 (가중 평균)
    df_summary = pd.DataFrame(summary)
    total_true = df_summary["true"].sum()
    total_evaluable = df_summary["total"].sum()
    overall_accuracy = round((total_true / total_evaluable) * 100, 2) if total_evaluable else 0

    # 합계 행 추가
    df_summary.loc["합계"] = {
        "종목명": "합계",
        "code": "",
        "true": total_true,
        "excluded": df_summary["excluded"].sum(),
        "total": total_evaluable,
        "정답률(%)": overall_accuracy
    }

    # 컬럼명을 대문자로 변경
    df_summary.columns = [col.upper() for col in df_summary.columns]
    df_all = pd.DataFrame(all_results)
    df_all.columns = [col.upper() for col in df_all.columns]
    # 결과 저장 (경로 수정됨)
    df_all = pd.DataFrame(all_results)
    df_all.columns = [col.upper() for col in df_all.columns]
    df_summary.columns = [col.upper() for col in df_summary.columns]

    df_all.to_csv(f"./preprocessed_data/llm_cmp_result/llm_result_{sector_name}.csv", index=False, encoding="utf-8-sig")
    df_summary.to_csv(f"./preprocessed_data/llm_cmp_result/llm_summary_{sector_name}.csv", index=False,
                      encoding="utf-8-sig")

    print(f"{sector_name} 저장 완료")
    print(f"- ./preprocessed_data/llm_cmp_result/llm_result_{sector_name}.csv")
    print(f"- ./preprocessed_data/llm_cmp_result/llm_summary_{sector_name}.csv")

