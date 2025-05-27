import os
import pandas as pd
import argparse
from datetime import timedelta

# 분기 맵
QUARTER_MAP = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

# 공시일 가져오는 함수 (기존)
def get_disclosure_date(year, quarter, code):
    df = pd.read_csv(f"./data_kr/date_regression/{year}_{quarter}.csv")
    row = df[df['code'].astype(str).str.zfill(6) == code]
    if row.empty:
        return None
    return pd.to_datetime(row.iloc[0]['disclosure_date'])

# 가격 가져오는 함수 (기존)
from_price_kind = { 'open': '시가', 'close': '종가' }
def get_price(df_price, target_date, kind="open"):
    target_date = pd.to_datetime(target_date)
    price_col = from_price_kind.get(kind, '시가')
    if kind == "open":
        while target_date <= df_price['날짜'].max():
            match = df_price[df_price['날짜'] == target_date]
            if not match.empty:
                return match.iloc[0][price_col]
            target_date += timedelta(days=1)
    else:
        while target_date >= df_price['날짜'].min():
            match = df_price[df_price['날짜'] == target_date]
            if not match.empty:
                return match.iloc[0][price_col]
            target_date -= timedelta(days=1)
    return None

# 예측 판단

def evaluate(pred, y):
    if pd.isna(pred) or pd.isna(y):
        return "excluded"
    p = pred.strip().lower()
    if p == 'up':
        return "true" if y > 0 else "false"
    if p == 'down':
        return "true" if y < 0 else "false"
    return "excluded"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM 예측 결과 통합 분석")
    parser.add_argument('start_year', type=int)
    parser.add_argument('start_quarter', choices=QUARTER_MAP.keys())
    parser.add_argument('end_year', type=int)
    parser.add_argument('end_quarter', choices=QUARTER_MAP.keys())
    args = parser.parse_args()

    # 주어진 기간 계산
    start_period = args.start_year * 4 + QUARTER_MAP[args.start_quarter]
    end_period   = args.end_year   * 4 + QUARTER_MAP[args.end_quarter]

    # LLM 예측 결과 폴더
    llm_paths = {
        "산업재": "./preprocessed_data/llm/predict/산업재/",
        "정보기술": "./preprocessed_data/llm/predict/정보기술/"
    }
    price_path = "./data_kr/price/"
    quarters_path = "./data_kr/date_regression/"
    quarters = sorted(os.listdir(quarters_path))

    all_results = []
    summary_list = []

    # 각 섹터 파일 읽어 통합
    for sector, folder in llm_paths.items():
        for fname in os.listdir(folder):
            if not fname.endswith('.csv'): continue
            code = fname.replace('.csv', '')
            # 예측 CSV 로드
            try:
                df_pred = pd.read_csv(os.path.join(folder, fname), encoding='utf-8-sig')
                df_price = pd.read_csv(os.path.join(price_path, f"{code}.csv"), encoding='utf-8-sig')
                df_price['날짜'] = pd.to_datetime(df_price['날짜'])
            except:
                continue
            if 'year' not in df_pred.columns or 'quarter' not in df_pred.columns:
                continue

            per_stock = []
            # 종목별 각 행 처리
            for _, r in df_pred.iterrows():
                y = r.get('year'); q = r.get('quarter')
                if pd.isna(y) or q not in QUARTER_MAP: continue
                period = int(y) * 4 + QUARTER_MAP[q]
                # 기간 필터
                if period < start_period or period > end_period:
                    continue
                code_str = str(code).zfill(6)
                disc = get_disclosure_date(int(y), q, code_str)
                if disc is None: continue
                # 다음 분기
                key = f"{int(y)}_{q}.csv"
                if key not in quarters: continue
                idx = quarters.index(key)
                if idx+1 >= len(quarters): continue
                next_df = pd.read_csv(os.path.join(quarters_path, quarters[idx+1]))
                nr = next_df[next_df['code'].astype(str).str.zfill(6)==code_str]
                if nr.empty: continue
                next_disc = pd.to_datetime(nr.iloc[0]['disclosure_date'])
                end_date = next_disc - timedelta(days=1)

                O = get_price(df_price, disc, kind='open')
                C = get_price(df_price, end_date, kind='close')
                if O is None or C is None or O == 0: continue
                ret = (C - O) / O
                j = evaluate(r.get('prediction',''), ret)

                per_stock.append({
                    '종목명': r.get('name',''), 'code': code_str,
                    'year': int(y), 'quarter': q,
                    '공시일': disc.date(), '시가': O, '종가': C,
                    '수익률': round(ret,4), 'LLM 예측': r.get('prediction',''), '판단': j
                })
            if not per_stock: continue
            df_s = pd.DataFrame(per_stock)
            true_n = (df_s['판단']=='true').sum()
            excl_n = (df_s['판단']=='excluded').sum()
            valid_n= len(df_s[df_s['판단']!='excluded'])
            acc = round(true_n/valid_n*100,2) if valid_n else 0
            summary_list.append({'종목명': df_s['종목명'].iloc[0], 'code': code_str,
                                 'true': true_n, 'excluded': excl_n,
                                 'total': valid_n, '정답률(%)': acc})
            all_results.extend(per_stock)

    # 통합 결과 DataFrame
    df_all = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(summary_list)
    # 전체 accuracy
    tot_true = df_summary['true'].sum()
    tot_valid= df_summary['total'].sum()
    overall = round(tot_true/tot_valid*100,2) if tot_valid else 0
    df_summary.loc['합계'] = ['합계','', tot_true, df_summary['excluded'].sum(), tot_valid, overall]

    # 파일명 및 저장 경로
    out_dir = "./preprocessed_data/llm_cmp_result"
    os.makedirs(out_dir, exist_ok=True)
    start_q = args.start_quarter; end_q = args.end_quarter
    base = f"{args.start_year}_{start_q}_to_{args.end_year}_{end_q}"
    result_fp = os.path.join(out_dir, f"llm_result_total_{base}.csv")
    summary_fp= os.path.join(out_dir, f"llm_summary_total_{base}.csv")

    df_all.to_csv(result_fp, index=False, encoding='utf-8-sig')
    df_summary.to_csv(summary_fp, index=False, encoding='utf-8-sig')
    print(f"저장 완료: {result_fp}\n         {summary_fp}")

