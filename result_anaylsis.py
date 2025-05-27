import pandas as pd
from datetime import timedelta

def print_disclosure_date_range(code: str, year: int, quarter: str, base_path: str = "./data_kr/merged"):
    """
    종목코드(code), 연도(year), 분기(quarter)에 해당하는 공시일과
    다음 분기의 공시일 범위를 출력합니다.

    예시 출력:
    000120, 2024 Q2의 공시일 범위: 2024-05-15 ~ 2024-08-16
    """
    # 1) CSV 읽어오기 & 날짜 파싱
    filepath = f"{base_path}/{code}.csv"
    df = pd.read_csv(filepath, parse_dates=["disclosure_date"])

    # 2) 분기 숫자로 매핑
    quarter_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    if quarter not in quarter_map:
        raise ValueError(f"quarter는 Q1, Q2, Q3, Q4 중 하나여야 합니다. 입력값: {quarter}")

    # 3) 현재 분기 공시일 찾기
    cur_mask = (df["year"] == year) & (df["quarter"] == quarter)
    if not cur_mask.any():
        raise ValueError(f"{year}년 {quarter} 데이터가 없습니다. 파일을 확인해주세요.")
    cur_date = df.loc[cur_mask, "disclosure_date"].iloc[0]

    # 4) 다음 분기 계산
    qnum = quarter_map[quarter]
    if qnum == 4:
        next_year, next_q = year + 1, "Q1"
    else:
        next_year, next_q = year, f"Q{qnum + 1}"

    # 5) 다음 분기 공시일 찾기
    next_mask = (df["year"] == next_year) & (df["quarter"] == next_q)
    if not next_mask.any():
        raise ValueError(f"다음 분기({next_year}년 {next_q}) 데이터가 없습니다.")
    next_date = df.loc[next_mask, "disclosure_date"].iloc[0]
    next_date = next_date - timedelta(days=1)
    # 6) 결과 출력
    print(f"{code}, {year} {quarter}의 공시일 범위: "
          f"{cur_date.strftime('%Y-%m-%d')} ~ {next_date.strftime('%Y-%m-%d')}")

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import timedelta

# ——— 한글 폰트 설정 ———
mpl.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False

def plot_and_save_close_price(code: str,
                              year: int,
                              quarter: str,
                              merged_path: str = "./data_kr/merged",
                              price_path: str = "./data_kr/price",
                              result_dir: str = "./result/result_analysis"):
    """
    1) merged/{code}.csv 에서 year, quarter 의 공시일 찾기 → start_date
    2) 다음 분기 공시일 전날 → end_date
    3) price/{code}.csv 에서 start_date~end_date 구간 '종가' 추출
    4) 각 포인트에 날짜 텍스트 달아서 그래프 그리기
    5) result_dir/{code}_{year}_{quarter}_price.csv 로 저장
    6) result_dir/{code}_{year}_{quarter}_plot.png 로 그래프 저장
    """
    # ── 1) 공시일 범위 계산 ──
    df_m = pd.read_csv(os.path.join(merged_path, f"{code}.csv"),
                       parse_dates=["disclosure_date"])
    q_map = {"Q1":1, "Q2":2, "Q3":3, "Q4":4}
    if quarter not in q_map:
        raise ValueError("quarter는 Q1, Q2, Q3, Q4 중 하나여야 합니다.")
    cur = df_m[(df_m.year==year)&(df_m.quarter==quarter)]
    if cur.empty:
        raise ValueError(f"{year}년 {quarter} 데이터가 없습니다.")
    start_date = cur.disclosure_date.iloc[0]
    qn = q_map[quarter]
    ny, nq = (year+1, "Q1") if qn==4 else (year, f"Q{qn+1}")
    nxt = df_m[(df_m.year==ny)&(df_m.quarter==nq)]
    if nxt.empty:
        raise ValueError(f"다음 분기({ny}년 {nq}) 데이터가 없습니다.")
    end_date = nxt.disclosure_date.iloc[0] - timedelta(days=1)

    # ── 2) 가격 데이터 읽기 & 필터링 ──
    df_p = pd.read_csv(os.path.join(price_path, f"{code}.csv"))
    date_candidates = ["날짜", "date", "Date", "일자"]
    date_col = next((c for c in date_candidates if c in df_p.columns), None)
    if date_col is None:
        raise ValueError(f"{code}.csv에 날짜 컬럼이 없습니다.")
    df_p[date_col] = pd.to_datetime(df_p[date_col])
    mask = (df_p[date_col] >= start_date) & (df_p[date_col] <= end_date)
    df_range = df_p.loc[mask, [date_col, "종가"]]
    if df_range.empty:
        print("해당 기간에 price 데이터가 없습니다.")
        return

    # ── 3) 그래프 그리기 (포인트 위에 날짜 표시) ──
    plt.figure(figsize=(10,4))
    plt.plot(df_range[date_col], df_range["종가"], marker="o")

    from adjustText import adjust_text

    texts = []
    for x, y in zip(df_range[date_col], df_range["종가"]):
        texts.append(
            plt.text(
                x, y,
                x.strftime("%Y-%m-%d"),
                fontsize=8
            )
        )
    adjust_text(texts,
                only_move={'points': 'y', 'texts': 'y'},
                arrowprops=dict(arrowstyle='-', color='gray'))

    title_str = f"{code}_{year}_{quarter}: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
    plt.title(title_str)
    plt.xlabel("Date")
    plt.ylabel("종가")
    plt.xlim(start_date, end_date)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # ── 4) 그래프 파일로 저장 ──
    os.makedirs(result_dir, exist_ok=True)
    plot_path = os.path.join(result_dir, f"{code}_{year}_{quarter}_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()
    print(f"그래프 저장 완료: {plot_path}")

    # ── 5) CSV 파일로 저장 ──
    csv_path = os.path.join(result_dir, f"{code}_{year}_{quarter}_price.csv")
    df_range.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"가격 데이터 CSV 저장 완료: {csv_path}")

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from datetime import timedelta
import mplcursors

# — 한글 폰트 설정 (Windows) —
mpl.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False

def plot_and_save_close_price(code: str,
                              year: int,
                              quarter: str,
                              merged_path: str = "./data_kr/merged",
                              price_path: str = "./data_kr/price",
                              result_dir: str = "./result/result_analysis"):
    # 1) 공시일 범위 계산
    df_m = pd.read_csv(f"{merged_path}/{code}.csv", parse_dates=["disclosure_date"])
    q_map = {"Q1":1, "Q2":2, "Q3":3, "Q4":4}
    if quarter not in q_map:
        raise ValueError("quarter는 Q1, Q2, Q3, Q4 중 하나여야 합니다.")
    cur = df_m[(df_m.year==year)&(df_m.quarter==quarter)]
    if cur.empty:
        raise ValueError(f"{year}년 {quarter} 데이터가 없습니다.")
    start_date = cur.disclosure_date.iloc[0]
    qn = q_map[quarter]
    ny, nq = (year+1, "Q1") if qn==4 else (year, f"Q{qn+1}")
    nxt = df_m[(df_m.year==ny)&(df_m.quarter==nq)]
    if nxt.empty:
        raise ValueError(f"다음 분기({ny}년 {nq}) 데이터가 없습니다.")
    end_date = nxt.disclosure_date.iloc[0] - timedelta(days=1)

    # 2) 가격 데이터 읽기 & 필터링
    df_p = pd.read_csv(f"{price_path}/{code}.csv")
    date_candidates = ["날짜","date","Date","일자"]
    date_col = next((c for c in date_candidates if c in df_p.columns), None)
    if date_col is None:
        raise ValueError(f"{code}.csv에 날짜 컬럼이 없습니다.")
    df_p[date_col] = pd.to_datetime(df_p[date_col])
    mask = (df_p[date_col]>=start_date)&(df_p[date_col]<=end_date)
    df_range = df_p.loc[mask, [date_col, "종가"]]
    if df_range.empty:
        print("해당 기간에 price 데이터가 없습니다.")
        return

    # 3) price CSV 저장
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, f"{code}_{year}_{quarter}_price.csv")
    df_range.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 저장: {csv_path}")

    # 4) 그래프 그리기
    fig, ax = plt.subplots(figsize=(10,4))
    line, = ax.plot(df_range[date_col], df_range["종가"], marker="o", linestyle='-')

    # x축 마진 제거 및 시작·끝일 눈금 추가
    plt.margins(x=0)
    orig_locs = ax.get_xticks().tolist()
    start_num = mdates.date2num(start_date)
    end_num   = mdates.date2num(end_date)
    new_locs  = [start_num] + orig_locs + [end_num]
    ax.set_xticks(new_locs)
    new_labels = (
        [start_date.strftime("%Y-%m-%d")] +
        [mdates.num2date(num).strftime("%Y-%m-%d") for num in orig_locs] +
        [end_date.strftime("%Y-%m-%d")]
    )
    ax.set_xticklabels(new_labels, rotation=45, ha='right')

    title_str = f"{code}_{year}_{quarter}: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
    ax.set_title(title_str)
    ax.set_xlabel("Date")
    ax.set_ylabel("종가")

    # 5) mplcursors로 마우스 오버 툴팁 활성화
    mplcursors.cursor(line, hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(
            df_range[date_col].iloc[sel.target.index].strftime("%Y-%m-%d")
        )
    )

    fig.tight_layout()
    # 6) 그래프 PNG 저장
    png_path = os.path.join(result_dir, f"{code}_{year}_{quarter}_plot.png")
    fig.savefig(png_path, dpi=300)
    print(f"Plot 저장: {png_path}")

    #plt.show()

if __name__ == "__main__":
    plot_and_save_close_price("047050", 2022, "Q4")
    plot_and_save_close_price("003570", 2022, "Q4")
    plot_and_save_close_price("006260", 2023, "Q1")
    plot_and_save_close_price("042700", 2023, "Q2")
    plot_and_save_close_price("001120", 2023, "Q3")
    plot_and_save_close_price("006260", 2023, "Q3")
