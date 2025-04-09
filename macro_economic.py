import pandas as pd
import os

# 파일 경로
output_path = "./data_kr/macro_economic/merged.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# CSV 파일 불러오기
cpi_df = pd.read_csv("./data_kr/macro_economic/consumer_price_index.csv")
exchange_df = pd.read_csv("./data_kr/macro_economic/exchange_rate.csv")
gdp_df = pd.read_csv("./data_kr/macro_economic/GDP.csv")
interest_df = pd.read_csv("./data_kr/macro_economic/interest_rate.csv")
export_df = pd.read_csv("./data_kr/macro_economic/total_export_amount.csv")
unemployment_df = pd.read_csv("./data_kr/macro_economic/unemployment_rate.csv")
oil_df = pd.read_csv("./data_kr/macro_economic/oil_price.csv")
home_df = pd.read_csv("./data_kr/macro_economic/home_price.csv")

# 날짜를 datetime으로 변환
for df in [cpi_df, exchange_df, gdp_df, interest_df, export_df, unemployment_df, oil_df, home_df]:
    df["observation_date"] = pd.to_datetime(df["observation_date"])

# 연도/분기 컬럼 생성 함수
def add_year_quarter(df):
    df["연도"] = df["observation_date"].dt.year
    df["분기"] = "Q" + df["observation_date"].dt.quarter.astype(str)
    return df

# 연도/분기 추가
cpi_df = add_year_quarter(cpi_df)
exchange_df = add_year_quarter(exchange_df)
unemployment_df = add_year_quarter(unemployment_df)
gdp_df = add_year_quarter(gdp_df)
interest_df = add_year_quarter(interest_df)
export_df = add_year_quarter(export_df)
oil_df = add_year_quarter(oil_df)
home_df = add_year_quarter(home_df)

# 월별 데이터는 분기별 평균
def quarterly_avg(df, value_col):
    return df.groupby(["연도", "분기"])[value_col].mean().reset_index()

cpi_q = quarterly_avg(cpi_df, "KORCPALTT01CTGYM").rename(columns={"KORCPALTT01CTGYM": "소비자물가지수"})
exchange_q = quarterly_avg(exchange_df, "EXKOUS").rename(columns={"EXKOUS": "환율"})
unemployment_q = quarterly_avg(unemployment_df, "LRUNTTTTKRM156S").rename(columns={"LRUNTTTTKRM156S": "실업률"})
oil_q = quarterly_avg(oil_df,"MCOILBRENTEU").rename(columns={"MCOILBRENTEU":"국제유가"})

# 분기별 데이터는 그대로
gdp_q = gdp_df[["연도", "분기", "NGDPRSAXDCKRQ"]].rename(columns={"NGDPRSAXDCKRQ": "GDP"})
interest_q = interest_df[["연도", "분기", "IRLTLT01KRQ156N"]].rename(columns={"IRLTLT01KRQ156N": "금리"})
export_q = export_df[["연도", "분기", "XTEXVA01KRQ664S"]].rename(columns={"XTEXVA01KRQ664S": "수출액"})
home_q = home_df[["연도", "분기", "QKRR628BIS"]].rename(columns={"QKRR628BIS": "주택가격"})

# 연도/분기 틀 생성 (2015 Q4 ~ 2024 Q3)
quarters = ["Q1", "Q2", "Q3", "Q4"]
timeline = [(year, q) for year in range(2015, 2025) for q in quarters]
timeline_df = pd.DataFrame(timeline, columns=["연도", "분기"])
timeline_df = timeline_df[~((timeline_df["연도"] == 2015) & (timeline_df["분기"].isin(["Q1", "Q2", "Q3"])))]
timeline_df = timeline_df[~((timeline_df["연도"] == 2024) & (timeline_df["분기"] == "Q4"))]

# 데이터 병합
merged = timeline_df.copy()
merged = merged.merge(cpi_q, on=["연도", "분기"], how="left")
merged = merged.merge(exchange_q, on=["연도", "분기"], how="left")
merged = merged.merge(gdp_q, on=["연도", "분기"], how="left")
merged = merged.merge(interest_q, on=["연도", "분기"], how="left")
merged = merged.merge(export_q, on=["연도", "분기"], how="left")
merged = merged.merge(unemployment_q, on=["연도", "분기"], how="left")
merged = merged.merge(oil_q, on=["연도","분기"],how="left")
merged = merged.merge(home_q, on=["연도","분기"],how="left")

# 저장
merged.to_csv(output_path, index=False, encoding='utf-8-sig')
