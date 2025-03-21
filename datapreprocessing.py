import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
from datetime import datetime, timedelta
from fancyimpute import SoftImpute
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import argparse
parser = argparse.ArgumentParser() # 입력 받을 하이퍼파라미터 설정

parser.add_argument('--isall',type=str,nargs='?',default="False")
args = parser.parse_args()

ks200_price = pd.read_csv("./data_kr/price/KS200.csv")
df_col = {}

#imputer = SoftImpute(verbose=False)
imputer = SimpleImputer(strategy='median')
#imputer = KNNImputer(n_neighbors=5)

#impute = SoftImpute(verbose=False)
impute = SimpleImputer(strategy='median')
#impute = KNNImputer(n_neighbors=5)

def quarter2date(year, quarter):
    if quarter == "Q1":
        start_date_str = f"{year}-01-01"
    elif quarter == "Q2":
        start_date_str = f"{year}-04-01"
    elif quarter == "Q3":
        start_date_str = f"{year}-07-01"
    elif quarter == "Q4":
        start_date_str = f"{year}-10-01"
    else:
        raise ValueError("쿼터는 'Q1', 'Q2', 'Q3', 'Q4' 중 하나여야 합니다.")

    quarter_start = datetime.strptime(start_date_str, "%Y-%m-%d")

    return quarter_start
def get_start_price(year, quarter, ticker):
    """
    지정한 연도와 쿼터의 시작일 이후 가장 가까운 거래일의 시가(Open)를 반환합니다.
    해당 날짜에 가격 데이터가 없으면, 데이터가 있을 때까지 날짜를 1일씩 증가시킵니다.

    예시:
      year=2015, quarter="Q1"이면 기준일은 "2015-01-01"이며,
      그 이후의 첫 거래일의 Open 가격을 반환합니다.

    :param year: int, 연도 (예: 2015)
    :param quarter: str, 쿼터 ("Q1", "Q2", "Q3", "Q4")
    :param ticker: str, 티커명 (예: "AAPL")
    :return: 거래일의 시가 (float)
    """
    # 쿼터 시작일 설정
    if quarter == "Q1":
        start_date_str = f"{year}-01-01"
    elif quarter == "Q2":
        start_date_str = f"{year}-04-01"
    elif quarter == "Q3":
        start_date_str = f"{year}-07-01"
    elif quarter == "Q4":
        start_date_str = f"{year}-10-01"
    else:
        raise ValueError("쿼터는 'Q1', 'Q2', 'Q3', 'Q4' 중 하나여야 합니다.")

    quarter_start = datetime.strptime(start_date_str, "%Y-%m-%d")

    # CSV 파일 읽기
    file_path = f"./data_kr/price/{ticker}.csv"
    df = pd.read_csv(file_path)

    # Date 컬럼을 datetime 형식으로 변환 후 정렬
    df['날짜'] = pd.to_datetime(df['날짜'])
    df.sort_values(by='날짜', inplace=True)

    # 기준일 이후에 데이터가 존재할 때까지 날짜를 증가시킴
    current_date = quarter_start
    while True:
        # 날짜만 비교하기 위해 .dt.date 사용
        matching_rows = df[df['날짜'].dt.date == current_date.date()]
        if not matching_rows.empty:
            # 해당 날짜의 첫 거래일의 시가 반환
            return matching_rows.iloc[0]['시가']
        current_date += timedelta(days=1)
        # 데이터 범위를 넘어가는 경우 예외 발생
        if current_date > df['날짜'].max():
            print(f"{ticker}의 데이터에서 {quarter_start} 이후의 거래일을 찾을 수 없습니다.")
            return 0


def get_end_price(year, quarter, ticker):
    """
    지정한 연도와 쿼터의 시작일 이전 가장 가까운 거래일의 종가(Close)를 반환합니다.
    해당 날짜에 가격 데이터가 없으면, 데이터가 있을 때까지 날짜를 1일씩 감소시킵니다.

    예시:
      year=2015, quarter="Q1"이면 기준일은 "2015-01-01"이며,
      그 이전의 마지막 거래일의 Close 가격을 반환합니다.

    :param year: int, 연도 (예: 2015)
    :param quarter: str, 쿼터 ("Q1", "Q2", "Q3", "Q4")
    :param ticker: str, 티커명 (예: "AAPL")
    :return: 거래일의 종가 (float)
    """
    # 쿼터 시작일 설정
    if quarter == "Q1":
        start_date_str = f"{year}-01-01"
    elif quarter == "Q2":
        start_date_str = f"{year}-04-01"
    elif quarter == "Q3":
        start_date_str = f"{year}-07-01"
    elif quarter == "Q4":
        start_date_str = f"{year}-10-01"
    else:
        raise ValueError("쿼터는 'Q1', 'Q2', 'Q3', 'Q4' 중 하나여야 합니다.")

    quarter_start = datetime.strptime(start_date_str, "%Y-%m-%d")

    # CSV 파일 읽기
    file_path = f"./data_kr/price/{ticker}.csv"
    df = pd.read_csv(file_path)

    # Date 컬럼을 datetime 형식으로 변환 후 정렬
    df['날짜'] = pd.to_datetime(df['날짜'])

    # 쿼터 시작일 이전의 거래일을 찾기 위해, 기준일의 전날부터 시작
    current_date = quarter_start - timedelta(days=1)
    while True:
        matching_rows = df[df['날짜'].dt.date == current_date.date()]
        if not matching_rows.empty:
            # 해당 날짜의 마지막 거래일의 종가 반환
            return matching_rows.iloc[-1]['종가']
        current_date -= timedelta(days=1)
        # 데이터 범위를 넘어가는 경우 예외 발생
        if current_date < df['날짜'].min():
            print(f"{ticker}의 데이터에서 {quarter_start} 이전의 거래일을 찾을 수 없습니다.")
            return 0


if args.isall == "False":
    cluster_list = [['Automotive_&_Parts', 'Construction_&_Real_Estate', 'Consumer_Services', 'Trading_&_Logistics', 'Transportation'], ['Consumer_Goods_&_Retail', 'Financials', 'Healthcare', 'Industrials_&_Machinery', 'Telecommunications_&_Media'], ['Materials_&_Chemicals', 'Technology', 'Energy_&_Utilities'], ['Agriculture_&_Food', 'Conglomerate']]
    for cluster_index in range(4):
        sector_list = cluster_list[cluster_index]
        clustered_ticker_list = []

        df_sector_stocks = pd.DataFrame()
        df_processing_data = pd.DataFrame()

        for sector in sector_list:
            ticker_list = pd.read_csv(f'./data_kr/date_sector/{sector}/sector_code.csv')
            clustered_ticker_list.extend(ticker_list["code"].to_list())

            for ticker in ticker_list["code"]:
                if ticker == "code":
                    continue
                ticker_str = str(ticker).zfill(6)
                df_price = pd.read_csv(f"./data_kr/price/{ticker_str}.csv")
                df_tmp = pd.read_csv(f"./data_kr/merged/{ticker_str}.csv")
                df_sector_stocks = pd.concat([df_sector_stocks, df_tmp])

        nan_ratio = df_sector_stocks.isna().mean()
        over_50_columns = nan_ratio[nan_ratio >= 0.5].index.to_list()
        for ticker in tqdm(clustered_ticker_list):
            if ticker == "code":
                continue
            ticker_str = str(ticker).zfill(6)
            df_data = pd.read_csv(f"./data_kr/merged/{ticker_str}.csv", index_col=[0])
            df_data.drop(over_50_columns, axis=1, inplace=True, errors="ignore")
            ###
            for col in df_data.columns[4:]:
                df_data[col] = df_data[col].astype(str).str.replace(',', '')
                df_data[col] = df_data[col].replace('-', np.nan).astype(float)
            ###
            df_data.iloc[:, 4:] = pd.DataFrame(impute.fit_transform(df_data.iloc[:, 4:]))

            df_pct_change = (-df_data.iloc[:, 4:].pct_change(fill_method=None))
            df_pct_change.columns = [col + "변화율" for col in df_pct_change.columns]
            df_data = pd.concat([df_data.iloc[:, :4], df_pct_change], axis=1)
            df_data.fillna(0.0, inplace=True)
            df_data.replace([-np.inf, np.inf], 0.0, inplace=True)

            # df_data = df_data[~df_data['year'].str.contains('2015')&~df_data['quarter'].str.contains('Q4')]

            df_data = df_data[
                ~(df_data['year'].astype(str).str.contains('2015') & df_data['quarter'].astype(str).str.contains('Q4'))
            ]

            # 시장초과수익률 계산
            df_price = pd.read_csv(f"./data_kr/price/{ticker_str}.csv", index_col=[0])
            stock_date = datetime.strptime(df_price.index[0], "%Y-%m-%d")
            past_return = []
            df_data = df_data.iloc[::-1].reset_index(drop=True)
            for i in range(0, len(df_data) - 1):

                start_price = get_start_price(df_data.iloc[i + 1, 2], df_data.iloc[i + 1, 3], ticker_str)
                end_price = get_end_price(df_data.iloc[i, 2], df_data.iloc[i, 3], ticker_str)

                ks200_start_price = get_start_price(df_data.iloc[i + 1, 2], df_data.iloc[i + 1, 3], "KS200")
                ks200_end_price = get_end_price(df_data.iloc[i, 2], df_data.iloc[i, 3], "KS200")

                if start_price == 0 or ks200_start_price == 0:
                    relative_return = 0.0
                else:
                    relative_return = (end_price - start_price) / start_price - (
                                ks200_end_price - ks200_start_price) / ks200_start_price

                past_return.append(relative_return)

            past_return.append(0)
            df_data["Relative Return"] = past_return

            df_price = pd.read_csv(f"./data_kr/price/{ticker_str}.csv", index_col=[0])
            df_ks200 = pd.read_csv(f"./data_kr/price/KS200.csv", index_col=[0])
            stock_date = datetime.strptime(df_price.index[0], "%Y-%m-%d")

            label = []
            start_date = '2024-07-01'
            end_date = '2024-09-30'

            start_price = get_start_price(df_data.iloc[1, 2], df_data.iloc[1, 3], ticker_str)
            end_price = get_end_price(df_data.iloc[0, 2], df_data.iloc[0, 3], ticker_str)

            ks200_start_price = get_start_price(df_data.iloc[1, 2], df_data.iloc[1, 3], "KS200")
            ks200_end_price = get_end_price(df_data.iloc[0, 2], df_data.iloc[0, 3], "KS200")
            if start_price == 0 or ks200_start_price == 0:
                relative_return = 0.0
            else:
                relative_return = (end_price - start_price) / start_price - (
                        ks200_end_price - ks200_start_price) / ks200_start_price

            label.append(relative_return)

            for i in range(1, len(df_data)):
                start_year = df_data.iloc[i, 2]
                end_year = df_data.iloc[i - 1, 2]
                start_quarter = df_data.iloc[i, 3]
                end_quarter = df_data.iloc[i - 1, 3]

                start_date = quarter2date(start_year, start_quarter)
                if start_date < stock_date:
                    label.append(0.0)
                    continue
                start_price = get_start_price(df_data.iloc[i, 2], df_data.iloc[i, 3], ticker_str)
                end_price = get_end_price(df_data.iloc[i - 1, 2], df_data.iloc[i - 1, 3], ticker_str)

                ks200_start_price = get_start_price(df_data.iloc[i, 2], df_data.iloc[i, 3], "KS200")
                ks200_end_price = get_end_price(df_data.iloc[i - 1, 2], df_data.iloc[i - 1, 3], "KS200")
                if start_price == 0 or ks200_start_price == 0:
                    relative_return = 0.0
                else:
                    relative_return = (end_price - start_price) / start_price - (
                                ks200_end_price - ks200_start_price) / ks200_start_price

                label.append(relative_return)

            df_data["Label"] = label
            df_data["Code"] = ticker_str

            if df_processing_data.empty:
                df_processing_data = df_data
            else:
                df_processing_data = pd.concat([df_processing_data, df_data], axis=0)

        if os.path.isdir(f"./data_kr/clustered_data/cluster_{cluster_index}") == False:
            os.mkdir(f"./data_kr/clustered_data/cluster_{cluster_index}")
        df_processing_data.to_csv(f"./data_kr/clustered_data/cluster_{cluster_index}/cluster_check.csv",
                                  encoding='utf-8-sig')

        # df_processing_data.iloc[:, 4:-2] = df_processing_data.iloc[:, 4:-2].replace([-np.inf, np.inf], 0.0)
        # df_processing_data.iloc[:, -2:-1] = df_processing_data.iloc[:, -2:-1].replace([-np.inf, np.inf], 0.0)

        # Feature Matrix (X) Imputation
        X_imputed = imputer.fit_transform(df_processing_data.iloc[:, 4:-2])
        df_processing_data.iloc[:, 4:-2] = pd.DataFrame(X_imputed, columns=df_processing_data.columns[4:-2],
                                                        index=df_processing_data.index)

        # Target (y) Imputation
        y_imputed = imputer.fit_transform(df_processing_data.iloc[:, -2:-1])
        df_processing_data.iloc[:, -2:-1] = pd.DataFrame(y_imputed, columns=df_processing_data.columns[-2:-1],
                                                         index=df_processing_data.index)

        rgr = RandomForestRegressor()
        rgr.fit(df_processing_data.iloc[:, 4:-2], df_processing_data.iloc[:, -2:-1])

        feature_importance = pd.Series(rgr.feature_importances_, index=df_processing_data.columns[4:-2]).sort_values(
            ascending=False)
        feature_importance[:6].to_csv(
            f"./data_kr/clustered_data/cluster_{cluster_index}/cluster_{cluster_index}_feature_imp.csv",
            encoding='utf-8-sig')

        select_col = feature_importance.index[:6]
        df_col[sector] = select_col

        df_processed_data = pd.concat(
            [df_processing_data.iloc[:, :4], df_processing_data[select_col], df_processing_data.iloc[:, -2:]], axis=1)

        start_year = 2015
        start_quarter = 4
        end_year = 2024
        end_quarter = 3

        year = start_year
        quarter = start_quarter

        while True:
            quarter_str = f"Q{quarter}"
            df_date_filtered = df_processed_data[
                df_processed_data['year'].astype(str).str.contains(str(year)) &
                df_processed_data['quarter'].astype(str).str.contains(quarter_str)
                ]
            symbol_index = df_date_filtered["Code"].sort_values()
            df_date_filtered = df_date_filtered.sort_values(by="Code")
            df_date_filtered.to_csv(f"./data_kr/clustered_data/cluster_{cluster_index}/{year}_{quarter_str}.csv",
                                    encoding='utf-8-sig')
            if year == end_year and quarter == end_quarter:
                break
            quarter += 1
            if quarter > 4:
                quarter = 1
                year += 1
        symbol_index.to_csv(f"./data_kr/clustered_data/cluster_{cluster_index}/symbol_index.csv", encoding='utf-8-sig')

    exit()
elif args.isall == "all":
    cluster_list = [
        ['Agriculture_&_Food'], ['Consumer_Goods_&_Retail'], ['Consumer_Services'], ['Healthcare'],
         ['Automotive_&_Parts'], ['Construction_&_Real_Estate'], ['Industrials_&_Machinery'], ['Materials_&_Chemicals'],
         ['Transportation'], ['Financials'], ['Energy_&_Utilities'], ['Trading_&_Logistics'], ['Conglomerate'],
         ['Technology'], ['Telecommunications_&_Media']
    ]
    for cluster_index in range(15):
        sector_list = cluster_list[cluster_index]
        clustered_ticker_list = []

        df_sector_stocks = pd.DataFrame()
        df_processing_data = pd.DataFrame()

        for sector in sector_list:
            if isinstance(sector, list):  # sector가 리스트라면
                sector = sector[0]  # 리스트의 첫 번째 요소(문자열)만 사용
            ticker_list = pd.read_csv(f'./data_kr/date_sector/{sector}/sector_code.csv')
            clustered_ticker_list.extend(ticker_list["code"].to_list())

            for ticker in ticker_list["code"]:
                if ticker == "code":
                    continue
                ticker_str = str(ticker).zfill(6)
                df_price = pd.read_csv(f"./data_kr/price/{ticker_str}.csv")
                df_tmp = pd.read_csv(f"./data_kr/merged/{ticker_str}.csv")
                df_sector_stocks = pd.concat([df_sector_stocks, df_tmp])

        nan_ratio = df_sector_stocks.isna().mean()
        over_50_columns = nan_ratio[nan_ratio >= 0.5].index.to_list()
        for ticker in tqdm(clustered_ticker_list):
            if ticker == "code":
                continue
            ticker_str = str(ticker).zfill(6)
            df_data = pd.read_csv(f"./data_kr/merged/{ticker_str}.csv", index_col=[0])
            df_data.drop(over_50_columns, axis=1, inplace=True, errors="ignore")
            ###
            for col in df_data.columns[4:]:
                df_data[col] = df_data[col].astype(str).str.replace(',', '')
                df_data[col] = df_data[col].replace('-', np.nan).astype(float)
            ###
            df_data.iloc[:, 4:] = pd.DataFrame(impute.fit_transform(df_data.iloc[:, 4:]))

            df_pct_change = (-df_data.iloc[:, 4:].pct_change(fill_method=None))
            df_pct_change.columns = [col + "변화율" for col in df_pct_change.columns]
            df_data = pd.concat([df_data.iloc[:, :4], df_pct_change], axis=1)
            df_data.fillna(0.0, inplace=True)
            df_data.replace([-np.inf, np.inf], 0.0, inplace=True)

            # df_data = df_data[~df_data['year'].str.contains('2015')&~df_data['quarter'].str.contains('Q4')]

            df_data = df_data[
                ~(df_data['year'].astype(str).str.contains('2015') & df_data['quarter'].astype(str).str.contains('Q4'))
            ]

            # 시장초과수익률 계산
            df_price = pd.read_csv(f"./data_kr/price/{ticker_str}.csv", index_col=[0])
            stock_date = datetime.strptime(df_price.index[0], "%Y-%m-%d")
            past_return = []
            df_data = df_data.iloc[::-1].reset_index(drop=True)
            for i in range(0, len(df_data) - 1):

                start_price = get_start_price(df_data.iloc[i + 1, 2], df_data.iloc[i + 1, 3], ticker_str)
                end_price = get_end_price(df_data.iloc[i, 2], df_data.iloc[i, 3], ticker_str)

                ks200_start_price = get_start_price(df_data.iloc[i + 1, 2], df_data.iloc[i + 1, 3], "KS200")
                ks200_end_price = get_end_price(df_data.iloc[i, 2], df_data.iloc[i, 3], "KS200")

                if start_price == 0 or ks200_start_price == 0:
                    relative_return = 0.0
                else:
                    relative_return = (end_price - start_price) / start_price - (
                            ks200_end_price - ks200_start_price) / ks200_start_price

                past_return.append(relative_return)

            past_return.append(0)
            #df_data["Relative Return"] = past_return

            df_price = pd.read_csv(f"./data_kr/price/{ticker_str}.csv", index_col=[0])
            df_ks200 = pd.read_csv(f"./data_kr/price/KS200.csv", index_col=[0])
            stock_date = datetime.strptime(df_price.index[0], "%Y-%m-%d")

            label = []
            start_date = '2024-07-01'
            end_date = '2024-09-30'

            start_price = get_start_price(df_data.iloc[1, 2], df_data.iloc[1, 3], ticker_str)
            end_price = get_end_price(df_data.iloc[0, 2], df_data.iloc[0, 3], ticker_str)

            ks200_start_price = get_start_price(df_data.iloc[1, 2], df_data.iloc[1, 3], "KS200")
            ks200_end_price = get_end_price(df_data.iloc[0, 2], df_data.iloc[0, 3], "KS200")
            if start_price == 0 or ks200_start_price == 0:
                relative_return = 0.0
            else:
                relative_return = (end_price - start_price) / start_price - (
                        ks200_end_price - ks200_start_price) / ks200_start_price

            label.append(relative_return)

            for i in range(1, len(df_data)):
                start_year = df_data.iloc[i, 2]
                end_year = df_data.iloc[i - 1, 2]
                start_quarter = df_data.iloc[i, 3]
                end_quarter = df_data.iloc[i - 1, 3]

                start_date = quarter2date(start_year, start_quarter)
                if start_date < stock_date:
                    label.append(0.0)
                    continue
                start_price = get_start_price(df_data.iloc[i, 2], df_data.iloc[i, 3], ticker_str)
                end_price = get_end_price(df_data.iloc[i - 1, 2], df_data.iloc[i - 1, 3], ticker_str)

                ks200_start_price = get_start_price(df_data.iloc[i, 2], df_data.iloc[i, 3], "KS200")
                ks200_end_price = get_end_price(df_data.iloc[i - 1, 2], df_data.iloc[i - 1, 3], "KS200")
                if start_price == 0 or ks200_start_price == 0:
                    relative_return = 0.0
                else:
                    relative_return = (end_price - start_price) / start_price - (
                            ks200_end_price - ks200_start_price) / ks200_start_price

                label.append(relative_return)

            df_data["Label"] = label
            df_data["Code"] = ticker_str

            if df_processing_data.empty:
                df_processing_data = df_data
            else:
                df_processing_data = pd.concat([df_processing_data, df_data], axis=0)

        if os.path.isdir(f"./data_kr/financial_with_Label/{sector_list[0]}") == False:
            os.mkdir(f"./data_kr/financial_with_Label/{sector_list[0]}")
        df_processing_data.to_csv(f"./data_kr/financial_with_Label/{sector_list[0]}/cluster_check.csv",
                                  encoding='utf-8-sig')

        # df_processing_data.iloc[:, 4:-2] = df_processing_data.iloc[:, 4:-2].replace([-np.inf, np.inf], 0.0)
        # df_processing_data.iloc[:, -2:-1] = df_processing_data.iloc[:, -2:-1].replace([-np.inf, np.inf], 0.0)


        # Feature Matrix (X) Imputation
        X_imputed = imputer.fit_transform(df_processing_data.iloc[:, 4:-2])
        df_processing_data.iloc[:, 4:-2] = pd.DataFrame(X_imputed, columns=df_processing_data.columns[4:-2],
                                                        index=df_processing_data.index)

        # Target (y) Imputation
        y_imputed = imputer.fit_transform(df_processing_data.iloc[:, -2:-1])
        df_processing_data.iloc[:, -2:-1] = pd.DataFrame(y_imputed, columns=df_processing_data.columns[-2:-1],
                                                         index=df_processing_data.index)

        rgr = RandomForestRegressor()
        rgr.fit(df_processing_data.iloc[:, 4:-2], df_processing_data.iloc[:, -2:-1])

        feature_importance = pd.Series(rgr.feature_importances_, index=df_processing_data.columns[4:-2]).sort_values(
            ascending=False)
        feature_importance[:3].to_csv(
            f"./data_kr/financial_with_Label/{sector_list[0]}/{cluster_index}_feature_imp.csv",
            encoding='utf-8-sig')

        select_col = feature_importance.index[:3]
        df_col[sector] = select_col

        df_processed_data = pd.concat(
            [df_processing_data.iloc[:, :4], df_processing_data[select_col], df_processing_data.iloc[:, -1:]], axis=1)

        start_year = 2015
        start_quarter = 4
        end_year = 2024
        end_quarter = 3

        year = start_year
        quarter = start_quarter

        while True:
            quarter_str = f"Q{quarter}"
            df_date_filtered = df_processed_data[
                df_processed_data['year'].astype(str).str.contains(str(year)) &
                df_processed_data['quarter'].astype(str).str.contains(quarter_str)
                ]
            symbol_index = df_date_filtered["Code"].sort_values()
            df_date_filtered = df_date_filtered.sort_values(by="Code")
            df_date_filtered.to_csv(f"./data_kr/financial_with_Label/{sector_list[0]}/{year}_{quarter_str}.csv",
                                    encoding='utf-8-sig')
            if year == end_year and quarter == end_quarter:
                break
            quarter += 1
            if quarter > 4:
                quarter = 1
                year += 1
        symbol_index.to_csv(f"./data_kr/financial_with_Label/{sector_list[0]}//symbol_index.csv", encoding='utf-8-sig')

    exit()
else:
    cluster_list = [
        ['Agriculture_&_Food', 'Consumer_Goods_&_Retail', 'Consumer_Services', 'Healthcare',
        'Automotive_&_Parts', 'Construction_&_Real_Estate', 'Industrials_&_Machinery', 'Materials_&_Chemicals',
         'Transportation', 'Financials', 'Energy_&_Utilities', 'Trading_&_Logistics', 'Conglomerate',
        'Technology', 'Telecommunications_&_Media']
    ]
    for index in range(1):
        sector_list = cluster_list[index]
        clustered_ticker_list = []

        df_sector_stocks = pd.DataFrame()
        df_processing_data = pd.DataFrame()

        for sector in sector_list:
            ticker_list = pd.read_csv(f'./data_kr/date_sector/{sector}/sector_code.csv')
            clustered_ticker_list.extend(ticker_list["code"].to_list())

            for ticker in ticker_list["code"]:
                if ticker == "code":
                    continue
                ticker_str = str(ticker).zfill(6)
                df_price = pd.read_csv(f"./data_kr/price/{ticker_str}.csv")
                df_tmp = pd.read_csv(f"./data_kr/merged/{ticker_str}.csv")
                df_sector_stocks = pd.concat([df_sector_stocks, df_tmp])

        nan_ratio = df_sector_stocks.isna().mean()
        over_50_columns = nan_ratio[nan_ratio >= 0.5].index.to_list()
        for ticker in tqdm(clustered_ticker_list):
            if ticker == "code":
                continue
            ticker_str = str(ticker).zfill(6)
            df_data = pd.read_csv(f"./data_kr/merged/{ticker_str}.csv", index_col=[0])
            df_data.drop(over_50_columns, axis=1, inplace=True, errors="ignore")
            ###
            for col in df_data.columns[4:]:
                df_data[col] = df_data[col].astype(str).str.replace(',', '')
                df_data[col] = df_data[col].replace('-', np.nan).astype(float)
            ###

            df_data.iloc[:, 4:] = pd.DataFrame(impute.fit_transform(df_data.iloc[:, 4:]))

            df_pct_change = (-df_data.iloc[:, 4:].pct_change(fill_method=None))
            df_pct_change.columns = [col + "변화율" for col in df_pct_change.columns]
            df_data = pd.concat([df_data.iloc[:, :4], df_pct_change], axis=1)
            df_data.fillna(0.0, inplace=True)
            df_data.replace([-np.inf, np.inf], 0.0, inplace=True)

            # df_data = df_data[~df_data['year'].str.contains('2015')&~df_data['quarter'].str.contains('Q4')]

            df_data = df_data[
                ~(df_data['year'].astype(str).str.contains('2015') & df_data['quarter'].astype(str).str.contains('Q4'))
            ]
            df_data2 = df_data

            # 시장초과수익률 계산
            df_price = pd.read_csv(f"./data_kr/price/{ticker_str}.csv", index_col=[0])
            stock_date = datetime.strptime(df_price.index[0], "%Y-%m-%d")
            past_return = []
            df_data = df_data.iloc[::-1].reset_index(drop=True)
            for i in range(0, len(df_data) - 1):

                start_price = get_start_price(df_data.iloc[i + 1, 2], df_data.iloc[i + 1, 3], ticker_str)
                end_price = get_end_price(df_data.iloc[i, 2], df_data.iloc[i, 3], ticker_str)

                ks200_start_price = get_start_price(df_data.iloc[i + 1, 2], df_data.iloc[i + 1, 3], "KS200")
                ks200_end_price = get_end_price(df_data.iloc[i, 2], df_data.iloc[i, 3], "KS200")

                if start_price == 0 or ks200_start_price == 0:
                    relative_return = 0.0
                else:
                    relative_return = (end_price - start_price) / start_price - (
                                ks200_end_price - ks200_start_price) / ks200_start_price

                past_return.append(relative_return)

            past_return.append(0)
            df_data["Relative Return"] = past_return

            df_price = pd.read_csv(f"./data_kr/price/{ticker_str}.csv", index_col=[0])
            df_ks200 = pd.read_csv(f"./data_kr/price/KS200.csv", index_col=[0])
            stock_date = datetime.strptime(df_price.index[0], "%Y-%m-%d")

            label = []
            start_date = '2024-07-01'
            end_date = '2024-09-30'

            start_price = get_start_price(df_data.iloc[1, 2], df_data.iloc[1, 3], ticker_str)
            end_price = get_end_price(df_data.iloc[0, 2], df_data.iloc[0, 3], ticker_str)

            ks200_start_price = get_start_price(df_data.iloc[1, 2], df_data.iloc[1, 3], "KS200")
            ks200_end_price = get_end_price(df_data.iloc[0, 2], df_data.iloc[0, 3], "KS200")
            if start_price == 0 or ks200_start_price == 0:
                relative_return = 0.0
            else:
                relative_return = (end_price - start_price) / start_price - (
                        ks200_end_price - ks200_start_price) / ks200_start_price

            label.append(relative_return)

            for i in range(1, len(df_data)):
                start_year = df_data.iloc[i, 2]
                end_year = df_data.iloc[i - 1, 2]
                start_quarter = df_data.iloc[i, 3]
                end_quarter = df_data.iloc[i - 1, 3]

                start_date = quarter2date(start_year, start_quarter)
                if start_date < stock_date:
                    label.append(0.0)
                    continue
                start_price = get_start_price(df_data.iloc[i, 2], df_data.iloc[i, 3], ticker_str)
                end_price = get_end_price(df_data.iloc[i - 1, 2], df_data.iloc[i - 1, 3], ticker_str)

                ks200_start_price = get_start_price(df_data.iloc[i, 2], df_data.iloc[i, 3], "KS200")
                ks200_end_price = get_end_price(df_data.iloc[i - 1, 2], df_data.iloc[i - 1, 3], "KS200")
                if start_price == 0 or ks200_start_price == 0:
                    relative_return = 0.0
                else:
                    relative_return = (end_price - start_price) / start_price - (
                                ks200_end_price - ks200_start_price) / ks200_start_price

                label.append(relative_return)

            df_data["Label"] = label
            df_data["Code"] = ticker_str

            if df_processing_data.empty:
                df_processing_data = df_data
            else:
                df_processing_data = pd.concat([df_processing_data, df_data], axis=0)

        if os.path.isdir(f"./data_kr/clustered_data/ALL") == False:
            os.mkdir(f"./data_kr/clustered_data/ALL")
        df_processing_data.to_csv(f"./data_kr/clustered_data/ALL/cluster_check.csv", encoding='utf-8-sig')

        # df_processing_data.iloc[:, 4:-2] = df_processing_data.iloc[:, 4:-2].replace([-np.inf, np.inf], 0.0)
        # df_processing_data.iloc[:, -2:-1] = df_processing_data.iloc[:, -2:-1].replace([-np.inf, np.inf], 0.0)


        # Feature Matrix (X) Imputation
        X_imputed = imputer.fit_transform(df_processing_data.iloc[:, 4:-2])
        df_processing_data.iloc[:, 4:-2] = pd.DataFrame(X_imputed, columns=df_processing_data.columns[4:-2],
                                                        index=df_processing_data.index)

        # Target (y) Imputation
        y_imputed = imputer.fit_transform(df_processing_data.iloc[:, -2:-1])
        df_processing_data.iloc[:, -2:-1] = pd.DataFrame(y_imputed, columns=df_processing_data.columns[-2:-1],
                                                         index=df_processing_data.index)

        rgr = RandomForestRegressor()
        rgr.fit(df_processing_data.iloc[:, 4:-2], df_processing_data.iloc[:, -2:-1])

        feature_importance = pd.Series(rgr.feature_importances_, index=df_processing_data.columns[4:-2]).sort_values(
            ascending=False)
        feature_importance[:6].to_csv(f"./data_kr/clustered_data/ALL/ALL_feature_imp.csv", encoding='utf-8-sig')

        select_col = feature_importance.index[:6]
        df_col[sector] = select_col

        df_processed_data = pd.concat(
            [df_processing_data.iloc[:, :4], df_processing_data[select_col], df_processing_data.iloc[:, -2:]], axis=1)

        start_year = 2015
        start_quarter = 4
        end_year = 2024
        end_quarter = 3

        year = start_year
        quarter = start_quarter

        while True:
            quarter_str = f"Q{quarter}"
            df_date_filtered = df_processed_data[
                df_processed_data['year'].astype(str).str.contains(str(year)) &
                df_processed_data['quarter'].astype(str).str.contains(quarter_str)
                ]
            symbol_index = df_date_filtered["Code"].sort_values()
            df_date_filtered = df_date_filtered.sort_values(by="Code")
            df_date_filtered.to_csv(f"./data_kr/clustered_data/ALL/{year}_{quarter_str}.csv", encoding='utf-8-sig')
            if year == end_year and quarter == end_quarter:
                break
            quarter += 1
            if quarter > 4:
                quarter = 1
                year += 1
        symbol_index.to_csv(f"./data_kr/clustered_data/ALL/symbol_index.csv", encoding='utf-8-sig')

    exit()



