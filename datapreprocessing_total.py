import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings('ignore')

import ast
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
df_col_k = {}
df_col_m = {}

impute = SoftImpute(verbose=False)
#impute = SimpleImputer(strategy='median')
#impute = KNNImputer(n_neighbors=5)

n_features_t = 9

def add_prefix_to_columns(df, prefix, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    new_columns = {col: (f"{prefix}{col}" if col not in exclude_cols else col) for col in df.columns}
    return df.rename(columns=new_columns)


def merge_year_quarter_from_csv(csv_path, drop_cols=None, total_option=False):
    if drop_cols is None:
        drop_cols = []

    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    # df = df.fillna(0)

    # total_option이 True이고 "관계" 컬럼이 존재한다면, "관계" 값이 "계"인 행 제거
    if total_option and "이름" in df.columns:
        df = df[df["이름"] != "계"]

    # drop_cols에 지정된 컬럼 제거 (존재하지 않는 컬럼은 무시)
    df = df.drop(columns=drop_cols, errors='ignore')

    # '연도'와 '분기' 컬럼의 타입을 적절하게 변환
    df['연도'] = df['연도'].astype(int)
    df['분기'] = df['분기'].astype(str)

    # '연도'와 '분기'를 제외한 나머지 컬럼들을 숫자형으로 변환 (쉼표 제거 후)
    for col in df.columns:
        if col not in ['연도', '분기']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    # 그룹화에 사용할 각 열의 평균을 계산할 딕셔너리 생성 ('연도', '분기' 제외)
    agg_dict = {col: 'mean' for col in df.columns if col not in ['연도', '분기']}

    # '연도'와 '분기'별로 그룹화하여 평균 계산
    grouped = df.groupby(['연도', '분기'], as_index=False).agg(agg_dict)

    # 2015_Q4부터 2024_Q3까지 모든 연도-분기 조합 생성
    all_pairs = []
    for year in range(2015, 2025):  # 2015 ~ 2024년 반복
        if year == 2015:
            quarters = ['Q4']
        elif year == 2024:
            quarters = ['Q1', 'Q2', 'Q3']
        else:
            quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        for q in quarters:
            all_pairs.append((year, q))

    full_index_df = pd.DataFrame(all_pairs, columns=['연도', '분기'])

    # 기존 데이터와 결합
    merged = pd.merge(full_index_df, grouped, on=['연도', '분기'], how='left')

    # 존재하지 않는 값 최근 값으로 채우기
    merged.ffill(inplace=True)

    merged.drop(columns=['연도', '분기'], errors='ignore', inplace=True)

    return merged

def concat_k_features(code):
    df_small = merge_year_quarter_from_csv(f"./data_kr/k_features/소액주주/{code}.csv",['접수번호','법인구분','회사코드','회사명','구분','결제일시','주주 수','주주 비율','보유 주식 비율'],False)
    df_total = merge_year_quarter_from_csv(f"./data_kr/k_features/주식총수/{code}.csv",['접수번호','법인구분','회사코드','회사명','구분','결제일시'],False)
    df_prime = merge_year_quarter_from_csv(f"./data_kr/k_features/주요주주_소유보고/{code}.csv",['접수번호','법인구분','회사코드','회사명','대표보고자','발행 회사 관계 임원(등기여부)','발행 회사 관계 임원 직위','발행 회사 관계 주요 주주','결제일시','특정 증권 등 소유 비율','특정 증권 등 소유 증감 비율'],False)
    df_jeungja = merge_year_quarter_from_csv(f"./data_kr/k_features/증자/{code}.csv",['접수번호','법인구분','고유번호','회사코드','회사명','증자일자','증자방식','증자주식종류','구분','결제일시'],False)
    df_employee = merge_year_quarter_from_csv(f"./data_kr/k_features/직원현황/{code}.csv", ['접수번호', '법인구분', '회사코드', '회사명', '직원 수','총 급여액','비고', '결제일시'],False)
    df_maximum = merge_year_quarter_from_csv(f"./data_kr/k_features/최대주주/{code}.csv", ['접수번호', '법인구분', '회사코드', '회사명','주식종류','이름','관계', '비고', '결제일시'],True)


    dfs = [df_small,df_prime,df_employee,df_maximum,df_total,df_jeungja]
    df_concated = pd.concat(dfs,axis=1)
    return df_concated


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
    # 저장된 cluster_result.txt 파일 경로 지정
    output_folder = "./data_kr/clusters_output"
    result_file = os.path.join(output_folder, "cluster_result.txt")

    # 파일 읽기 (인코딩은 저장할 때 사용한 것과 동일하게 설정)
    with open(result_file, "r", encoding="utf-8") as f:
        cluster_result_str = f.read()

    # 문자열을 안전하게 리스트로 변환
    cluster_list = ast.literal_eval(cluster_result_str)

    for cluster_index in range(len(cluster_list)):
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
                df_kfeature_tmp = concat_k_features(ticker_str)
                df_macro_tmp = pd.read_csv(f"./data_kr/macro_economic/merged.csv")
                df_macro_tmp.drop(columns=['연도', '분기'], errors='ignore', inplace=True)

                df_tmp = add_prefix_to_columns(df_tmp, "F_",exclude_cols=["year", "quarter", "code", "name", "sector"])
                df_kfeature_tmp = add_prefix_to_columns(df_kfeature_tmp, "K_", exclude_cols=["연도", "분기"])
                df_macro_tmp = add_prefix_to_columns(df_macro_tmp, "M_", exclude_cols=["연도", "분기"])

                df_concat = pd.concat([df_tmp, df_macro_tmp, df_kfeature_tmp], axis=1)
                df_sector_stocks = pd.concat([df_sector_stocks, df_concat])

        nan_ratio = df_sector_stocks.isna().mean()
        over_50_columns = nan_ratio[nan_ratio >= 0.5].index.to_list()

        for ticker in tqdm(clustered_ticker_list):
            if ticker == "code":
                continue
            ticker_str = str(ticker).zfill(6)
            df_data = pd.read_csv(f"./data_kr/merged/{ticker_str}.csv")
            df_kfeature = concat_k_features(ticker_str)
            df_macro = pd.read_csv(f"./data_kr/macro_economic/merged.csv")
            df_macro.drop(columns=['연도', '분기'], errors='ignore', inplace=True)
            df_data = add_prefix_to_columns(df_data, "F_", exclude_cols=["year", "quarter","code","name","sector"])
            df_kfeature = add_prefix_to_columns(df_kfeature, "K_", exclude_cols=["연도", "분기"])
            df_macro = add_prefix_to_columns(df_macro, "M_", exclude_cols=["연도", "분기"])
            df_data = pd.concat([df_data, df_macro,df_kfeature], axis=1)

            base_cols = ['code', 'name', 'year', 'quarter', 'sector']
            feature_cols = df_data.filter(regex="^(F_|M_|K_)").columns.tolist()
            final_cols = base_cols + feature_cols

            df_data.drop(over_50_columns, axis=1, inplace=True, errors="ignore")

            for col in df_data.columns[5:]:
                df_data[col] = df_data[col].astype(str).str.replace(',', '')
                df_data[col] = df_data[col].replace('-', np.nan).astype(float)
            df_data.iloc[:, 5:] = pd.DataFrame(impute.fit_transform(df_data.iloc[:, 5:]))

            df_pct_change = (-df_data.iloc[:, 5:].pct_change(fill_method=None))
            df_pct_change.columns = [col + " 변화율" for col in df_pct_change.columns]
            df_data = pd.concat([df_data.iloc[:, :5], df_pct_change], axis=1)
            df_data.fillna(0.0, inplace=True)
            df_data.replace([-np.inf, np.inf], 0.0, inplace=True)


            df_data = df_data[
                ~(df_data['year'].astype(str).str.contains('2015') & df_data['quarter'].astype(str).str.contains('Q4'))
            ]

            # 시장초과수익률 계산
            df_price = pd.read_csv(f"./data_kr/price/{ticker_str}.csv", index_col=[0])
            stock_date = datetime.strptime(df_price.index[0], "%Y-%m-%d")
            past_return = []
            df_data = df_data.iloc[::-1].reset_index(drop=True)
            for i in range(0, len(df_data) - 1):

                start_price = get_start_price(df_data.iloc[i + 1, 3], df_data.iloc[i + 1, 4], ticker_str)
                end_price = get_end_price(df_data.iloc[i, 3], df_data.iloc[i, 4], ticker_str)

                ks200_start_price = get_start_price(df_data.iloc[i + 1, 3], df_data.iloc[i + 1, 4], "KS200")
                ks200_end_price = get_end_price(df_data.iloc[i, 3], df_data.iloc[i, 4], "KS200")

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

            start_price = get_start_price(df_data.iloc[1, 3], df_data.iloc[1, 4], ticker_str)
            end_price = get_end_price(df_data.iloc[0, 3], df_data.iloc[0, 4], ticker_str)

            ks200_start_price = get_start_price(df_data.iloc[1, 3], df_data.iloc[1, 4], "KS200")
            ks200_end_price = get_end_price(df_data.iloc[0, 3], df_data.iloc[0, 4], "KS200")
            if start_price == 0 or ks200_start_price == 0:
                relative_return = 0.0
            else:
                relative_return = (end_price - start_price) / start_price - (
                        ks200_end_price - ks200_start_price) / ks200_start_price

            label.append(relative_return)

            for i in range(1, len(df_data)):
                start_year = df_data.iloc[i, 3]
                end_year = df_data.iloc[i - 1, 3]
                start_quarter = df_data.iloc[i, 4]
                end_quarter = df_data.iloc[i - 1, 4]

                start_date = quarter2date(start_year, start_quarter)
                if start_date < stock_date:
                    label.append(0.0)
                    continue
                start_price = get_start_price(df_data.iloc[i, 3], df_data.iloc[i, 4], ticker_str)
                end_price = get_end_price(df_data.iloc[i - 1, 3], df_data.iloc[i - 1, 4], ticker_str)

                ks200_start_price = get_start_price(df_data.iloc[i, 3], df_data.iloc[i, 4], "KS200")
                ks200_end_price = get_end_price(df_data.iloc[i - 1, 3], df_data.iloc[i - 1, 4], "KS200")
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

        base_cols = ['code', 'name','year','quarter','sector']  # 예시로 기본 정보 열을 지정합니다.
        feature_cols = df_processing_data.filter(regex="^(F_|M_|K_)").columns.tolist()
        final_cols = base_cols + feature_cols + ['Relative Return','Label', 'Code']

        # 최종 DataFrame 재정렬
        df_processing_data = df_processing_data[final_cols]

        X_imputed = impute.fit_transform(df_processing_data.iloc[:, 5:-2])
        df_processing_data.iloc[:, 5:-2] = pd.DataFrame(X_imputed, columns=df_processing_data.columns[5:-2],
                                                        index=df_processing_data.index)

        # Target (y) Imputation
        y_imputed = impute.fit_transform(df_processing_data.iloc[:, -2:-1])
        df_processing_data.iloc[:, -2:-1] = pd.DataFrame(y_imputed, columns=df_processing_data.columns[-2:-1],
                                                         index=df_processing_data.index)

        rgr = RandomForestRegressor()
        rgr.fit(df_processing_data.iloc[:, 5:-3], df_processing_data.iloc[:, -2:-1])
        feature_importance = pd.Series(rgr.feature_importances_, index=df_processing_data.columns[5:-3]).sort_values(
            ascending=False)
        feature_importance[:n_features_t].to_csv(
            f"./data_kr/clustered_data/cluster_{cluster_index}/cluster_{cluster_index}_T_feature_imp.csv",
            encoding='utf-8-sig')

        select_col = feature_importance.index[:n_features_t]
        df_col[sector] = select_col

        df_processed_data = pd.concat(
            [df_processing_data.iloc[:, 1:5], df_processing_data.iloc[:, -3:-2], df_processing_data[select_col],
             df_processing_data.iloc[:, -2:]],
            axis=1)

        df_processed_data.columns = df_processed_data.columns.str.replace(r'^(F_|M_|K_)', '', regex=True)

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
elif args.isall == "cluster":
    cluster_list = [
        ['유통'], ['음식료·담배'], ['제약'] ,['운송·창고'], ['기타금융'] ,['화학'] ,['운송장비·부품'] ,['비금속'] ,['전기·전자'] ,['금속'] ,['건설']
         ,['섬유·의류'] ,['전기·가스'] ,['종이·목재'] ,['일반서비스'] ,['통신'] ,['기계·장비'] ,['IT 서비스'] ,['오락·문화'] ,['기타제조']]

    #cluster_list = [['건설'], ['경기소비재'],['산업재'],['생활소비재'],['에너지_화학'],['정보기술'],['중공업'],['철강_소재'],['커뮤니케이션서비스'],['헬스케어']]
    for cluster_index in range(len(cluster_list)):
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
                df_kfeature_tmp = concat_k_features(ticker_str)
                df_macro_tmp = pd.read_csv(f"./data_kr/macro_economic/merged.csv")
                df_macro_tmp.drop(columns=['연도', '분기'], errors='ignore', inplace=True)

                df_tmp = add_prefix_to_columns(df_tmp, "F_", exclude_cols=["year", "quarter", "code", "name", "sector"])
                df_kfeature_tmp = add_prefix_to_columns(df_kfeature_tmp, "K_", exclude_cols=["연도", "분기"])
                df_macro_tmp = add_prefix_to_columns(df_macro_tmp, "M_", exclude_cols=["연도", "분기"])

                df_concat = pd.concat([df_tmp, df_macro_tmp, df_kfeature_tmp], axis=1)
                df_sector_stocks = pd.concat([df_sector_stocks, df_concat])

        nan_ratio = df_sector_stocks.isna().mean()
        over_50_columns = nan_ratio[nan_ratio >= 0.5].index.to_list()
        for ticker in tqdm(clustered_ticker_list):
            if ticker == "code":
                continue
            ticker_str = str(ticker).zfill(6)
            df_data = pd.read_csv(f"./data_kr/merged/{ticker_str}.csv")
            df_kfeature = concat_k_features(ticker_str)
            df_macro = pd.read_csv(f"./data_kr/macro_economic/merged.csv")
            df_macro.drop(columns=['연도', '분기'], errors='ignore', inplace=True)
            df_data = add_prefix_to_columns(df_data, "F_", exclude_cols=["year", "quarter", "code", "name", "sector"])
            df_kfeature = add_prefix_to_columns(df_kfeature, "K_", exclude_cols=["연도", "분기"])
            df_macro = add_prefix_to_columns(df_macro, "M_", exclude_cols=["연도", "분기"])
            df_data = pd.concat([df_data, df_macro, df_kfeature], axis=1)

            base_cols = ['code', 'name', 'year', 'quarter', 'sector']
            feature_cols = df_data.filter(regex="^(F_|M_|K_)").columns.tolist()
            final_cols = base_cols + feature_cols

            df_data.drop(over_50_columns, axis=1, inplace=True, errors="ignore")
            ###
            for col in df_data.columns[5:]:
                df_data[col] = df_data[col].astype(str).str.replace(',', '')
                df_data[col] = df_data[col].replace('-', np.nan).astype(float)
            ###
            df_data.iloc[:, 5:] = pd.DataFrame(impute.fit_transform(df_data.iloc[:, 5:]))
            # df_data = pd.concat([df_data.iloc[:,:5],pd.DataFrame(impute.fit_transform(df_data.iloc[:, 5:]))])

            df_pct_change = (-df_data.iloc[:, 5:].pct_change(fill_method=None))
            df_pct_change.columns = [col + " 변화율" for col in df_pct_change.columns]
            df_data = pd.concat([df_data.iloc[:, :5], df_pct_change], axis=1)
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

                start_price = get_start_price(df_data.iloc[i + 1, 3], df_data.iloc[i + 1, 4], ticker_str)
                end_price = get_end_price(df_data.iloc[i, 3], df_data.iloc[i, 4], ticker_str)

                ks200_start_price = get_start_price(df_data.iloc[i + 1, 3], df_data.iloc[i + 1, 4], "KS200")
                ks200_end_price = get_end_price(df_data.iloc[i, 3], df_data.iloc[i, 4], "KS200")

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

            start_price = get_start_price(df_data.iloc[1, 3], df_data.iloc[1, 4], ticker_str)
            end_price = get_end_price(df_data.iloc[0, 3], df_data.iloc[0, 4], ticker_str)

            ks200_start_price = get_start_price(df_data.iloc[1, 3], df_data.iloc[1, 4], "KS200")
            ks200_end_price = get_end_price(df_data.iloc[0, 3], df_data.iloc[0, 4], "KS200")
            if start_price == 0 or ks200_start_price == 0:
                relative_return = 0.0
            else:
                relative_return = (end_price - start_price) / start_price - (
                        ks200_end_price - ks200_start_price) / ks200_start_price

            label.append(relative_return)

            for i in range(1, len(df_data)):
                start_year = df_data.iloc[i, 3]
                end_year = df_data.iloc[i - 1, 3]
                start_quarter = df_data.iloc[i, 4]
                end_quarter = df_data.iloc[i - 1, 4]

                start_date = quarter2date(start_year, start_quarter)
                if start_date < stock_date:
                    label.append(0.0)
                    continue
                start_price = get_start_price(df_data.iloc[i, 3], df_data.iloc[i, 4], ticker_str)
                end_price = get_end_price(df_data.iloc[i - 1, 3], df_data.iloc[i - 1, 4], ticker_str)

                ks200_start_price = get_start_price(df_data.iloc[i, 3], df_data.iloc[i, 4], "KS200")
                ks200_end_price = get_end_price(df_data.iloc[i - 1, 3], df_data.iloc[i - 1, 4], "KS200")
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

        base_cols = ['code', 'name', 'year', 'quarter', 'sector']  # 예시로 기본 정보 열을 지정합니다.
        feature_cols = df_processing_data.filter(regex="^(F_|M_|K_)").columns.tolist()
        final_cols = base_cols + feature_cols + ['Relative Return', 'Label', 'Code']

        # 최종 DataFrame 재정렬
        df_processing_data = df_processing_data[final_cols]

        X_imputed = impute.fit_transform(df_processing_data.iloc[:, 5:-2])
        df_processing_data.iloc[:, 5:-2] = pd.DataFrame(X_imputed, columns=df_processing_data.columns[5:-2],
                                                        index=df_processing_data.index)

        # Target (y) Imputation
        y_imputed = impute.fit_transform(df_processing_data.iloc[:, -2:-1])
        df_processing_data.iloc[:, -2:-1] = pd.DataFrame(y_imputed, columns=df_processing_data.columns[-2:-1],
                                                         index=df_processing_data.index)

        rgr = RandomForestRegressor()
        rgr.fit(df_processing_data.iloc[:, 5:-3], df_processing_data.iloc[:, -2:-1])
        feature_importance = pd.Series(rgr.feature_importances_, index=df_processing_data.columns[5:-3]).sort_values(
            ascending=False)
        feature_importance[:n_features_t].to_csv(
            f"./data_kr/financial_with_Label/{sector_list[0]}/{sector_list[0]}_T_feature_imp.csv",
            encoding='utf-8-sig')

        select_col = feature_importance.index[:n_features_t]
        df_col[sector] = select_col

        df_processed_data = pd.concat(
            [df_processing_data.iloc[:, 1:5], df_processing_data.iloc[:, -3:-2], df_processing_data[select_col],
             df_processing_data.iloc[:, -2:]],
            axis=1)

        df_processed_data.columns = df_processed_data.columns.str.replace(r'^(F_|M_|K_)', '', regex=True)

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
elif args.isall == "True":
    cluster_list = [['유통', '음식료·담배', '제약', '운송·창고', '기타금융', '화학', '운송장비·부품', '비금속' ,'전기·전자', '금속', '건설'
         ,'섬유·의류' ,'전기·가스', '종이·목재', '일반서비스', '통신', '기계·장비', 'IT 서비스', '오락·문화', '기타제조']]

    #cluster_list = [['건설', '경기소비재', '산업재', '생활소비재', '에너지_화학', '정보기술', '중공업', '철강_소재','커뮤니케이션서비스', '헬스케어']]
    for cluster_index in range(len(cluster_list)):
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
                df_kfeature_tmp = concat_k_features(ticker_str)
                df_macro_tmp = pd.read_csv(f"./data_kr/macro_economic/merged.csv")
                df_macro_tmp.drop(columns=['연도', '분기'], errors='ignore', inplace=True)

                df_tmp = add_prefix_to_columns(df_tmp, "F_", exclude_cols=["year", "quarter", "code", "name", "sector"])
                df_kfeature_tmp = add_prefix_to_columns(df_kfeature_tmp, "K_", exclude_cols=["연도", "분기"])
                df_macro_tmp = add_prefix_to_columns(df_macro_tmp, "M_", exclude_cols=["연도", "분기"])

                df_concat = pd.concat([df_tmp, df_macro_tmp, df_kfeature_tmp], axis=1)
                df_sector_stocks = pd.concat([df_sector_stocks, df_concat])

        nan_ratio = df_sector_stocks.isna().mean()
        over_50_columns = nan_ratio[nan_ratio >= 0.5].index.to_list()

        for ticker in tqdm(clustered_ticker_list):
            if ticker == "code":
                continue
            ticker_str = str(ticker).zfill(6)
            df_data = pd.read_csv(f"./data_kr/merged/{ticker_str}.csv")
            df_kfeature = concat_k_features(ticker_str)
            df_macro = pd.read_csv(f"./data_kr/macro_economic/merged.csv")
            df_macro.drop(columns=['연도', '분기'], errors='ignore', inplace=True)
            df_data = add_prefix_to_columns(df_data, "F_", exclude_cols=["year", "quarter", "code", "name", "sector"])
            df_kfeature = add_prefix_to_columns(df_kfeature, "K_", exclude_cols=["연도", "분기"])
            df_macro = add_prefix_to_columns(df_macro, "M_", exclude_cols=["연도", "분기"])
            df_data = pd.concat([df_data, df_macro, df_kfeature], axis=1)

            base_cols = ['code', 'name', 'year', 'quarter', 'sector']
            feature_cols = df_data.filter(regex="^(F_|M_|K_)").columns.tolist()
            final_cols = base_cols + feature_cols

            df_data.drop(over_50_columns, axis=1, inplace=True, errors="ignore")
            ###
            for col in df_data.columns[5:]:
                df_data[col] = df_data[col].astype(str).str.replace(',', '')
                df_data[col] = df_data[col].replace('-', np.nan).astype(float)
            ###
            df_data.iloc[:, 5:] = pd.DataFrame(impute.fit_transform(df_data.iloc[:, 5:]))
            # df_data = pd.concat([df_data.iloc[:,:5],pd.DataFrame(impute.fit_transform(df_data.iloc[:, 5:]))])

            df_pct_change = (-df_data.iloc[:, 5:].pct_change(fill_method=None))
            df_pct_change.columns = [col + " 변화율" for col in df_pct_change.columns]
            df_data = pd.concat([df_data.iloc[:, :5], df_pct_change], axis=1)
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

                start_price = get_start_price(df_data.iloc[i + 1, 3], df_data.iloc[i + 1, 4], ticker_str)
                end_price = get_end_price(df_data.iloc[i, 3], df_data.iloc[i, 4], ticker_str)

                ks200_start_price = get_start_price(df_data.iloc[i + 1, 3], df_data.iloc[i + 1, 4], "KS200")
                ks200_end_price = get_end_price(df_data.iloc[i, 3], df_data.iloc[i, 4], "KS200")

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

            start_price = get_start_price(df_data.iloc[1, 3], df_data.iloc[1, 4], ticker_str)
            end_price = get_end_price(df_data.iloc[0, 3], df_data.iloc[0, 4], ticker_str)

            ks200_start_price = get_start_price(df_data.iloc[1, 3], df_data.iloc[1, 4], "KS200")
            ks200_end_price = get_end_price(df_data.iloc[0, 3], df_data.iloc[0, 4], "KS200")
            if start_price == 0 or ks200_start_price == 0:
                relative_return = 0.0
            else:
                relative_return = (end_price - start_price) / start_price - (
                        ks200_end_price - ks200_start_price) / ks200_start_price

            label.append(relative_return)

            for i in range(1, len(df_data)):
                start_year = df_data.iloc[i, 3]
                end_year = df_data.iloc[i - 1, 3]
                start_quarter = df_data.iloc[i, 4]
                end_quarter = df_data.iloc[i - 1, 4]

                start_date = quarter2date(start_year, start_quarter)
                if start_date < stock_date:
                    label.append(0.0)
                    continue
                start_price = get_start_price(df_data.iloc[i, 3], df_data.iloc[i, 4], ticker_str)
                end_price = get_end_price(df_data.iloc[i - 1, 3], df_data.iloc[i - 1, 4], ticker_str)

                ks200_start_price = get_start_price(df_data.iloc[i, 3], df_data.iloc[i, 4], "KS200")
                ks200_end_price = get_end_price(df_data.iloc[i - 1, 3], df_data.iloc[i - 1, 4], "KS200")
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
        df_processing_data.to_csv(f"./data_kr/clustered_data/ALL/cluster_check.csv",
                                  encoding='utf-8-sig')

        # df_processing_data.iloc[:, 4:-2] = df_processing_data.iloc[:, 4:-2].replace([-np.inf, np.inf], 0.0)
        # df_processing_data.iloc[:, -2:-1] = df_processing_data.iloc[:, -2:-1].replace([-np.inf, np.inf], 0.0)

        # Feature Matrix (X) Imputation

        base_cols = ['code', 'name', 'year', 'quarter', 'sector']  # 예시로 기본 정보 열을 지정합니다.
        feature_cols = df_processing_data.filter(regex="^(F_|M_|K_)").columns.tolist()
        final_cols = base_cols + feature_cols + ['Relative Return', 'Label', 'Code']

        # 최종 DataFrame 재정렬
        df_processing_data = df_processing_data[final_cols]

        X_imputed = impute.fit_transform(df_processing_data.iloc[:, 5:-2])
        df_processing_data.iloc[:, 5:-2] = pd.DataFrame(X_imputed, columns=df_processing_data.columns[5:-2],
                                                        index=df_processing_data.index)

        # Target (y) Imputation
        y_imputed = impute.fit_transform(df_processing_data.iloc[:, -2:-1])
        df_processing_data.iloc[:, -2:-1] = pd.DataFrame(y_imputed, columns=df_processing_data.columns[-2:-1],
                                                         index=df_processing_data.index)

        rgr = RandomForestRegressor()
        rgr.fit(df_processing_data.iloc[:,5:-3], df_processing_data.iloc[:, -2:-1])
        feature_importance = pd.Series(rgr.feature_importances_, index=df_processing_data.columns[5:-3]).sort_values(ascending=False)
        feature_importance[:n_features_t].to_csv(
            f"./data_kr/clustered_data/ALL/ALL_T_feature_imp.csv",
            encoding='utf-8-sig')

        select_col = feature_importance.index[:n_features_t]
        df_col[sector] = select_col

        df_processed_data = pd.concat(
            [df_processing_data.iloc[:, 1:5], df_processing_data.iloc[:, -3:-2], df_processing_data[select_col],
             df_processing_data.iloc[:, -2:]],
            axis=1)

        df_processed_data.columns = df_processed_data.columns.str.replace(r'^(F_|M_|K_)', '', regex=True)

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
else:
    print("Wrong COMMAND")
    exit()



