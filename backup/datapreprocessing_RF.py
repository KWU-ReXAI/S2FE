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
from sklearn.ensemble import RandomForestRegressor
import argparse
from feature_selection import *
from dimensionality_reduction import *

def calculate_bhar(row, stock_prices, market_prices):
    """
    한 행의 데이터(종목코드, 공시일)를 받아 BHAR을 계산합니다.
    """
    ticker = row['code']
    announcement_date = row['disclosure_date']

    # 공시일이 유효하지 않으면 0 반환
    if pd.isna(announcement_date):
        return 0.0

    try:
        stock_df = stock_prices[ticker]

        # BHAR 계산 기간 설정 (T-1 ~ T+30)
        start_date = pd.to_datetime(announcement_date) - timedelta(days=1)
        end_date = pd.to_datetime(announcement_date) + timedelta(days=30)

        # 해당 기간 데이터 필터링
        stock_period = stock_df.loc[start_date:end_date]
        market_period = market_prices.loc[start_date:end_date]

        # 데이터가 부족한 경우 0 반환 (예: 20 거래일 미만)
        if len(stock_period) < 20 or len(market_period) < 20:
            return 0.0

        # 누적 수익률 계산 (1 + 일일수익률)의 곱
        # 첫 날의 pct_change는 NaN이므로 0으로 처리
        stock_comp_return = (1 + stock_period['종가'].pct_change().fillna(0)).prod()
        market_comp_return = (1 + market_period['종가'].pct_change().fillna(0)).prod()

        # BHAR = 개별주식 누적수익률 - 시장 누적수익률
        bhar = stock_comp_return - market_comp_return

        return bhar if np.isfinite(bhar) else 0.0

    except (KeyError, IndexError) as e:
        # 해당 종목 또는 날짜에 데이터가 없는 경우
        return 0.0
    except Exception as e:
        # 기타 예외 처리
        # print(f"BHAR 계산 오류 ({ticker}, {announcement_date}): {e}")
        return 0.0


# --- 기본 설정 ---
parser = argparse.ArgumentParser()
parser.add_argument('--isall', type=str, nargs='?', default="False")
args = parser.parse_args()

impute = SoftImpute(verbose=False)
save_folder = "./preprocessed_data_RF"
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

# --- BHAR 계산을 위한 데이터 사전 로드 ---
print("BHAR 계산을 위한 데이터 사전 로드 중...")
# 시장 데이터 로드
market_prices = pd.read_csv("./data_kr/price/KS200.csv")
market_prices['날짜'] = pd.to_datetime(market_prices['날짜'])
market_prices.set_index('날짜', inplace=True)

# 모든 종목 코드 리스트업
cluster_list_for_tickers = [['건설', '경기소비재', '산업재', '생활소비재', '에너지_화학', '정보기술', '중공업', '철강_소재', '커뮤니케이션서비스', '헬스케어']]
all_tickers = []
for sector_group in cluster_list_for_tickers:
    for sector in sector_group:
        ticker_df = pd.read_csv(f'./data_kr/date_sector/{sector}/sector_code.csv')
        all_tickers.extend(ticker_df["code"].astype(str).str.zfill(6).tolist())
all_tickers = sorted(list(set(all_tickers) - {'code'}))

# 모든 종목의 가격 데이터를 딕셔너리에 저장
stock_prices = {}
for ticker in tqdm(all_tickers, desc="종목별 가격 데이터 로드"):
    try:
        df_price = pd.read_csv(f"./data_kr/price/{ticker}.csv")
        df_price['날짜'] = pd.to_datetime(df_price['날짜'])
        df_price.set_index('날짜', inplace=True)
        stock_prices[ticker] = df_price
    except FileNotFoundError:
        # print(f"가격 파일 없음: {ticker}.csv")
        pass

# 모든 공시일 데이터를 하나의 DataFrame으로 통합
all_disclosures = []
for year in range(2015, 2025):
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        try:
            df_disclosure = pd.read_csv(f"./data_kr/date_regression/{year}_{q}.csv")
            df_disclosure['year'] = year
            df_disclosure['quarter'] = q
            all_disclosures.append(df_disclosure)
        except FileNotFoundError:
            continue
disclosure_df = pd.concat(all_disclosures, ignore_index=True)
disclosure_df['code'] = disclosure_df['code'].astype(str).str.zfill(6)
disclosure_df['disclosure_date'] = pd.to_datetime(disclosure_df['disclosure_date'])

# --- 메인 전처리 로직 시작 ---
if True:
    cluster_list = [['건설', '경기소비재', '산업재', '생활소비재', '에너지_화학', '정보기술', '중공업', '철강_소재', '커뮤니케이션서비스', '헬스케어']]
    for cluster_index in range(len(cluster_list)):
        sector_list = cluster_list[cluster_index]
        clustered_ticker_list = []

        df_sector_stocks = pd.DataFrame()
        df_processing_data = pd.DataFrame()

        for sector in sector_list:
            ticker_list_df = pd.read_csv(f'./data_kr/date_sector/{sector}/sector_code.csv')
            clustered_ticker_list.extend(ticker_list_df["code"].tolist())

            for ticker in ticker_list_df["code"]:
                if ticker == "code": continue
                ticker_str = str(ticker).zfill(6)
                try:
                    df_financial_tmp = pd.read_csv(f"./data_kr/merged/{ticker_str}.csv")
                    df_sector_stocks = pd.concat([df_sector_stocks, df_financial_tmp])
                except FileNotFoundError:
                    continue

        # 결측치 50% 이상인 컬럼 식별
        nan_ratio = df_sector_stocks.isna().mean()
        over_50_columns = nan_ratio[nan_ratio >= 0.5].index.tolist()

        # 각 Ticker에 대해 특징 생성
        for ticker in tqdm(clustered_ticker_list, desc="특징 생성"):
            if ticker == "code": continue
            ticker_str = str(ticker).zfill(6)

            try:
                df_data = pd.read_csv(f"./data_kr/merged/{ticker_str}.csv")
            except FileNotFoundError:
                continue

            # 1. 데이터 타입 변환 및 결측치 처리 준비
            df_data = df_data.drop(columns=over_50_columns, errors="ignore")
            feature_cols_to_process = [col for col in df_data.columns if
                                       col not in ['code', 'name', 'sector', 'year', 'quarter', 'disclosure_date']]
            for col in feature_cols_to_process:
                df_data[col] = pd.to_numeric(df_data[col].astype(str).str.replace(',', ''), errors='coerce')

            # 2. SoftImpute 적용
            df_data[feature_cols_to_process] = pd.DataFrame(
                impute.fit_transform(df_data[feature_cols_to_process]),
                columns=feature_cols_to_process,
                index=df_data.index
            )

            # 3. 정규화
            # 총자산으로 정규화할 대차대조표 항목 리스트
            balance_sheet_cols = [
                '유동자산',
                '비유동자산',
                '유동부채',
                '비유동부채',
                '부채총계',
                '이익잉여금',
                '자본총계',
                '자본금',
                '파생상품자산',
                '파생상품부채',
                '당기손익-공정가치측정금융자산',
                '기타포괄손익-공정가치측정금융자산',
                '차입부채',
                '상각후원가측정금융자산'
            ]

            # 총매출로 정규화할 손익계산서 및 현금흐름표 항목 리스트
            income_cashflow_cols = [
                '영업이익',
                '법인세차감전 순이익',
                '당기순이익',
                '당기순이익(손실)',
                '총포괄손익',
                '파생상품관련손익',
                '영업비용',
                '이자수익',
                '이자비용',
                '금융상품관련순손익'
            ]

            # 정규화 기준 변수 (이 변수들은 나누는 값으로 사용됨)
            ASSET_COL = '자산총계'
            SALES_COL = '매출액'

            if ASSET_COL in df_data.columns and SALES_COL in df_data.columns:
                for col in balance_sheet_cols:
                    if col in df_data.columns: df_data[col] = df_data[col].div(df_data[ASSET_COL].replace(0, np.nan))
                for col in income_cashflow_cols:
                    if col in df_data.columns: df_data[col] = df_data[col].div(df_data[SALES_COL].replace(0, np.nan))
                df_data.replace([np.inf, -np.inf], np.nan, inplace=True)
                df_data.fillna(0, inplace=True)

            df_data.drop([ASSET_COL, SALES_COL], axis=1, inplace=True)
            # 4. 롤링 윈도우 특징 생성
            feature_cols = [col for col in df_data.columns if
                            col not in ['code', 'name', 'sector', 'year', 'quarter', 'disclosure_date']]
            final_feature_dfs = []
            for i in range(0, 4):
                df_shifted = df_data[feature_cols].shift(i)
                df_shifted.columns = [f"{col}_lag{i}" for col in feature_cols]
                final_feature_dfs.append(df_shifted)

            df_final_features = pd.concat(final_feature_dfs, axis=1)

            # 4-2. 컬럼 순서 재정렬 (원본 특징별로 그룹화)
            sorted_columns = []
            for base_col in feature_cols:
                for i in range(0, 4):
                    sorted_columns.append(f"{base_col}_lag{i}")
            df_final_features = df_final_features[sorted_columns]

            df_data_with_features = pd.concat([
                df_data[['code', 'name', 'sector', 'year', 'quarter']],
                df_final_features
            ], axis=1)

            df_data_with_features.dropna(inplace=True)
            df_data_with_features.reset_index(drop=True, inplace=True)

            # 전체 데이터셋에 추가
            df_processing_data = pd.concat([df_processing_data, df_data_with_features], ignore_index=True)

        # --- BHAR 레이블 생성 및 최종 데이터셋 결합 ---
        print("BHAR 레이블 생성 중...")

        # ## 수정 코드 ##: merge 전, 두 데이터프레임의 'code' 컬럼 타입을 문자열(str)로 통일
        df_processing_data['code'] = df_processing_data['code'].astype(str).str.zfill(6)
        disclosure_df['code'] = disclosure_df['code'].astype(str).str.zfill(6)

        # 공시일 정보 결합 (기존 코드)
        df_processing_data = pd.merge(df_processing_data, disclosure_df[['code', 'year', 'quarter', 'disclosure_date']],
                                      on=['code', 'year', 'quarter'], how='left')

        # BHAR 계산 적용
        labels = df_processing_data.apply(lambda row: calculate_bhar(row, stock_prices, market_prices), axis=1)
        df_processing_data['Label'] = labels

        # 불필요한 공시일 컬럼 제거
        df_processing_data.drop(columns=['disclosure_date'], inplace=True)
        df_processing_data.rename(columns={'code': 'Code'}, inplace=True)

        cols = df_processing_data.columns.tolist()  # 현재 컬럼 리스트를 가져옵니다.
        cols.remove('Code')  # 'code' 컬럼을 리스트에서 제거합니다.
        label_index = cols.index('Label')  # 'Label' 컬럼의 위치를 찾습니다.
        cols.insert(label_index + 1, 'Code')  # 'Label' 컬럼 바로 다음에 'code'를 삽입합니다.
        df_processing_data = df_processing_data[cols]

        # 최종 결과물 저장
        save_path = f"{save_folder}/ALL"
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # 분기별로 나누어 저장

        df_processed_data = df_processing_data.copy()
        start_year, start_quarter = 2015, 4
        end_year, end_quarter = 2024, 3

        year, quarter_num = start_year, start_quarter
        while True:
            quarter_str = f"Q{quarter_num}"

            df_date_filtered = df_processed_data[
                (df_processed_data['year'] == year) &
                (df_processed_data['quarter'] == quarter_str)
                ]

            if not df_date_filtered.empty:
                df_date_filtered = df_date_filtered.sort_values(by="Code").reset_index(drop=True)
                df_date_filtered.to_csv(f"{save_path}/{year}_{quarter_str}.csv", index=[0], encoding='utf-8-sig')

            if year == end_year and quarter_num == end_quarter:
                break

            quarter_num += 1
            if quarter_num > 4:
                quarter_num = 1
                year += 1

        print(f"최종 데이터 처리 및 저장이 완료되었습니다. -> {save_path}")
    exit()
else:
    print("Wrong COMMAND")
    exit()


