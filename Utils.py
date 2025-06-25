########################################################################################################################
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import itertools
import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))

########################################################################################################################
# 포트폴리오 성과 분석을 위한 유틸리티 함수 모음
# MDD: 최대 낙폭 계산
# Sharpe Ratio: 위험 대비 수익률 평가
# CAGR: 연평균 성장률 계산
# 주가 데이터 처리 및 포트폴리오 수익률 계산
class Utils:
    def __init__(self):
        pass

    def get_MDD(self,memory): # 최대 낙폭 계산: 투자 기간동안 포트폴리오가 가장 많이 하락한 비율
        # memory: 포트폴리오의 가치(주가 혹은 누적 수익률 리스트)
        if len(memory) <= 1: return 0
        else:
            memory = np.cumprod(memory)
            mdd = max((peak_pv - pv) / peak_pv for peak_pv, pv in zip(itertools.accumulate(memory, max), memory))
            return mdd

    def get_sharpe_ratio(self,memory): # 샤프 비율 계산
        # money_history ex: [1.0, 1.2, 1.5, 1.3, 1.6]
        if len(memory) == 0: return 0
        elif len(memory) == 1: return memory[0] * 100
        else:
            return np.mean(memory) * np.sqrt(len(memory)) / (np.std(memory) + 1e-6)

    def get_CAGR(self,money_history): # 연평균 성장률 계산
        n = 1
        cagr = np.power((money_history[-1]/money_history[0]),1/n)-1
        
        return cagr

    def get_start_date(self,year, quarter, ticker):
        """
        date_regression 폴더의 공시일(disclosure_date)을 기준으로
        가장 가까운 거래일의 시가(Open)를 반환합니다.

        :param year: int, 연도
        :param quarter: str, "Q1" ~ "Q4"
        :param ticker: str, 종목 코드 (6자리 문자열)
        :return: 해당 종목의 시가 (float)
        """
        # 1. disclosure_date 가져오기
        try:
            reg_path = f"./data_kr/date_regression/{year}_{quarter}.csv"
            reg_df = pd.read_csv(reg_path)
            reg_row = reg_df[reg_df['code'].astype(str).str.zfill(6) == ticker]
            if reg_row.empty:
                print(f"{ticker}: {year}_{quarter} 공시일 정보를 찾을 수 없습니다.")
                return 0
            current_date = pd.to_datetime(reg_row.iloc[0]['disclosure_date'])
        except Exception as e:
            print(f"{ticker}: 공시일 파일 로드 실패 - {e}")
            return 0

        # 2. 가격 데이터 로딩
        try:
            file_path = f"./data_kr/price/{ticker}.csv"
            df = pd.read_csv(file_path)
            df['날짜'] = pd.to_datetime(df['날짜'])
            df.sort_values(by='날짜', inplace=True)
        except Exception as e:
            print(f"{ticker}: 가격 데이터 로드 실패 - {e}")
            return 0

        # 3. disclosure_date 이후 가장 가까운 거래일의 시가 반환
        while True:
            match = df[df['날짜'].dt.date == current_date.date()]
            if not match.empty:
                return current_date.strftime("%Y-%m-%d")
            current_date += timedelta(days=1)
            if current_date > df['날짜'].max():
                print(f"{ticker}: {year}_{quarter} 이후 거래일 없음.")
                return 0

    def get_end_date(self,year, quarter, ticker):
        """
        date_regression 폴더에서 가져온 disclosure_date 기준 이전 마지막 거래일의 종가(Close)를 반환합니다.

        :param year: int, 연도
        :param quarter: str, "Q1" ~ "Q4"
        :param ticker: str, 종목 코드 (6자리 문자열)
        :return: 해당 종목의 종가 (float)
        """
        # 1. disclosure_date 가져오기
        try:
            reg_path = f"./data_kr/date_regression/{year}_{quarter}.csv"
            reg_df = pd.read_csv(reg_path)
            reg_row = reg_df[reg_df['code'].astype(str).str.zfill(6) == ticker]
            if reg_row.empty:
                print(f"{ticker}: {year}_{quarter} 공시일 정보를 찾을 수 없습니다.")
                return 0
            current_date = pd.to_datetime(reg_row.iloc[0]['disclosure_date']) - timedelta(days=1)
        except Exception as e:
            print(f"{ticker}: 공시일 파일 로드 실패 - {e}")
            return 0

        # 2. 가격 데이터 로딩
        try:
            file_path = f"./data_kr/price/{ticker}.csv"
            df = pd.read_csv(file_path)
            df['날짜'] = pd.to_datetime(df['날짜'])
            df.sort_values(by='날짜', inplace=True)
        except Exception as e:
            print(f"{ticker}: 가격 데이터 로드 실패 - {e}")
            return 0

        # 3. disclosure_date 이전 가장 가까운 거래일의 종가 반환
        while True:
            match = df[df['날짜'].dt.date == current_date.date()]
            if not match.empty:
                return current_date.strftime("%Y-%m-%d")
            current_date -= timedelta(days=1)
            if current_date < df['날짜'].min():
                print(f"{ticker}: {year}_{quarter} 이전 거래일 없음.")
                return 0

    def get_KOSPI_start_date(self,year, quarter, ticker):
        """
        date_regression 폴더의 공시일(disclosure_date)을 기준으로
        가장 가까운 거래일의 시가(Open)를 반환합니다.

        :param year: int, 연도
        :param quarter: str, "Q1" ~ "Q4"
        :param ticker: str, 종목 코드 (6자리 문자열)
        :return: 해당 종목의 시가 (float)
        """
        # 1. disclosure_date 가져오기
        try:
            reg_path = f"./data_kr/merged/KS200.csv"
            reg_df = pd.read_csv(reg_path)

            # 자료형 정리
            reg_df['quarter'] = reg_df['quarter'].astype(str).str.strip()
            reg_df['year'] = reg_df['year'].astype(int)

            # 조건 필터링
            reg_row = reg_df[(reg_df['year'] == int(year)) & (reg_df['quarter'] == str(quarter))]
            if reg_row.empty:
                print(f"{ticker}: {year}_{quarter} 공시일 정보를 찾을 수 없습니다.")
                return 0
            current_date = pd.to_datetime(reg_row.iloc[0]['disclosure_date'])
        except Exception as e:
            print(f"{ticker}: 공시일 파일 로드 실패 - {e}")
            return 0

        # 2. 가격 데이터 로딩
        try:
            file_path = f"./data_kr/price/KS200.csv"
            df = pd.read_csv(file_path)
            df['날짜'] = pd.to_datetime(df['날짜'])
            df.sort_values(by='날짜', inplace=True)
        except Exception as e:
            print(f"{ticker}: 가격 데이터 로드 실패 - {e}")
            return 0

        # 3. disclosure_date 이후 가장 가까운 거래일의 시가 반환
        while True:
            match = df[df['날짜'].dt.date == current_date.date()]
            if not match.empty:
                return current_date.strftime("%Y-%m-%d")
            current_date += timedelta(days=1)
            if current_date > df['날짜'].max():
                print(f"{ticker}: {year}_{quarter} 이후 거래일 없음.")
                return 0

    def get_KOSPI_end_date(self,year, quarter, ticker):
        """
        date_regression 폴더에서 가져온 disclosure_date 기준 이전 마지막 거래일의 종가(Close)를 반환합니다.

        :param year: int, 연도
        :param quarter: str, "Q1" ~ "Q4"
        :param ticker: str, 종목 코드 (6자리 문자열)
        :return: 해당 종목의 종가 (float)
        """
        # 1. disclosure_date 가져오기
        try:
            reg_path = f"./data_kr/merged/KS200.csv"
            reg_df = pd.read_csv(reg_path)
            reg_df['quarter'] = reg_df['quarter'].astype(str).str.strip()
            reg_df['year'] = reg_df['year'].astype(int)

            # 조건 필터링
            reg_row = reg_df[(reg_df['year'] == int(year)) & (reg_df['quarter'] == str(quarter))]

            if reg_row.empty:
                print(f"{ticker}: {year}_{quarter} 공시일 정보를 찾을 수 없습니다.")
                return 0
            current_date = pd.to_datetime(reg_row.iloc[0]['disclosure_date']) - timedelta(days=1)
        except Exception as e:
            print(f"{ticker}: 공시일 파일 로드 실패 - {e}")
            return 0

        # 2. 가격 데이터 로딩
        try:
            file_path = f"./data_kr/price/KS200.csv"
            df = pd.read_csv(file_path)
            df['날짜'] = pd.to_datetime(df['날짜'])
            df.sort_values(by='날짜', inplace=True)
        except Exception as e:
            print(f"{ticker}: 가격 데이터 로드 실패 - {e}")
            return 0

        # 3. disclosure_date 이전 가장 가까운 거래일의 종가 반환
        while True:
            match = df[df['날짜'].dt.date == current_date.date()]
            if not match.empty:
                return current_date.strftime("%Y-%m-%d")
            current_date -= timedelta(days=1)
            if current_date < df['날짜'].min():
                print(f"{ticker}: {year}_{quarter} 이전 거래일 없음.")
                return 0


    def get_portfolio_memory(self,stocks,strdate,next_strdate,isKS200=False): # 포트폴리오 수익률 계산

        if isKS200:
            df_total_price = pd.DataFrame()

            year_now, quarter_now = strdate.split('_')
            year_next, quarter_next = next_strdate.split('_')
            ticker_str = "KS200"

            start_date = self.get_KOSPI_start_date(year_now, quarter_now, ticker_str)
            end_date = self.get_KOSPI_end_date(year_next, quarter_next, ticker_str)

            price = pd.read_csv(f"./data_kr/price/{ticker_str}.csv", index_col=[0])['종가'].loc[start_date:end_date]

            df_total_price = pd.concat([df_total_price, price], axis=1, join='outer')

            df_total_price.sort_index(inplace=True)

            df_total_price = df_total_price.fillna(method='ffill').fillna(method='bfill').sum(axis=1)
            df_pf = df_total_price / df_total_price.iloc[0]
            daily_change = df_pf.pct_change().dropna()
            # 일일 변동률(%)을 계산하여 반환
            return daily_change.values.tolist()

        else:
            if len(stocks) == 0:
                return []

            df_total_price = pd.DataFrame()

            year_now, quarter_now = strdate.split('_')
            year_next, quarter_next = next_strdate.split('_')
            for ticker in stocks:  # stocks 리스트에 있는 종목들의 주가 데이터를 가져옴
                ticker_str = str(ticker).zfill(6)

                start_date = self.get_start_date(year_now, quarter_now, ticker_str)
                end_date = self.get_end_date(year_next, quarter_next, ticker_str)

                price = pd.read_csv(f"./data_kr/price/{ticker_str}.csv", index_col=[0])['종가'].loc[start_date:end_date]

                df_total_price = pd.concat([df_total_price, price], axis=1, join='outer')

                df_total_price.sort_index(inplace=True)

            df_total_price = df_total_price.fillna(method='ffill').fillna(method='bfill').sum(axis=1)
            df_pf = df_total_price / df_total_price.iloc[0]
            daily_change = df_pf.pct_change().dropna()
            # 일일 변동률(%)을 계산하여 반환
            return daily_change.values.tolist()


            
if __name__ == "__main__":
    ut = Utils()
    stocks = [33780, 11170, 35420, 36570, 5490, 18880]
    strdate = '2020_Q3'
    next_strdate = '2020_Q4'

    fs = pd.read_csv(f"./data_kr/date_regression/{strdate}.csv")
    next_fs = pd.read_csv(f"./data_kr/date_regression/{next_strdate}.csv")

    mem = ut.get_portfolio_memory(stocks,strdate,next_strdate,fs,next_fs)

    print(mem)
    print(ut.get_sharpe_ratio(mem))

    
