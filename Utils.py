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
            return max((peak_pv - pv) / peak_pv for peak_pv, pv in zip(itertools.accumulate(memory, max), memory))

    def get_sharpe_ratio(self,memory): # 샤프 비율 계산
        # money_history ex: [1.0, 1.2, 1.5, 1.3, 1.6]
        if len(memory) == 0: return 0
        elif len(memory) == 1: return memory[0] * 100
        else:
            return np.mean(memory) * np.sqrt(len(memory)) / (np.std(memory) + 1e-6)

    def get_CAGR(self,money_history): # 연평균 성장률 계산
        n = len(money_history)/4
        cagr = np.power((money_history[-1]/money_history[0]),1/n)-1
        
        return cagr

    def get_date_range(self,year, quarter, ticker):
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
                return current_date.strftime("%Y-%m-%d")
            current_date += timedelta(days=1)
            # 데이터 범위를 넘어가는 경우 예외 발생
            if current_date > df['날짜'].max():
                print(f"{ticker}의 데이터에서 {quarter_start} 이후의 거래일을 찾을 수 없습니다.")
                return 0
    
    def get_portfolio_memory(self,stocks,strdate,next_strdate): # 포트폴리오 수익률 계산

        if len(stocks) == 0: return []
        
        df_total_price = pd.DataFrame()

        year_now, quarter_now = strdate.split('_')
        year_next, quarter_next = next_strdate.split('_')
        for ticker in stocks: # stocks 리스트에 있는 종목들의 주가 데이터를 가져옴
            if ticker != "KS200":
                ticker_str = str(ticker).zfill(6)
            else:
                ticker_str = ticker  # 또는 필요한 경우 다른 형식으로 변환
            start_date = self.get_date_range(year_now, quarter_now, ticker_str)
            end_date = self.get_date_range(year_next, quarter_next, ticker_str)
            price = pd.read_csv(f"./data_kr/price/{ticker_str}.csv",index_col=[0])['종가'].loc[
                    start_date:end_date]

            df_total_price = pd.concat([df_total_price,price],axis=1,join='outer')

            df_total_price.sort_index(inplace=True)

        df_total_price = df_total_price.fillna(method='ffill').fillna(method='bfill').sum(axis=1)
        df_pf = df_total_price/df_total_price.iloc[0]
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

    
