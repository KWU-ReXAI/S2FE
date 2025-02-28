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

    def get_start_price(self,price,date):
        str_date = datetime.strptime(date,"%Y-%m-%d")
        try:
            out = price.loc[date]['Close']
        except:
            start_date = str_date + timedelta(1)
            out = self.get_start_price(price,datetime.strftime(start_date,"%Y-%m-%d"))
        return out

    def get_end_price(self,price,date):
        str_date = datetime.strptime(date,"%Y-%m-%d")
        end_date = str_date - timedelta(1)
        try:
            out = price.loc[datetime.strftime(end_date,"%Y-%m-%d")]['Close']
        except:
            out = self.get_end_price(price,datetime.strftime(end_date,"%Y-%m-%d"))
        return out
    
    def get_portfolio_memory(self,stocks,strdate,next_strdate,fs,next_fs): # 포트폴리오 수익률 계산

        if len(stocks) == 0: return []
        
        df_total_price = pd.DataFrame()
        for ticker in stocks: # stocks 리스트에 있는 종목들의 주가 데이터를 가져옴
            row = fs.index[fs['symbol']==ticker][0]

            if ticker == "BRK.B":
                ticker = "BRK-B"
            elif ticker == "BF.B":
                ticker = "BF-B"

            start_date = fs['Filing Date'][row]
            if next_strdate != '2024-06-30':
                next_filing_date = next_fs['Filing Date'][row]
            else:
                next_filing_date = '2024-07-31'

            price = pd.read_csv(f"./data/price/{ticker}.csv",index_col=[0])['Close'].loc[start_date:next_filing_date]

            df_total_price = pd.concat([df_total_price,price],axis=1,join='outer')

            df_total_price.sort_index(inplace=True)

        df_total_price = df_total_price.fillna(method='ffill').fillna(method='bfill').sum(axis=1)

        df_pf = df_total_price/df_total_price.iloc[0] # 포트폴리오 기준일을 1로 정규화

        daily_change = df_pf.pct_change().dropna()
        # 일일 변동률(%)을 계산하여 반환
        return daily_change.values.tolist()
    
    def get_portfolio_memory_v(self,stocks,strdate,next_strdate): # 비슷한 방식이지만 timedelta(4)을 더해서 특정 기간 이후의 데이터를 확인

        if len(stocks) == 0: return []
        
        df_total_price = pd.DataFrame()
        for ticker in stocks:
            row = fs.index[fs['symbol']==ticker][0]

            if ticker == "BRK.B":
                ticker = "BRK-B"
            elif ticker == "BF.B":
                ticker = "BF-B"

            price = pd.read_csv(f"./data/price/{ticker}.csv",index_col=[0])['Close'].loc[strdate+timedelta(40):next_strdate+timedelta(40)] # 40일 후의 날짜를 계산

            df_total_price = pd.concat([df_total_price,price],axis=1,join='outer')

            df_total_price.sort_index(inplace=True)

        df_total_price = df_total_price.fillna(method='ffill').fillna(method='bfill').sum(axis=1)

        df_pf = df_total_price/df_total_price.iloc[0]

        daily_change = df_pf.pct_change().dropna()

        return daily_change.values.tolist()

        

        # return_ratio =

            
if __name__ == "__main__":
    ut = Utils()
    stocks = ['J','PWR','CSX','ADP']
    strdate = '2010-06-30'
    next_strdate = '2010-09-30'

    fs = pd.read_csv(f"./data/date_regression/{strdate}.csv")
    if next_strdate != '2024-06-30':
        next_fs = pd.read_csv(f"./data/date_regression/{next_strdate}.csv")

    mem = ut.get_portfolio_memory(stocks,strdate,next_strdate,fs,next_fs)

    print(mem)
    print(ut.get_sharpe_ratio(mem))

    
