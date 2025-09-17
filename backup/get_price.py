from pykrx import stock
from pykrx import bond
import pykrx
import FinanceDataReader as fdr
import pandas as pd
import os

# symbol.csv 파일에서 종목 코드를 문자열로 읽고, 6자리로 맞춥니다.
symbol_df = pd.read_csv('./data_kr/symbol.csv', encoding='utf-8-sig', dtype={'code': str})
ticker_list = symbol_df['code'].astype(str).str.zfill(6)

os.makedirs('./data_kr/price', exist_ok=True)

start_date = '2015-01-01'
end_date = '2025-09-17'
for ticker in ticker_list:
    try:
        df_price = stock.get_market_ohlcv_by_date(fromdate=start_date,
                                          todate=end_date,
                                          ticker=ticker)
        df_price.to_csv(f'./data_kr/price/{ticker}.csv', index=True, encoding='utf-8-sig')
        print(f"{ticker}의 가격 데이터를 저장했습니다.")
    except Exception as e:
        print(f"{ticker}의 가격 데이터를 가져오는 중 오류 발생: {e}")


# KOSPI200 지수(코드: 1028)의 일별 가격 데이터 가져오기
df_ks200 = stock.get_index_ohlcv_by_date(fromdate=start_date,
                                         todate=end_date,
                                         ticker="1028")  # KOSPI200

# 결과 저장
df_ks200.to_csv('./data_kr/price/KS200.csv', encoding='utf-8-sig')
print("KS200 지수 데이터를 저장했습니다.")

'''

"""
import pandas as pd
import os

# symbol.csv에서 종목 코드를 문자열로 읽고, 6자리 형식으로 맞춥니다.
symbol_df = pd.read_csv("./data_kr/symbol.csv", encoding="utf-8-sig", dtype={'code': str})
ticker_list = symbol_df['code'].tolist()

price_folder = "./data_kr/price"

for ticker in ticker_list:
    file_path = os.path.join(price_folder, f"{ticker}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, encoding="utf-8-sig")
        if "Change" in df.columns:
            # Change 컬럼의 값에 100을 곱함
            df["Change"] = df["Change"] * 100
            df.to_csv(file_path, encoding="utf-8-sig")
            print(f"{ticker}의 Change 컬럼에 100을 곱했습니다.")
        else:
            print(f"{ticker} 파일에 'Change' 컬럼이 없습니다.")
    else:
        print(f"{file_path} 파일이 존재하지 않습니다.")"""
'''