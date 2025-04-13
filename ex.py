import pandas as pd
from predict_stock_outlook import predict_market

selected_stocks = pd.read_csv("./result/train_result_dir_1/train_selected_stocks_p1.csv")
selected_stocks = selected_stocks.iloc[:, 1:]  # 첫 번째 열 제거

print(selected_stocks)

all_stocks = pd.read_csv("./data_kr/symbols.csv")

for i in range(4): # i분기
	date = quater2date(f"Q{i + 1}")
	stocks =  selected_stocks.iloc[i, :]
	stocks.dropna(inplace=True)

	final_stocks = []
	for stock in stocks:
		stock = str(int(stock)).zfill(6)
		stock_name = all_stocks[all_stocks["corp_code"]==50]["corp_name"][0]
		
		prediction = predict_market(stock_name, date)
		if prediction == "up":
			final_stocks.append(stock)

		# 저장