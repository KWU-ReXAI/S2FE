import pandas as pd



cagr = []
sharpe_ratio = []
mdd = []
		# trainNum별로 결과가 각각 저장돼있어서 다 읽어오는 부분분
for trainNum in range(10):
	dir = f"./result/result_S3CE_SectorAll/train_result_dir_{trainNum+1}/train_result_file_{trainNum+1}.csv" # 훈련 후 검증결과과 저장파일
	df = pd.read_csv(dir, index_col=0, header=0)

	cagr.append(df["Average"][0])
	sharpe_ratio.append(df["Average"][1])
	mdd.append(df["Average"][2])

		# 읽어오 결과로 평균내기기
avg_cagr = sum(cagr) / len(cagr)
avg_sharpe_ratio = sum(sharpe_ratio) / len(sharpe_ratio)
avg_mdd = sum(mdd) / len(mdd)
print({"CAGR": avg_cagr, "SHARPE RATIO": avg_sharpe_ratio, "MDD": avg_mdd})