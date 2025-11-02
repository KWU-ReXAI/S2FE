import pandas as pd


cagr = []
sharpe_ratio = []
mdd = []

for trainNum in range(1):
    dir = f"./result/result_S2FE_SectorAll/train_result_dir_{trainNum+1}/train_result_file_{trainNum+1}.csv" # 훈련 후 검증결과과 저장파일
    df = pd.read_csv(dir, index_col=0, header=0)

    cagr.append(df["Average"].iloc[0])
    sharpe_ratio.append(df["Average"].iloc[1])
    mdd.append(df["Average"].iloc[2])

avg_cagr = sum(cagr) / len(cagr)
avg_sharpe_ratio = sum(sharpe_ratio) / len(sharpe_ratio)
avg_mdd = sum(mdd) / len(mdd)

print({"CAGR": avg_cagr, "SHARPE RATIO": avg_sharpe_ratio, "MDD": avg_mdd})