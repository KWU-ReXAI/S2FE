import pandas as pd

dir = f"./result/result_S3CE_SectorAll/test_result_dir/test_result_file.csv"  # 훈련 후 검증결과과 저장파일
df = pd.read_csv(dir, header=[0, 1], index_col=0)
df_avg = df["Average"].mean(axis=1)

print(f"CAGR: {df_avg['CAGR']}, SHARPE RATIO: {df_avg['Sharpe Ratio']}, MDD: {df_avg['MDD']}")