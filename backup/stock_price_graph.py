import pandas as pd
import matplotlib.pyplot as plt

# 예시 종목 이름
csv = pd.read_csv(f"../result/test_result_dir/test_selected_stocks_p3_1.csv")
unique_date = csv.iloc[:, 0].unique().tolist()

dates = ['2023-04-01', '2023-05-16', '2023-08-15', '2023-11-15', '2024-03-22']

for idx, date in enumerate(unique_date):
    date_csv = csv[csv['0'] == date]
    stock_names = date_csv.iloc[0, 2:].to_list()
    stock_names = [str(int(x)).zfill(6) for x in stock_names]

    start_date = pd.to_datetime(dates[idx])
    end_date = pd.to_datetime(dates[idx+1])
    dfs = []
    for stock in stock_names:
        dfs.append(pd.read_csv(f"../data_kr/price/{stock}.csv"))

    plt.figure(figsize=(12, 6))
    # 각 종목별로 그래프 그리기
    for i, df in enumerate(dfs):
        df['날짜'] = pd.to_datetime(df['날짜'])
        df_filtered = df[(df['날짜'] >= start_date) & (df['날짜'] <= end_date)]
        plt.plot(df_filtered['날짜'], df_filtered['종가'], label=stock_names[i])

    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{start_date.strftime("%Y-%m-%d")} ~ {end_date.strftime("%Y-%m-%d")}_price.png")