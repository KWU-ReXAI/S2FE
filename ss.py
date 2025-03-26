import pandas as pd

# symbol.csv 파일 읽기 (인코딩이 필요하면 'utf-8-sig' 또는 'cp949' 등으로 조정하세요)
df = pd.read_csv("./data_kr/symbol.csv", encoding="utf-8-sig")

print(df["sector"].unique())