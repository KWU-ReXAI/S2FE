import pandas as pd

base_url = '../data_kr/video'

df = pd.read_csv(f"{base_url}/merged.csv")

# 조건 1: url과 category 모두 NaN
cond1 = df['url'].isna() & df['category'].isna()

# 조건 2: url에 'youtube' 포함 AND category == 'video'
cond2 = df['url'].str.contains('youtube', case=False, na=False) & (df['category'] == 'video')

# 조건 3: url에 'youtube' 미포함 AND category == 'article'
cond3 = ~df['url'].str.contains('youtube', case=False, na=False) & (df['category'] == 'article')


# 최종 조건: cond1 OR cond2 OR cond3
final_condition = cond1 | cond2 | cond3

# 조건에 해당하지 않는 행만 선택
filtered_df = df[final_condition]

print(filtered_df)