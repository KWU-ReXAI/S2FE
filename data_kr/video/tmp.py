import pandas as pd

df = pd.read_csv('error_rows.csv', encoding='utf-8-sig')
print(df[df['url'].isna()])
# origin = pd.read_csv('뉴스 영상 수집본.csv', encoding='utf-8-sig')
#
# for id, row in df.iterrows():
# 	name, after, before = origin.loc[(origin['year'] == df.iloc[id]['year'])
# 								 & (origin['month'] == df.iloc[id]['month'])
# 								 & (origin['week'] == df.iloc[id]['week'])
# 								 & (origin['code'] == df.iloc[id]['code']), ['name', 'after', 'before']].squeeze().tolist()
# 	df.at[id, 'name'] = name
# 	df.at[id, 'after'] = after
# 	df.at[id, 'before'] = before
df.to_csv('error_rows.csv', encoding='utf-8-sig', index=False)