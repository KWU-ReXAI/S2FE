import pandas as pd

base_url = '../data_kr/video'
df_target = pd.read_csv(f"{base_url}/자료 수집 통합본_origin.csv")
df_source = pd.read_csv(f"{base_url}/동영상 수집 통합본 최신.csv")

df_target['after'] = df_target['filtering'].str.extract(r'after:([0-9\-]+)')
df_target['before'] = df_target['filtering'].str.extract(r'before:([0-9\-]+)')
df_target['after'] = pd.to_datetime(df_target['after'])
df_target['before'] = pd.to_datetime(df_target['before'])

df_source['upload_date'] = pd.to_datetime(df_source['upload_date'])

for row in df_source.itertuples():
	df_target.loc[(df_target['code'] == row.code)
				 & (df_target['after'] <= row.upload_date)
				 & (df_target['before'] >= row.upload_date), ['url', 'category', 'upload_dt']] = [row.url, row.category, row.upload_date]
df_target['upload_dt'] = pd.to_datetime(df_target['upload_dt'], errors='coerce')
df_target['upload_dt'] = df_target['upload_dt'].dt.strftime("%Y-%m-%d")
df_target.drop(['after', 'before'], axis=1, inplace=True)

df_target.to_csv('자료 수집 통합본.csv', encoding='utf-8-sig', index=False)