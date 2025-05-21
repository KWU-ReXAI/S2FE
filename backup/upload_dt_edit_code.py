import pandas as pd

target = pd.read_csv('../data_kr/video/동영상 수집 통합본_ver2.csv')
source = pd.read_csv('../data_kr/video/동영상 찐 통합본.csv')

for row in source.itertuples():
	target.loc[(target['year'] == row.year) & (target['quarter'] == row.quarter) & (target['code'] == row.code)& (target['category'] == 'article'), 'upload_date'] = row.upload_dt

target.to_csv('동영상 수집 통합본 최신.csv', index=False, encoding='utf-8-sig')