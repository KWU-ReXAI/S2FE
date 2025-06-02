import pandas as pd
from tqdm import tqdm
url = '../data_kr/video'

target = pd.read_csv(f"{url}/자료 수집 통합본_최신.csv")
sources = [pd.read_csv(f"{url}/김재현_검토완료.csv"), pd.read_csv(f"{url}/김태완_검토완료.csv"), pd.read_csv(f"{url}/이종혁_검토완료.csv"), pd.read_csv(f"{url}/자료수집통합본_최종수정_제영.csv")]

for source in tqdm(sources):
	for row in source.itertuples():
		target.loc[((target['year'] == row.year)
				   & (target['month'] == row.month)
				   & (target['week'] == row.week)
				   & (target['code'] == row.code)), ['url', 'category', 'upload_dt']] = [row.url, row.category, row.upload_dt]
target.to_csv(f"{url}/자료 수집 통합본_최최신.csv", index=False, encoding='utf-8-sig')