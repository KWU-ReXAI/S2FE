import pandas as pd
import os
from tqdm import tqdm
base_url = '../data_kr/video'

if __name__ == "__main__":
	df = pd.read_csv(f"{base_url}/자료 수집 최종본.csv")
	df = df[df['url'].notna()]
	df_new = pd.DataFrame(columns=df.columns)
	for row in tqdm(df.itertuples(index=False), total=len(df)):
		# 저장할 디렉토리 및 파일명 설정
		fpath = os.path.join(base_url, "text", str(row.sector), str(row.code).zfill(6))
		os.makedirs(fpath, exist_ok=True)
		fname = os.path.join(fpath, f"{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt")

		file_size = os.path.getsize(fname)
		if file_size < 200:
			df_new.loc[len(df_new)] = row
	sli = int(len(df_new) / 5)
	name = ['태완', '동우', '재현', '제영', '종혁']
	for i in range(5):
		if i == 4:
			df_tmp = df_new.iloc[i * sli:].copy(deep=True)
		else:
			df_tmp = df_new.iloc[i * sli : (i + 1) * sli].copy(deep=True)
		df_tmp.to_csv(f"{name[i]}_수동_크롤링.csv", index=False, encoding='utf-8-sig')