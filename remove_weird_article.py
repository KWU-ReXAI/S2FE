import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os

base_url = './data_kr/video'
pre_url = './preprocessed_data/llm'

df = pd.read_csv(f"{base_url}/뉴스 기사 수집본.csv", encoding='utf-8-sig')
text = pre = mix = 0
for row in tqdm(df.itertuples(), total=len(df)):
	after = datetime.strptime(row.after, "%Y-%m-%d")
	before = datetime.strptime(row.before, "%Y-%m-%d")
	if pd.isna(row.upload_dt):
		continue
	dt = datetime.strptime(row.upload_dt, "%Y-%m-%d")
	cond = after <= dt <= before
	if not cond:
		# 뉴스 기사 수집본.csv 편집
		df.loc[row.Index, ["url", "upload_dt"]] = None, None
		df.to_csv(f"{base_url}/뉴스 기사 수집본.csv", encoding='utf-8-sig', index=False)
		# 원본 기사 텍스트 삭제
		fpath = os.path.join(base_url, "text", str(row.sector), str(row.code).zfill(6))
		fname = os.path.join(fpath, f"{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt")
		if os.path.exists(fname):
			text += 1
			# os.remove(fname)
		# text 예측 파일 삭제
		fpath = os.path.join(pre_url, "predict_text", str(row.sector), str(row.code).zfill(6))
		fname = os.path.join(fpath, f"{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt")
		if os.path.exists(fname):
			pre += 1
			# os.remove(fname)
		# text 예측 csv(종목별) 편집
		fpath = os.path.join(pre_url, "predict_text", str(row.sector), str(row.code).zfill(6))
		fname = os.path.join(fpath, f"{str(row.code).zfill(6)}.csv")
		tmp = pd.read_csv(fname, encoding='utf-8-sig')
		tmp.loc[((tmp['year'] == row.year)
				 & (tmp['month'] == row.month)
				 & (tmp['week'] == row.week)), ]
		fname = os.path.join(fpath, f"{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt")
		# text 예측 csv(전체) 편집
		fpath = os.path.join(pre_url, "predict_text", str(row.sector), str(row.code).zfill(6))
		fname = os.path.join(fpath, f"{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt")

		# mix 예측 파일 삭제
		fpath = os.path.join(pre_url, "predict_text", str(row.sector), str(row.code).zfill(6))
		fname = os.path.join(fpath, f"{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt")
		if os.path.exists(fname):
			pre += 1
			# os.remove(fname)
		# mix 예측 csv(종목별) 편집
		fpath = os.path.join(pre_url, "predict_text", str(row.sector), str(row.code).zfill(6))
		fname = os.path.join(fpath, f"{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt")
		# mix 예측 csv(전체) 편집
		fpath = os.path.join(pre_url, "predict_text", str(row.sector), str(row.code).zfill(6))
		fname = os.path.join(fpath, f"{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt")


