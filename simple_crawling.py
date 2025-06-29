from newspaper import Article
import pandas as pd
import os
from tqdm import tqdm

base_url = 'data_kr/video'

# 에러 로그 파일 경로 (base_url 아래에 생성)
LOG_FILE = os.path.join(base_url, 'crawling_error_log.txt')

def crawling_news(url : str, news_path : str):
	# Article 객체 생성 및 다운로드
	try:
		article = Article(url, language='ko')  # 한글 기사라면 language='ko'
		article.download()
		article.parse()
		with open(news_path, "w", encoding='utf-8') as file:
			file.write(article.text)
	except Exception as e:
		with open(LOG_FILE, 'a', encoding='utf-8') as f:
			f.write(f"{url}\tREQUEST_FAILED\t{str(e)}\n")

if __name__ == "__main__":
	df = pd.read_csv(f"{base_url}/뉴스 기사 수집본.csv")
	for row in tqdm(df.itertuples(), total=len(df)):
		if pd.isna(row.url):
			continue

		# 저장할 디렉토리 및 파일명 설정
		fpath = os.path.join(base_url, "text", str(row.sector), str(row.code).zfill(6))
		os.makedirs(fpath, exist_ok=True)
		fname = os.path.join(fpath, f"{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt")

		crawling_news(row.url, fname)