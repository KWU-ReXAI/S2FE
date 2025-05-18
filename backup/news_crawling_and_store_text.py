from newspaper import Article
import pandas as pd

base_url = '../data_kr/video'

def crawling_news(url: str):
	# Article 객체 생성 및 다운로드
	try:
		article = Article(url, language='ko')  # 한글 기사라면 language='ko'
		article.download()
		article.parse()
	except Exception as e:
		print('에러 url: ', url)
	return article.text, article.publish_date.strftime("%Y-%m-%d") if article.publish_date is not None else None

def save_news(row):
	if pd.isna(row.text):
		return
	news_path = f"{base_url}/text/{row.sector}/{str(int(row.code)).zfill(6)}/{int(row.year)}-{row.quarter}.txt"
	with open(news_path, "w", encoding='utf-8') as file:
		file.write(row.text)
	print(f"{row.year}-{row.quarter} {row.code} 추출 완료")

df = pd.read_csv(f"{base_url}/동영상 수집 통합본 최신.csv")

df[['text', 'upload_dt']] = df.apply(lambda row: pd.Series(crawling_news(row['url']))
	if row['category'] == 'article' else pd.Series([pd.NA, pd.NA]), axis=1)

df.apply(save_news, axis=1)

df.drop('text', inplace=True, axis=1)

df.to_csv('동영상 찐 통합본.csv', encoding='utf-8-sig', index=False)