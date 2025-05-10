from dotenv import load_dotenv
import os
from pathlib import Path
from datetime import datetime, timedelta

import torch
from googleapiclient.discovery import build
import yt_dlp
import whisper
import re
from tqdm import tqdm

import google.generativeai as genai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

import pandas as pd

load_dotenv()  # .env 파일에서 환경변수 불러오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# whisper 모델 로드 (base, small, medium, large 중 선택 가능)
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("medium").to(device)

# gemini 모델 로드
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# GPT 모델 로드
gpt_model = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=OPENAI_API_KEY)

# YOUTUBE 빌드
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# -----------------------------
# 유튜브 채널명으로 채널ID 추출
# 입력 예: "한국경제TV" 또는 "@wowtv"
# 출력: 채널 ID
# -----------------------------
def get_channel_id(channel_name):
	res = youtube.search().list(
		q=channel_name,
		type="channel",
		part="snippet",
		maxResults=1
	).execute()

	# 결과에서 채널 ID 추출
	if res.get("items"):
		channel_id = res["items"][0]["snippet"]["channelId"]
		print("📺 채널 ID:", channel_id)
	else:
		print("채널을 찾을 수 없습니다.")

	return channel_id

# -----------------------------
# 채널 ID로 업로드 playlist ID 얻기
# 입력: channel id
# 출력: playlist id
# -----------------------------
def get_uploads_playlist_id(channel_id):
    res = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()
    return res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

# -----------------------------
# playlist ID로 플레이리스트 내 모든 video ID와 업로드 날짜 얻기
# 입력: playlist id
# 출력: type: 튜플 리스트  ex) [(id1, date1, 조회수1), (id2, date2, 조회수2) ...]
# -----------------------------
def get_video_datas_from_playlist(playlist_id):
    video_list = []
    next_page_token = None
    video_id_date_pairs = []

    # 1. playlistItems API로 video ID + 업로드 날짜 가져오기
    while True:
        res = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        for item in res["items"]:
            snippet = item["snippet"]
            video_id = snippet["resourceId"]["videoId"]
            published_at = snippet["publishedAt"]
            video_id_date_pairs.append((video_id, published_at))

        next_page_token = res.get("nextPageToken")
        if not next_page_token:
            break

    # 2. video ID로 조회수 가져오기
    for i in range(0, len(video_id_date_pairs), 50):
        batch = video_id_date_pairs[i:i+50]
        ids_only = [vid for vid, _ in batch]

        res = youtube.videos().list(
            part="statistics",
            id=",".join(ids_only)
        ).execute()

        stats = {item["id"]: int(item["statistics"].get("viewCount", 0)) for item in res["items"]}

        # 3. 튜플로 저장: (video_id, published_at, view_count)
        for video_id, published_at in batch:
            view_count = stats.get(video_id, 0)
            video_list.append((video_id, published_at, view_count))

    return video_list

# -----------------------------
# video ids로 업로드 날짜 필터링하기
# 입력: video_datas(type 튜플 리스트), 시작날짜, 끝날짜 ("2001-04-30" 형식으로 입력)
# 출력: 필터링 된 video_datas(type 튜플 리스트)
# -----------------------------
def filter_by_date(video_datas, start_date, end_date):
    # 문자열 → datetime 객체로 변환
    start_dt = datetime.strptime(start_date + "T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ")
    end_dt = datetime.strptime(end_date + "T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ")

    filtered = []
    for video_id, published_at, view_count in video_datas:
        pub_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
        if start_dt <= pub_date < end_dt:
            filtered.append((video_id, published_at, view_count))

    return filtered
    
# -----------------------------
# 조회수 기준 필터링
# 입력: video_datas(type 튜플 리스트), 최소조회수
# 출력: video_datas(type 튜플 리스트)
# -----------------------------
def filter_videos_by_view_count(video_tuples, min_views=0, max_views=float("inf")):
    return [
        (video_id, published_at, view_count)
        for video_id, published_at, view_count in video_tuples
        if min_views <= view_count <= max_views
    ]
    
# -----------------------------
# 채널영상을 날짜, 조회수 기준 필터링
# 입력: channel_id, 기간, 최소조회수
# 출력: video_datas(type 튜플 리스트)
# -----------------------------
def get_filtered_videos_by_channel(channel_id, start_date, end_date, min_views=0):
    video_results = []
    next_page_token = None

    # 날짜 문자열 → datetime 객체 → ISO 형식으로 변환
    start_iso = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=7)).isoformat("T") + "Z"
    end_iso = datetime.strptime(end_date, "%Y-%m-%d").isoformat("T") + "Z"

    # 1. search().list() → video ID + publishedAt
    temp_videos = []
    while True:
        res = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            publishedAfter=start_iso,
            publishedBefore=end_iso,
            maxResults=50,
            pageToken=next_page_token,
            type="video",
            order="date"
        ).execute()

        for item in res["items"]:
            video_id = item["id"]["videoId"]
            published_at = item["snippet"]["publishedAt"]
            temp_videos.append((video_id, published_at))

        next_page_token = res.get("nextPageToken")
        if not next_page_token:
            break

    # 2. videos().list() → 조회수 붙이기
    for i in range(0, len(temp_videos), 50):
        batch = temp_videos[i:i+50]
        ids = [vid for vid, _ in batch]

        res = youtube.videos().list(
            part="statistics",
            id=",".join(ids)
        ).execute()

        stats = {
            item["id"]: int(item["statistics"].get("viewCount", 0))
            for item in res["items"]
        }

        for video_id, published_at in batch:
            view_count = stats.get(video_id, 0)
            if view_count >= min_views:
                video_results.append((video_id, published_at, view_count))

    return video_results

# -----------------------------
# 유튜브 영상 검색
# date 이전의 query 검색 결과들만 보여줌 
# date는 '2025-04-11' 형식으로 입력
# -----------------------------
def search_videos(query, date):
	date += 'T00:00:00Z' # ISO 8601 형식으로 변경

	# Step 1: 검색
	search_res = youtube.search().list(
		q=query,
		part='snippet',
		type='video',
		maxResults=50,
		order='date',
		publishedBefore=date
	).execute()
	
	# Step 2: videoId 수집
	video_ids = [item['id']['videoId'] for item in search_res['items']]
	if not video_ids:
		print("조건에 맞는 영상 없음")

	return video_ids

# -----------------------------
# 유튜브 링크 or id에서 음성파일 추출
# INPUT: method(link or id), youtube id or link, 오디오 다운로드 경로
# -----------------------------
def extract_video_audio(method, video_id, audio_dir):
	os.makedirs('./audio', exist_ok=True)
	
	if method == "link":
		url = video_id
	elif method == "id":
		url = "https://www.youtube.com/watch?v=" + video_id
	else:
		print('method인자로 link or id를 입력하세요')
		return False

	ydl_opts = {
		'format': 'bestaudio/best',
		'outtmpl': f'{audio_dir}.%(ext)s',
		'postprocessors': [{
			'key': 'FFmpegExtractAudio',
			'preferredcodec': 'mp3',
		}]
	}

	try:
		with yt_dlp.YoutubeDL(ydl_opts) as ydl:
			ydl.download([url])
		print("음성파일 다운로드 성공")
		return True
	except:
		print("음성파일 다운로드 실패")
		return False

# -----------------------------
# 음성파일을 텍스트로 변환
# INPUT: youtube id, 오디오 다운로드 경로
# -----------------------------
def audio2text(audio_dir):
	# 음성 파일 STT 수행
	result = whisper_model.transcribe(f'{audio_dir}.mp3', language="ko")  # wav, mp4 등도 OK

	# 텍스트 출력
	print(result["text"])

	return result["text"]

# -----------------------------
# 자막 전처리 함수
# -----------------------------
def clean_srt(srt_text: str) -> str:
	srt_text = re.sub(r"(좀|그냥|뭐랄까|그러니까|아니|약간|뭔가|뭐냐면요)", "", srt_text)
	return srt_text


# ------------------------
# GPT-4o 요약
# ------------------------
def summarize_text(text: str) -> str:
	prompt = f"""
The following is a transcript from a YouTube video about economics. Please summarize the content by extracting key economic ideas, arguments, data, and cited examples.  
Your summary should be **concise and well-organized**, capturing the **main points** and **logical structure** of the video.  
A **section-based summary** or **bullet-point outline** is preferred if appropriate.
Transcript: {text}
"""
	response = gpt_model([HumanMessage(content=prompt)])
	return response.content.strip()



# ------------------------
# GPT-4o 등락 예측
# ------------------------
def predict_market_from_summary(summary: str, stock: str) -> str:
	prompt = f"""
You are a financial analyst AI.

Based on the following summary of an economics-related YouTube video transcript, assess whether the stock of **{stock}** is likely to go up, down, or remain neutral based on the video's content. 

If the content has **no meaningful connection** to the stock or its industry, clearly state:  
"이 영상 요약본은 {stock} 주가와 관련이 없습니다."

Respond in Korean with a short but insightful analysis.

Summary:
{summary}
"""
	response = gpt_model([HumanMessage(content=prompt)])
	return response.content.strip()


# ------------------------
# 앞의 모든 함수를 이용한 최종 함수
#
# 주식과 날짜 입력하면 주가전망 예측
# date는 '2025-04-11' 형식으로 입력
# ------------------------
def predict_market(stock: str, date: str) -> str:
	SEARCH_QUERY = f'{stock} 주가전망'
	BEFORE_DATE = date
	print("SEARCH_QUERY:", SEARCH_QUERY)
	video_id = search_videos(SEARCH_QUERY, BEFORE_DATE)
	if video_id == None:
		return "middle"

	# 유튜브 영상 음성 추출
	audio_dir = f'audio/{stock}_{BEFORE_DATE}'
	if not extract_video_audio(video_id, audio_dir):
		return "middle"
	text = audio2text(audio_dir)

	# 자막 전처리
	cleaned = clean_srt(text)
	print("전처리:\n", cleaned)

	# 전처리 자막 요약
	summary = summarize_text(cleaned)

	# 주식 등락 예측
	prediction = predict_market_from_summary(summary, stock)

	print("\n📄 GEMINI 요약 결과:\n", summary)
	print("\n📈 GPT-4 예측 결과:\n\n", prediction)

	if prediction == '오를 가능성 있음':
		return 'up'
	elif prediction == '오를 가능성 낮음':
		return 'down'
	else:
		return 'middle'




# ------------------------
# 모든 분기의 범위 구하기
# disclosure_date_range.csv 파일 생성
# ------------------------
def get_disclosure_range():
    # 모든 종목의 모든 분기 공시일을 하나의 파일로
	root_path = Path('./data_kr/merged')
	all_symbols_disclosure = pd.DataFrame()
	for file_path in root_path.rglob("*.csv"):
		df_ = pd.read_csv(file_path)
		df_ = df_[["code", "name", "year", "quarter", "disclosure_date"]]
		all_symbols_disclosure = pd.concat([all_symbols_disclosure, df_])


	years = [2015] + ([y for y in range(2016, 2025) for _ in range(4)])
	quarters = ["Q4"] + ([q for _ in range(2016, 2025) for q in ["Q1", "Q2", "Q3", "Q4"]])
	df_disclosure = pd.DataFrame({
		"year": years,
		"quarter": quarters,
		"min_disclosure_date": [None] * len(years),
		"max_disclosure_date": [None] * len(years)
	})

	for i, row in enumerate(df_disclosure.itertuples()):
		disclosures = all_symbols_disclosure[(all_symbols_disclosure["year"] == row.year) & (all_symbols_disclosure["quarter"] == row.quarter)]["disclosure_date"]
		df_disclosure.loc[i, "min_disclosure_date"] = disclosures.min()
		df_disclosure.loc[i, "max_disclosure_date"] = disclosures.max()

	os.makedirs('./data_kr/audio', exist_ok=True)
	df_disclosure.to_csv("./data_kr/audio/disclosure_date_range.csv", index=False)
 
# ------------------------
# disclosure range를 만족하고 조회수가 min_view_cnt 이상인 video id 구하기
# channel_name: str
# min_view_cnt: int
# data_kr/audoi/에 연도-분기.csv 파일 생성
# ------------------------
def get_video_datas(channel_name, min_view_cnt):
	channel_id = get_channel_id(channel_name) 
	dir = f'data_kr/audio/{channel_name}'
	os.makedirs(dir, exist_ok=True)
 
	df_disclosure = pd.read_csv('data_kr/audio/disclosure_date_range.csv')
	for row in df_disclosure.itertuples():
		start = datetime.strptime(row.min_disclosure_date, "%Y-%m-%d")
		start -= timedelta(days=7)
		start = start.strftime("%Y-%m-%d")
		end = row.max_disclosure_date

		video_datas = get_filtered_videos_by_channel(channel_id, start, end, min_view_cnt)
		year_quarter = f'{row.year}-{row.quarter}'
		os.makedirs(f'{dir}/{year_quarter}', exist_ok=True)
		pd.DataFrame(video_datas, columns=['video_id', 'published_at', 'view_count']).to_csv(f'{dir}/{year_quarter}/{year_quarter}.csv', index=False)
    
if __name__ == "__main__":    
    # ### 오디오 다운로드 ###
	# df = pd.read_csv('data_kr/video/동영상 수집 통합본.csv')
	# for row in df.itertuples():
	# 	if pd.isna(row.url) or row.url == '':
	# 		continue

	# 	code = str(row.code).zfill(6)
	# 	audio_dir = f'data_kr/video/audio/{row.sector}/{code}/'
	# 	text_dir = f'data_kr/video/text/{row.sector}/{code}/'
	# 	os.makedirs(audio_dir, exist_ok=True)

	# 	if extract_video_audio("link", row.url, audio_dir + f'{row.year}-{row.quarter}'):
	# 		with open('data_kr/video/log.txt', "a", encoding="utf-8") as log_file:
	# 			timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	# 			log_file.write(f"{timestamp} audio download completed: {audio_dir + f'{row.year}-{row.quarter}'}\n")
	# 	else:
	# 		with open('data_kr/video/log.txt', "a", encoding="utf-8") as log_file:
	# 			timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	# 			log_file.write(f"{timestamp} audio download error: {audio_dir + f'{row.year}-{row.quarter}'}\n")
	
	# ### 텍스트로 변환 ###
	# df = pd.read_csv('data_kr/video/동영상 수집 통합본.csv')
	# for row in df.itertuples():
	# 	if pd.isna(row.url) or row.url == '':
	# 		continue

	# 	code = str(row.code).zfill(6)
	# 	audio_dir = f'data_kr/video/audio/{row.sector}/{code}/'
	# 	text_dir = f'data_kr/video/text/{row.sector}/{code}/'
	# 	os.makedirs(text_dir, exist_ok=True)
		
	# 	try:
	# 		text = audio2text(audio_dir + f'{row.year}-{row.quarter}')
	# 		with open(text_dir + f'{row.year}-{row.quarter}.txt', "w", encoding="utf-8") as f:
	# 			f.write(text)
	# 		with open('data_kr/video/log.txt', "a", encoding="utf-8") as log_file:
	# 			timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	# 			log_file.write(f"{timestamp} whisper completed: {text_dir + f'{row.year}-{row.quarter}'}\n")
	# 	except Exception as e:
	# 		with open('data_kr/video/log.txt', "a", encoding="utf-8") as log_file:
	# 			timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	# 			log_file.write(f"{timestamp} whisper error: {text_dir + f'{row.year}-{row.quarter}'}\n")

	# ### 텍스트 token 수 확인  ###
	# import tiktoken
	# # 예: GPT-4용 인코더 불러오기
	# encoding = tiktoken.encoding_for_model("gpt-4")
	# total_tokens = 0

	# df = pd.read_csv('data_kr/video/동영상 수집 통합본.csv')
	# for row in df.itertuples():
	# 	if pd.isna(row.url) or row.url == '':
	# 		continue

	# 	code = str(row.code).zfill(6)
	# 	audio_dir = f'data_kr/video/audio/{row.sector}/{code}/'
	# 	text_dir = f'data_kr/video/text/{row.sector}/{code}/'
	# 	os.makedirs(text_dir, exist_ok=True)
		
	# 	try:
	# 		with open(text_dir + f'{row.year}-{row.quarter}.txt', "r", encoding="utf-8") as file:
	# 			text = file.read()
	# 			tokens = encoding.encode(text)
	# 			total_tokens += len(tokens)
	# 		with open('data_kr/video/num_token.txt', "a", encoding="utf-8") as log_file:
	# 			timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	# 			log_file.write(f"{text_dir}{row.year}-{row.quarter} len: {len(text)}, token: {len(tokens)}\n")
	# 	except Exception as e:
	# 		with open('data_kr/video/num_token.txt', "a", encoding="utf-8") as log_file:
	# 			timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	# 			log_file.write(f"{text_dir}" + f"{row.year}-{row.quarter} error: " + e + "\n")

	# with open('data_kr/video/num_token.txt', "a", encoding="utf-8") as log_file:
	# 	log_file.write(f"total tokens: {total_tokens}\n")

	# ### LLM으로 영상 자막 요약 ###
	# df = pd.read_csv('data_kr/video/동영상 수집 통합본.csv')
	# for row in tqdm(df.itertuples(), total=len(df), desc="LLM summarizing"):
	# 	if pd.isna(row.url) or row.url == '':
	# 		continue

	# 	code = str(row.code).zfill(6)
	# 	text_dir = f'data_kr/video/text/{row.sector}/{code}/'
	# 	summary_dir = f'preprocessed_data/llm/summary/{row.sector}/{code}/'
	# 	os.makedirs(summary_dir, exist_ok=True)

	# 	try:
	# 		with open(text_dir + f'{row.year}-{row.quarter}.txt', "r", encoding="utf-8") as file:
	# 			text = file.read()
	# 		summary = summarize_text(text)
	# 		with open(summary_dir + f'{row.year}-{row.quarter}.txt', "w", encoding="utf-8") as f:
	# 			f.write(summary)
	# 		with open('preprocessed_data/llm/summary/log.txt', "a", encoding="utf-8") as log_file:
	# 			timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	# 			log_file.write(f"{timestamp} summary completed: {summary_dir + f'{row.year}-{row.quarter}'}\n")
	# 	except Exception as e:
	# 		with open('preprocessed_data/llm/summary/log.txt', "a", encoding="utf-8") as log_file:
	# 			timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
	# 			log_file.write(f"{timestamp} summary error: {summary_dir + f'{row.year}-{row.quarter}'}\n")

	### LLM으로 자막요약을 통해 등락 예측 ###
	df = pd.read_csv('data_kr/video/동영상 수집 통합본.csv')
	for row in tqdm(df.itertuples(), total=len(df), desc="LLM predicting"):
		if pd.isna(row.url) or row.url == '':
			continue

		code = str(row.code).zfill(6)	
		name = row.name
		summary_dir = f'preprocessed_data/llm/summary/{row.sector}/{code}/'
		predict_dir = f'preprocessed_data/llm/predict/{row.sector}/{code}/'
		os.makedirs(summary_dir, exist_ok=True)

		try:
			with open(summary_dir + f'{row.year}-{row.quarter}.txt', "r", encoding="utf-8") as file:
				summary = file.read()
			predict = predict_market_from_summary(summary, f'{name}({code})')
			with open(predict_dir + f'{row.year}-{row.quarter}.txt', "w", encoding="utf-8") as f:
				f.write(predict)
			with open('preprocessed_data/llm/predict/log.txt', "a", encoding="utf-8") as log_file:
				timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
				log_file.write(f"{timestamp} predict completed: {predict_dir + f'{row.year}-{row.quarter}'}\n")
		except Exception as e:
			with open('preprocessed_data/llm/predict/log.txt', "a", encoding="utf-8") as log_file:
				timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
				log_file.write(f"{timestamp} predict error: {predict_dir + f'{row.year}-{row.quarter}'}\n")