from dotenv import load_dotenv
import os

from googleapiclient.discovery import build
import yt_dlp
import whisper
import re

import google.generativeai as genai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()  # .env 파일에서 환경변수 불러오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


# -----------------------------
# 유튜브 영상 검색
# date 이전의 query 검색 결과들만 보여줌 
# date는 '2025-04-11' 형식으로 입력
# -----------------------------
def search_videos(query, date):
	date += 'T00:00:00Z' # ISO 8601 형식으로 변경
	youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

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
		exit()

	# Step 3: 상세 정보 조회 (조회수 등)
	video_details = youtube.videos().list(
		part='statistics,snippet',
		id=','.join(video_ids)
	).execute()

	# Step 4: 필터링 & 정렬
	filtered = []
	for item in video_details['items']:
		stats = item['statistics']
		snippet = item['snippet']
		publish_date = snippet['publishedAt']

		if publish_date < date:
			filtered.append({
				'videoId': item['id'],
				'title': snippet['title'],
				'views': int(stats.get('viewCount', 0)),
				'publishedAt': publish_date
			})

	if not filtered:
		print("필터 조건에 맞는 영상 없음")
		exit()

	# Step 5: 조회수 기준 최상위 영상 선택
	top_video = max(filtered, key=lambda x: x['views'])
	video_id = top_video['videoId']
	print(f"🎬 가장 인기 있는 영상: {top_video['title']}")
	print(f"🔗 https://www.youtube.com/watch?v={video_id}")
	print(f"👀 조회수: {top_video['views']}")
	print(f"📅 업로드: {top_video['publishedAt']}")

	return video_id

# -----------------------------
# 유튜브 id에서 음성파일 추출
# INPUT: youtube id, 오디오 다운로드 경로
# -----------------------------
def extract_video_audio(video_id, audio_dir):
	os.makedirs('./audio', exist_ok=True)
	
	url = "https://www.youtube.com/watch?v=" + video_id

	ydl_opts = {
		'format': 'bestaudio/best',
		'outtmpl': f'{audio_dir}.%(ext)s',
		'postprocessors': [{
			'key': 'FFmpegExtractAudio',
			'preferredcodec': 'mp3',
		}]
	}

	with yt_dlp.YoutubeDL(ydl_opts) as ydl:
		ydl.download([url])

# -----------------------------
# 음성파일을 텍스트로 변환
# INPUT: youtube id, 오디오 다운로드 경로
# -----------------------------
def audio2text(audio_dir):
	# 모델 로드 (base, small, medium, large 중 선택 가능)
	model = whisper.load_model("small")

	# 음성 파일 STT 수행
	result = model.transcribe(f'{audio_dir}.mp3')  # wav, mp4 등도 OK

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
# Gemini 요약
# ------------------------
def summarize_with_gemini(text: str) -> str:
	genai.configure(api_key=GEMINI_API_KEY)
	gemini_model = genai.GenerativeModel("gemini-2.0-flash")

	prompt = f"""
다음은 경제 관련 유튜브 영상 자막입니다.
말투는 제거하고, 핵심 내용 위주로 간결하게 뉴스 스타일로 요약해주세요:

{text}
"""
	response = gemini_model.generate_content(prompt)
	return response.text.strip()



# ------------------------
# GPT-4 등락 예측
# ------------------------
def predict_market_from_summary(summary: str, stock: str) -> str:
	GPT_llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=OPENAI_API_KEY)

	prompt = f"""
아래는 경제 뉴스의 요약입니다.

"{summary}"

이 뉴스의 내용이 주식 종목 "{stock}"에 긍정적인 영향을 미칠 가능성이 있을까요? 그렇다면 '오를 가능성 있음', 아니라면 '오를 가능성 낮음'이라고만 답해 주세요.
"""
	response = GPT_llm([HumanMessage(content=prompt)])
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

	video_id = search_videos(SEARCH_QUERY, BEFORE_DATE)

	# 유튜브 영상 음성 추출
	audio_dir = f'audio/{stock}_{BEFORE_DATE}'
	extract_video_audio(video_id, audio_dir)
	text = audio2text(audio_dir)

	# 자막 전처리
	cleaned = clean_srt(text)
	print("전처리:\n", cleaned)

	# 전처리 자막 요약
	summary = summarize_with_gemini(cleaned)

	# 주식 등락 예측
	stock = "삼성전자"
	prediction = predict_market_from_summary(summary, stock)

	print("\n📄 GEMINI 요약 결과:\n", summary)
	print("\n📈 GPT-4 예측 결과:\n", prediction)

	if prediction == '오를 가능성 있음':
		return 'up'
	elif prediction == '오를 가능성 낮음':
		return 'down'
	else:
		return 'middle'
