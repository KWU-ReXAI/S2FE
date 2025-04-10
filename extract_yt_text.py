from dotenv import load_dotenv
import os

from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import re

import google.generativeai as genai
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

load_dotenv()  # .env 파일에서 환경변수 불러오기

# -----------------------------
# 유튜브링크에서 id 추출
# -----------------------------
def extract_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url:
        parsed = urlparse(url)
        return parse_qs(parsed.query).get("v", [None])[0]
    return None

# -----------------------------
# 유튜브링크에서 자막 추출
# -----------------------------
def extract_video_text(url):
	video_id = extract_video_id(url)

	transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])

	# 모든 문장을 띄어쓰기로 이어붙이기
	full_text = " ".join([entry['text'] for entry in transcript])
	
	return full_text

# -----------------------------
# 자막 전처리 함수
# -----------------------------
def clean_srt(srt_text: str) -> str:
    srt_text = re.sub(r"(좀|그냥|뭐랄까|그러니까|아니|약간|뭔가|뭐냐면요)", "", srt_text)
    return srt_text


# ------------------------
# Gemini 요약
# ------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

def summarize_with_gemini(text: str) -> str:
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
GPT_llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

def predict_market_from_summary(summary: str, stock: str) -> str:
    prompt = f"""
아래는 경제 뉴스의 요약입니다.

"{summary}"

이 뉴스의 내용이 주식 종목 "{stock}"에 긍정적인 영향을 미칠 가능성이 있을까요? 그렇다면 '오를 가능성 있음', 아니라면 '오를 가능성 낮음'이라고만 답해 주세요.
"""
    response = GPT_llm([HumanMessage(content=prompt)])
    return response.content.strip()

# 유튜브 영상에서 자막 추출
url = "https://www.youtube.com/watch?v=DjhmSamDlhw"
text = extract_video_text(url)

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