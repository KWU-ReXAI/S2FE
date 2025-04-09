from dotenv import load_dotenv
import os

from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

load_dotenv()  # .env 파일에서 환경변수 불러오기

def extract_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url:
        parsed = urlparse(url)
        return parse_qs(parsed.query).get("v", [None])[0]
    return None

def extract_video_text(url):
	video_id = extract_video_id(url)

	transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])

	# 모든 문장을 띄어쓰기로 이어붙이기
	full_text = " ".join([entry['text'] for entry in transcript])
	
	return full_text

# 유튜브 영상에서 자막 추출
url = "https://www.youtube.com/watch?v=XLAUgCEIaS0"
text = extract_video_text(url)
print("YOUTUBE title:\n", text)

# 랭체인으로 LLM에 질문하고 대답 가져오기기
# 1. OpenAI LLM 설정 (OPENAI_API_KEY가 환경 변수로 설정되어 있어야 함)
# 창의성은 70%, 대답 수 max는 10 tokens
llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

# 2. 프롬프트 템플릿 정의
# 요약 단계 프롬프트
summary_prompt = PromptTemplate(
    input_variables=["article"],
    template="""
다음 경제 뉴스 기사를 간결하고 명확하게 요약해 주세요:

"{article}"

요약:
"""
)

# 상승 여부 예측 프롬프트
prediction_prompt = PromptTemplate(
    input_variables=["summary", "stock"],
    template="""
아래는 경제 뉴스의 요약입니다.

"{summary}"

이 뉴스의 내용이 주식 종목 "{stock}"에 긍정적인 영향을 미칠 가능성이 있을까요? 그렇다면 '오를 가능성 있음', 아니라면 '오를 가능성 낮음'이라고만 답해 주세요.
"""
)

# 3. LLMChain 생성
# 각 체인 정의
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")
prediction_chain = LLMChain(llm=llm, prompt=prediction_prompt, output_key="prediction")

# 전체 파이프라인 연결
full_chain = SequentialChain(
    chains=[summary_chain, prediction_chain],
    input_variables=["article", "stock"],
    output_variables=["summary", "prediction"],
    verbose=True
)

stock_name = "엔비디아"

result = full_chain({
    "article": text,
    "stock": stock_name
})

print("요약:", result['summary'])
print("예측:", result['prediction'])