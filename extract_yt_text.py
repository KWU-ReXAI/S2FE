from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from transformers import pipeline

from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

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

def summary_text(text):
	summarizer = pipeline("summarization", model="t5-base")
	summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    
	return summary
    

# 유튜브 영상에서 자막 추출
url = "https://www.youtube.com/watch?v=PuxGPqk9IQA"
text = extract_video_text(url)
summary = summary_text(text)

# 랭체인으로 LLM에 질문하고 대답 가져오기기
# 1. OpenAI LLM 설정 (OPENAI_API_KEY가 환경 변수로 설정되어 있어야 함)
# 창의성은 70%, 대답 수 max는 10 tokens
llm = OpenAI(
				temperature=0.7,
				max_tokens=30,
				openai_api_key="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"
            )

# 2. 프롬프트 템플릿 정의
prompt = PromptTemplate(
    input_variables=["question"],
    template="너는 친절한 경제 전문가야. 다음 질문에 대해 설명해줘:\n\n질문: {question}\n\n답변:"
)

# 3. LLMChain 생성
chain = LLMChain(llm=llm, prompt=prompt)

corp_list = ["삼성", "LG"]
for corp in corp_list:
	# 4. 질문 입력 후 실행
	question = f"\"{summary}\"를 바탕으로 {corp}의 주가가 상승할지 하락할지만 예측해줘줘"
	response = chain.run(question)

	print(response)