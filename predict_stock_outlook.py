from dotenv import load_dotenv
import os
from pathlib import Path
from datetime import datetime, timedelta

from tqdm import tqdm

from google import genai
from google.genai import types
from langchain.schema import SystemMessage, HumanMessage


import pandas as pd

load_dotenv()  # .env 파일에서 환경변수 불러오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# gemini 모델 로드
#genai.configure(api_key=GEMINI_API_KEY)
client=genai.Client(api_key=GEMINI_API_KEY)

# GPT 모델 로드
#gpt_model = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=OPENAI_API_KEY)

# ------------------------
# GPT-4o 등락 예측
# ------------------------
def predict_market_from_summary(summary: str, stock: str) -> str:
	system_prompt = """
당신은 주어진 기업 소식을 종합적으로 분석하여, 특정 주식 종목의 단기 등락 가능성을 판단하는 정보 분석 전문가입니다. 제시되는 분석 단계를 따라 논리적으로 추론한 후, 최종 판단을 단 하나의 정수로만 내려야 합니다.
"""

	user_prompt = f"""
한국 상장 기업 "{stock}"과 관련된 소식이 제공됩니다.

[분석 작업]
아래 **[분석 단계]**에 따라 머릿속으로 단계별로 생각한 후, "{stock}"의 단기 주가 등락에 대한 최종 판단을 **[출력 지시사항]**에 맞춰 출력하세요.

[분석 단계 (Chain of Thought)]
1단계: 핵심 정보 식별
- 제공된 소식의 가장 중요한 사실(Fact)은 무엇인가?
- 이 소식의 주체와 대상은 누구인가? (예: 정부 정책, 기업 발표, 시장 루머 등)

2단계: 정보의 성격 및 강도 분석
- 이 정보는 기업에 긍정적인가(호재), 부정적인가(악재), 혹은 중립적인가?
- 정보의 영향력은 어느 정도인가? (예: 1회성 해프닝, 지속적인 성장 동력, 구조적 리스크 등)

3단계: 주가 영향력 평가
- 이 정보가 단기 주가에 즉각적으로 영향을 미칠 가능성이 있는가?
- 시장에서 이미 예상하고 있던 내용인가(선반영)? 혹은 예상치 못한 새로운 정보(서프라이즈)인가?
- 시장의 전반적인 투자 심리(투심)와 "{stock}"이 속한 산업의 현재 상황을 고려할 때, 이 정보의 파급력은 어떠할 것인가?

4단계: 종합 결론 도출
- 위 1, 2, 3단계를 종합했을 때, "{stock}"의 주가는 단기적으로 상승, 하락, 보합(변동 미미) 중 어느 방향으로 움직일 가능성이 가장 높은가?

[출력 지시사항]
1. ❗️오직 '+1', '0', '-1' 중 하나의 정수만 출력해야 합니다.
2. 어떠한 경우에도 위 [분석 단계]에 대한 설명, 자신의 생각 과정, 근거, 부가적인 텍스트, 줄바꿈 등 다른 어떤 문자도 포함해서는 안 됩니다.
3. 최종 판단 결과인 정수 값 외에 다른 모든 출력은 금지됩니다.
- 주가 상승 예상: +1
- 주가 변동 미미 또는 예측 불가 예상: 0
- 주가 하락 예상: -1

[출력 예시]
+1

[기업 소식]
{summary}

"""

	response = client.models.generate_content(
		model="gemini-2.5-flash",
		config=types.GenerateContentConfig(
			system_instruction=system_prompt),
		contents=user_prompt
	)

	return response.text

# ------------------------
# GPT-4o 등락 예측
# ------------------------
def predict_market_from_mix(news_article: str, video_script:str, stock: str) -> str:
	system_prompt = """
당신은 주어진 뉴스 기사와 경제 영상 스크립트를 종합적으로 분석하여, 특정 주식 종목의 단기 등락 가능성을 판단하는 정보 분석 전문가입니다. 제시되는 분석 단계를 따라 논리적으로 추론한 후, 최종 판단을 단 하나의 정수로만 내려야 합니다.
"""

	user_prompt = f"""
한국 상장 기업 "{stock}"과 관련된 **뉴스 기사와 경제 영상 스크립트**가 제공됩니다.

[분석 작업]
아래 **[분석 단계]**에 따라 머릿속으로 단계별로 생각한 후, "{stock}"의 단기 주가 등락에 대한 최종 판단을 **[출력 지시사항]**에 맞춰 출력하세요.

[분석 단계 (Chain of Thought)]
1단계: 핵심 정보 식별
- 제공된 소식의 가장 중요한 사실(Fact)은 무엇인가?
- 이 소식의 주체와 대상은 누구인가? (예: 정부 정책, 기업 발표, 시장 루머 등)

2단계: 정보의 성격 및 강도 분석
- 이 정보는 기업에 긍정적인가(호재), 부정적인가(악재), 혹은 중립적인가?
- 정보의 영향력은 어느 정도인가? (예: 1회성 해프닝, 지속적인 성장 동력, 구조적 리스크 등)

3단계: 주가 영향력 평가
- 이 정보가 단기 주가에 즉각적으로 영향을 미칠 가능성이 있는가?
- 시장에서 이미 예상하고 있던 내용인가(선반영)? 혹은 예상치 못한 새로운 정보(서프라이즈)인가?
- 시장의 전반적인 투자 심리(투심)와 "{stock}"이 속한 산업의 현재 상황을 고려할 때, 이 정보의 파급력은 어떠할 것인가?

4단계: 종합 결론 도출
- 위 1, 2, 3단계를 종합했을 때, "{stock}"의 주가는 단기적으로 상승, 하락, 보합(변동 미미) 중 어느 방향으로 움직일 가능성이 가장 높은가?

[출력 지시사항]
1. ❗️오직 '+1', '0', '-1' 중 하나의 정수만 출력해야 합니다.
2. 어떠한 경우에도 위 [분석 단계]에 대한 설명, 자신의 생각 과정, 근거, 부가적인 텍스트, 줄바꿈 등 다른 어떤 문자도 포함해서는 안 됩니다.
3. 최종 판단 결과인 정수 값 외에 다른 모든 출력은 금지됩니다.
- 주가 상승 예상: +1
- 주가 변동 미미 또는 예측 불가 예상: 0
- 주가 하락 예상: -1

[출력 예시]
+1

[뉴스 기사]
{news_article}
---
[경제 영상]
{video_script}
---

"""

	response = client.models.generate_content(
		model="gemini-2.5-flash",
		config=types.GenerateContentConfig(
			system_instruction=system_prompt),
		contents=user_prompt
	)

	return response.text

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
