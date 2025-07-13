import pandas as pd
from tqdm import tqdm  # 진행 상황을 시각적으로 보여주기 위한 라이브러리
from dotenv import load_dotenv
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from google import genai
from google.genai import types
load_dotenv()  # .env 파일에서 환경변수 불러오기
# --- 1. 설정 ---
# 본인의 Gemini API 키를 입력하세요.
# 키를 직접 노출하는 대신 환경 변수로 설정하는 것이 더 안전합니다.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client=genai.Client(api_key=GEMINI_API_KEY)
# 사용할 파일 경로 (이전에 기술적 지표를 추가한 파일)
# 예시: './ta_preprocessed_data/000120.csv'
# 여기서는 사용자가 업로드한 원본 파일을 사용한다고 가정하고 진행합니다.
# 원활한 예측을 위해 기술적 지표가 추가된 파일을 사용하시는 것을 강력히 권장합니다.
FILE_PATH = './ta_preprocessed_data/000120.csv'

# 예측에 사용할 과거 데이터 기간 (일)
HISTORY_WINDOW = 8
# 매수 후 보유 기간 (일)
HOLDING_PERIOD = 7


# --- 2. LLM 호출 함수 정의 ---
def get_llm_prediction(history_df, target_date_str):
    """과거 데이터를 기반으로 LLM에게 매수 결정을 물어보는 함수"""
    # 과거 데이터를 텍스트(CSV 형식)로 변환
    history_data_str = history_df.to_csv(index=False)

    # 프롬프트 생성
    prompt = f"""
    당신은 과거 주가 데이터를 분석하여 미래 수익 가능성을 예측하는 전문 금융 분석가입니다.

    아래는 한 주식 종목의 과거 {HISTORY_WINDOW}일간의 데이터입니다. 데이터에는 날짜, 시가, 고가, 저가, 종가, 거래량 등이 포함되어 있습니다.

    <과거 데이터>
    {history_data_str}

    이 데이터를 면밀히 분석하여, {target_date_str}에 이 주식을 매수하고 {HOLDING_PERIOD}일 후에 매도하는 전략이 수익을 낼 가능성이 높다고 판단되면 '1'을, 그렇지 않다고 판단되면 '0'을 반환해주세요.

    오직 '1' 또는 '0' 숫자 하나만 응답해야 하며, 다른 어떤 설명이나 문장도 추가해서는 안 됩니다.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                ),
            contents=prompt
        )

        # LLM의 답변에서 숫자만 추출 (공백 등 제거)
        prediction = int(response.text.strip())
        # 답변이 0 또는 1이 아닐 경우를 대비
        return prediction if prediction in [0, 1] else 0
    except Exception as e:
        # API 오류나 파싱 오류 발생 시 0 (매수 안함)으로 처리
        print(f"Error on {target_date_str}: {e}")
        return 0


# --- 3. 메인 로직 실행 ---
def run_prediction_on_timeseries(file_path):
    """시계열 데이터를 순회하며 LLM 예측을 수행하고 결과를 저장하는 메인 함수"""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"오류: {file_path} 파일을 찾을 수 없습니다.")
        return

    # 날짜 열을 datetime 형식으로 변환 (오류 발생 시 무시)
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df = df.dropna(subset=['날짜']).sort_values('날짜').reset_index(drop=True)

    predictions = []
    # 예측을 수행할 데이터 범위 설정 (과거 기록이 충분한 시점부터 시작)
    # tqdm을 사용하여 반복문의 진행 상황을 표시
    start_index = HISTORY_WINDOW
    for i in tqdm(range(start_index, len(df)), desc="Predicting..."):
        # 과거 데이터 윈도우 슬라이싱
        history_df = df.iloc[i - HISTORY_WINDOW: i]

        # 예측 대상 날짜
        target_date = df.iloc[i]['날짜']
        target_date_str = target_date.strftime('%Y-%m-%d')

        # LLM 예측 실행
        prediction = get_llm_prediction(history_df, target_date_str)
        predictions.append(prediction)

    # 결과 데이터프레임 생성
    # 예측은 start_index부터 시작했으므로, 그에 맞춰 데이터프레임을 자름
    result_df = df.iloc[start_index:].copy()
    result_df['LLM_Signal'] = predictions

    # 결과 저장
    output_filename = f"llm_predicted_{os.path.basename(file_path)}"
    result_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print("\n✅ 예측이 완료되었습니다!")
    print(f"결과가 {output_filename} 파일로 저장되었습니다.")
    print("\n--- 결과 데이터 샘플 (마지막 5개 행) ---")
    print(result_df.tail())


# 스크립트 실행
run_prediction_on_timeseries(FILE_PATH)