import os
import sys
from dotenv import load_dotenv
import pandas as pd
from google import genai
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

load_dotenv()

# YouTube API 클라이언트 초기화
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

# 한 번에 몇 주치를 검색할 지
SEARCH = 1

# CSV 파일 경로 (이미 반드시 존재)
csv_path = f"./data_kr/video/error_rows.csv"
## 뉴스 영상 제목, 설명으로 연관성있는 뉴스인지 판별하는 함수
### 검색 과정에서 쓰기에는 호출 횟수가 너무 많아질 수 있어(한 번의 검색 당 최대 50 + a번) 일단 사용 보류
### 나중에 다 검색 후 내용 판별할 때 사용 예정
# def decide_news(title: str, desc: str, company: str) -> int:
#     client = genai.Client()
#     user_prompt = "\n".join([
#         # ------------------------------------------------------------
#         # ① 역할(시스템)
#         "## 역할",
#         "너는 기업 관련 경제 뉴스 영상을 분류하는 전문 애널리스트이다.",
#         "",
#         # ------------------------------------------------------------
#         # ② 분류 목표
#         "## 분류 목표",
#         "다음 **제목·설명**이 특정 기업(또는 그 계열사)의 주가·사업·경영과 직접적으로 관련된 경제 뉴스/정보 영상인지 판별한다.",
#         "",
#         # ------------------------------------------------------------
#         # ③ 분류 규칙
#         "## 분류 규칙",
#         "1. 관련 있음(1):",
#         "   - 기업의 실적·주가·사업 전략·지분 구조·계열사 소식·업계 평가 등을 다루는 **정형 뉴스/종목 해설 영상**",
#         "2. 관련 없음(0):",
#         "   - 스포츠·예능·팬 영상, 단순 행사·제품 발표·광고·콘퍼런스 중계 등 경제성과 무관한 경우",
#         "3. 회사명이 **'그룹'으로 끝나는 경우**:",
#         "   - 그 지주사 및 **모든 계열사**(예: ○○건설, ○○전자 등)를 다루면 ‘관련 있음’으로 간주",
#         "4. 회사명을 축약·영문·티커로 표기한 경우도 동일하게 취급",
#         "5. 반드시 **숫자 하나만 출력**한다 — 관련 있으면 `1`, 관련 없으면 `0` (추가 설명 절대 금지)",
#         "",
#         # ------------------------------------------------------------
#         # ④ 입력
#         "## 입력",
#         f"회사명: {company}",
#         f"제목: {title}",
#         f"설명: {desc}",
#         "",
#         # ------------------------------------------------------------
#         # ⑤ 출력 형식
#         "## 출력 형식",
#         "```\n<0또는1>\n```",
#         "",
#         # ------------------------------------------------------------
#         # ⑥ 예시(참고용, 모델 출력에 포함 X)
#         "## 예시",
#         "- 회사명: 삼성전자 / 제목: ‘삼전 8만전자 눈앞…’ / 설명: 삼성전자 주가 분석 → **출력: 1**",
#         "- 회사명: 삼성전자 / 제목: ‘갤럭시 언팩 풀영상’ / 설명: 제품 발표회 라이브 → **출력: 0**",
#         "- 회사명: 두산그룹 / 제목: ‘두산로보틱스 상장 첫날 급등’ / 설명: 두산로보틱스 주가 전망 → **출력: 1**",
#         "- 회사명: 두산그룹 / 제목: ‘2023 두산 베어스 삼진 모음’ / 설명: 야구 경기 하이라이트 → **출력: 0**",
#     ])
#
#     try:
#         resp = client.models.generate_content(
#             model="gemini-2.5-pro-preview-05-06",
#             contents=user_prompt
#         )
#         return int(resp.text)
#     except Exception:
#         return 0


if __name__ == "__main__":
    # 기존 CSV 로드
    df = pd.read_csv(csv_path)

    # URL이 비어 있는(first missing) 행부터 시작
    mask_missing = df["url"].isna() | (df["url"] == "")
    if not mask_missing.any():
        print("모든 기간이 처리되었습니다.")
        exit()

    start_idx = mask_missing.idxmax()

    # 나머지 행 순회
    for idx in tqdm(range(start_idx, len(df), SEARCH)):

        rows = df.iloc[idx : idx + SEARCH] if idx + SEARCH <= len(df) \
            else df.iloc[idx : len(df)] # SEARCH 주차 만큼의 뉴스 URL 저장 DB
        company    = rows.iloc[0]["name"]         # 기업명

        if company in ['LS', '두산', 'LG', 'SK']:
            company = company + '그룹'

        publishedAfter      = rows.iloc[0]["after"]        # "YYYY-MM-DD"
        publishedBefore     = rows.iloc[-1]["before"]      # "YYYY-MM-DD"
        # ISO 포맷 시간 문자열
        publishedAfter  = publishedAfter  + "T00:00:00Z"
        publishedBefore = publishedBefore + "T23:59:59Z"

        try:
            resp = youtube.search().list(
                q=f"{company}",
                part="snippet",
                type="video",
                order="viewCount",
                topicId="/m/098wr", # 사회 카테고리(뉴스 카테고리 별도 X)
                videoCaption='closedCaption',
                maxResults=50,
                publishedAfter=publishedAfter,
                publishedBefore=publishedBefore,
                relevanceLanguage="ko",
                regionCode="KR",
                # 영상 길이 필터: short(<4분), medium(4~20분), long(>20분)
                videoDuration="medium"
            ).execute()

            for id, row in rows.iterrows():
                after = df.iloc[id]['after']
                before = df.iloc[id]['before']
                url, upload_dt = "X", "X"

                for item in resp.get("items", []):
                    publishedAt = item["snippet"]["publishedAt"][:10]
                    if (after <= publishedAt) and (publishedAt <= before):
                        title       = item["snippet"]["title"]
                        description = item["snippet"]["description"]
                        # if decide_news(company=company, title=title, desc=description):
                        vid_id   = item["id"]["videoId"]
                        url      = f"https://youtu.be/{vid_id}"
                        upload_dt = publishedAt
                        break
                if pd.isna(df.iloc[id]['url']):
                    df.at[id, 'url'] = url
                if pd.isna(df.iloc[id]['upload_dt']):
                    df.at[id, 'upload_dt'] = upload_dt

        except HttpError as e:
            if e.resp.status == 403 and 'quotaExceeded' in str(e):
                print("YouTube API quota exhausted. 스크립트를 종료합니다.")
                sys.exit(1)
            print(f"[ERROR] row {idx} — {company} {after}~{before}: {e}")

        # 결과 중간저장
        df.to_csv(f"./data_kr/video/error_rows.csv", encoding='utf-8-sig', index=False)

