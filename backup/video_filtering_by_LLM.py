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

csv_path = f"../data_kr/video/뉴스 영상 수집본.csv"

def get_video_id_simple(url: str) -> str:
    # 마지막 `/` 뒤의 모든 문자를 ID로 간주
    return url.rstrip('/').split('/')[-1]

def decide_news(title: str, desc: str, company: str) -> int:
    client = genai.Client()
    user_prompt = "\n".join([
        # ------------------------------------------------------------
        # ① 역할(시스템)
        "## 역할",
        "너는 기업 관련 경제 뉴스 영상을 분류하는 전문 애널리스트이다.",
        "",
        # ------------------------------------------------------------
        # ② 분류 목표
        "## 분류 목표",
        "다음 **제목·설명**이 특정 기업(또는 그 계열사)의 주가·사업·경영과 직접적으로 관련된 경제 뉴스/정보 영상인지 판별한다.",
        "",
        # ------------------------------------------------------------
        # ③ 분류 규칙
        "## 분류 규칙",
        "1. 관련 있음(1):",
        "   - 기업의 실적·주가·사업 전략·지분 구조·계열사 소식·업계 평가 등을 다루는 **정형 뉴스/종목 해설 영상**",
        "2. 관련 없음(0):",
        "   - 스포츠·예능·팬 영상, 단순 행사·제품 발표·광고·콘퍼런스 중계 등 경제성과 무관한 경우",
        "3. 회사명이 **'그룹'으로 끝나는 경우**:",
        "   - 그 지주사 및 **모든 계열사**(예: ○○건설, ○○전자 등)를 다루면 ‘관련 있음’으로 간주",
        "4. 회사명을 축약·영문·티커로 표기한 경우도 동일하게 취급",
        "5. 반드시 **숫자 하나만 출력**한다 — 관련 있으면 `1`, 관련 없으면 `0` (추가 설명 절대 금지)",
        "",
        # ------------------------------------------------------------
        # ④ 입력
        "## 입력",
        f"회사명: {company}",
        f"제목: {title}",
        f"설명: {desc}",
        "",
        # ------------------------------------------------------------
        # ⑤ 출력 형식
        "## 출력 형식",
        "```\n<0또는1>\n```",
        "",
        # ------------------------------------------------------------
        # ⑥ 예시(참고용, 모델 출력에 포함 X)
        "## 예시",
        "- 회사명: 삼성전자 / 제목: ‘삼전 8만전자 눈앞…’ / 설명: 삼성전자 주가 분석 → **출력: 1**",
        "- 회사명: 삼성전자 / 제목: ‘갤럭시 언팩 풀영상’ / 설명: 제품 발표회 라이브 → **출력: 0**",
        "- 회사명: 두산그룹 / 제목: ‘두산로보틱스 상장 첫날 급등’ / 설명: 두산로보틱스 주가 전망 → **출력: 1**",
        "- 회사명: 두산그룹 / 제목: ‘2023 두산 베어스 삼진 모음’ / 설명: 야구 경기 하이라이트 → **출력: 0**",
    ])

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-pro-preview-05-06",
            contents=user_prompt
        )
        return int(resp.text)
    except Exception:
        return 0

if __name__ == '__main__':
    df = pd.read_csv(csv_path)
    df_new = pd.DataFrame(columns=['name', 'url', 'upload_dt'])
    for row in tqdm(df.itertuples(), total=len(df)):
        if pd.isna(row.url):
            continue
        vid = get_video_id_simple(row.url)
        try:
            resp = youtube.videos().list(id=vid, part="snippet", maxResults=1).execute()
            items = resp.get("items", [])
            item = items[0]

            company = row.name
            title = item["snippet"]["title"]
            desc = item["snippet"]["description"]
            if decide_news(company=company, title=title, desc=desc) == 0:
                df_new.loc[len(df_new)] = [row.name, row.url, row.upload_dt]
        except HttpError as e:
            if e.resp.status == 403 and 'quotaExceeded' in str(e):
                print("YouTube API quota exhausted. 스크립트를 종료합니다.")
                sys.exit(1)
            print(f"[ERROR] {e}")

        # 결과 중간저장
        df_new.to_csv(f"../data_kr/video/다소 상관없는 뉴스.csv", encoding='utf-8-sig', index=False)