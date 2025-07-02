import pandas as pd
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from google import genai
import concurrent.futures
import dotenv
import os

from tqdm import tqdm
dotenv.load_dotenv()

def choose_most_relevant_index(titles: list[str], company_name: str) -> int:
    """
    titles: 0~9번까지의 기사 제목 리스트 (최대 10개)
    company_name: 관련도를 판단할 회사명

    반환값:
      - 0~9: titles 중 가장 회사와 관련 있다고 판단된 제목의 인덱스
      - 10: 모두 관련 없다고 판단된 경우
    """
    client = genai.Client()

    user_lines = [f"{i}: {t}" for i, t in enumerate(titles)]
    user_prompt = "\n".join([
        "## 당신은 기업과 가장 관련된 뉴스 기사를 제목만 보고도 찾아낼 수 있는 뉴스 분석 전문가입니다."
        f"## 회사명: {company_name}",
        "## 다음은 '기업명 뉴스'로 검색한 결과 페이지의 제목 목록입니다:",
        *user_lines,
        "",
        "이 중에서 뉴스 기사라고 판별되고, 기사 제목에 기업명이 포함되어 있으면서, 가장 기업의 주가와 관련된 기사라고 생각되는 기사 제목의 번호(0~9)를 하나만 **숫자로만** 출력해주세요.",
        "뉴스 기사는 youtube, facebook, instagram 등 SNS에 게시된 것을 제외한 뉴스 사이트에 게재된 웹 페이지만 취급합니다.",
        "만약 어떤 제목도 관련이 없다고 판단되면 **10**을 출력해주세요.",
        "### example - 기업명: 삼성전자",
        "0: '삼성은 세개의 별? 푸하하'",
        "1: '삼성역 3번 출구'",
        "2: '삼성전자 역대급 실적에도 주가는 하락세'",
        "Response: 2"
    ])

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                client.models.generate_content,
                model="gemini-2.5-pro-preview-05-06",
                contents=user_prompt
            )
            resp = future.result(timeout=60)  # 60초 내 응답 없으면 TimeoutError
        return int(resp.text)
    except concurrent.futures.TimeoutError:
        return 10
    except Exception as e:
        # 그 밖의 에러 발생 시에도 안전하게 10 반환
        return 10

def _extract_publish_date(item: dict):
    pm = item.get("pagemap", {})

    for art in pm.get("newsarticle", []):
        for k in ("datepublished", "datecreated"):
            if art.get(k):
                dt = pd.to_datetime(art[k], errors="coerce")
                if not pd.isna(dt):
                    return dt.strftime("%Y-%m-%d")

    for meta in pm.get("metatags", []):
        for k in ("article:published_time", "og:published_time", "date"):
            if meta.get(k):
                dt = pd.to_datetime(meta[k], errors="coerce")
                if not pd.isna(dt):
                    return dt.strftime("%Y-%m-%d")

    # 못 찾으면 결측
    return pd.NA

def fetch_google_news(keyword: str,
                      start_date: str,
                      end_date: str):
    service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_API_KEY"))

    try:
        key = f"{keyword}그룹" if keyword == "LS" or keyword == "두산" or keyword == "LG" or keyword == "SK" else keyword
        res = service.cse().list(q=f"{key} 뉴스 after:{start_date} before:{end_date}", cx=os.getenv("CX"), num=10, hl="ko", lr="lang_ko").execute()
    except HttpError as e:
        # ── 404 (Requested entity was not found) → CX 잘못 등: 바로 반환
        if e.resp.status == 404:
            return pd.NA, pd.NA, pd.NA
        # ── 다른 HTTP 오류는 호출자에게 넘겨서 로그 등 확인
        raise

    if len(res.get("items", [])) == 0:
        return pd.NA, pd.NA, pd.NA
    items = res.get("items", [])
    titles = [item['title'] for item in items]
    num = choose_most_relevant_index(titles, key)

    if num == 10:
        return pd.NA, pd.NA, pd.NA
    item = items[num]
    upload_dt = _extract_publish_date(item)
    return item["link"], 'article', upload_dt

if __name__ == "__main__":
    codes = [6260, 10120, 11200, 25540, 47050, 51600]
    for code in codes:
        df = pd.read_csv(f"김태완_{code}_검토끝.csv")
        df['after'] = df['filtering'].str.extract(r'after:([0-9\-]+)')
        df['before'] = df['filtering'].str.extract(r'before:([0-9\-]+)')

        tqdm.pandas()
        mask = (df['url'].isna())
        # mask에 해당하는 행만 progress_apply로 처리
        results = df.loc[mask].progress_apply(
            lambda row: pd.Series(
                fetch_google_news(
                    keyword=row['name'],
                    start_date=row['after'],
                    end_date=row['before']
                )
            ),
            axis=1
        )
        # 결과 대입
        df.loc[mask, ['url', 'category', 'upload_dt']] = results.values
        df.drop(['after', 'before'], axis=1, inplace=True)
        df.to_csv(f"김태완_{code}_진짜_검토끝.csv", index=False, encoding='utf-8-sig')