import pandas as pd
from tqdm import tqdm
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
import dotenv
import os

dotenv.load_dotenv()

# ──────────────────────────────────────────────
# 1) 날짜 추출 헬퍼: 사이트별 날짜 정보 추출 코드입니다
#
# ──────────────────────────────────────────────
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
        res = service.cse().list(q=f"{key} 뉴스 after:{start_date} before:{end_date}", cx=os.getenv("CX"), num=1, hl="ko", lr="lang_ko").execute()
    except HttpError as e:
        # ── 404 (Requested entity was not found) → CX 잘못 등: 바로 반환
        if e.resp.status == 404:
            return pd.NA, pd.NA, pd.NA
        # ── 다른 HTTP 오류는 호출자에게 넘겨서 로그 등 확인
        raise

    if len(res.get("items", [])) == 0:
        return pd.NA, pd.NA, pd.NA
    item = res.get("items", [])[0]
    upload_dt = _extract_publish_date(item)
    return item["link"], 'article', upload_dt

if __name__ == "__main__":
    df = pd.read_csv('data_kr/video/자료 수집 통합본.csv')
    df['after'] = df['filtering'].str.extract(r'after:([0-9\-]+)')
    df['before'] = df['filtering'].str.extract(r'before:([0-9\-]+)')

    mask = ((df['url'].notna()) | (df['code'] == 8060)
                                | (df['code'] == 9150)
                                | (df['code'] == 11070))
    tqdm.pandas()
    # mask에 해당하는 행만 progress_apply로 처리
    results = df.loc[~mask].progress_apply(
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
    df.loc[~mask, ['url', 'category', 'upload_dt']] = results.values
    df.drop(['after', 'before'], axis=1, inplace=True)
    df.to_csv('자료 수집 통합본_자동 스크랩.csv', index=False, encoding='utf-8-sig')
