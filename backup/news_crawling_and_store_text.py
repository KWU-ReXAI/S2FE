import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from dateutil import parser
from tqdm import tqdm

# ————————————————————————————————————————
# 설정
# — base_url: CSV 파일이 있는 경로(및 텍스트 저장 경로의 기준)가 됩니다.
base_url = '../data_kr/video'

# 에러 로그 파일 경로 (base_url 아래에 생성)
LOG_FILE = os.path.join(base_url, 'error_log.txt')

# ————————————————————————————————————————
def extract_news(url, timeout=10):
    """
    주어진 URL에서 기사 본문(text)과 게시일(date)을 추출합니다.
    - text: 페이지 내 모든 텍스트(HTML 태그 제외) – '\n'으로 구분된 문자열
    - date: "YYYY-MM-DD" 형식의 게시일 문자열 (없으면 None)
    403 Forbidden이 돌아오면 error_log.txt에 기록하고, (None, None) 반환.
    """
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        # 네트워크 문제 등으로 요청 자체가 실패한 경우
        # 로그에 남긴 뒤 (None, None) 반환
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{url}\tREQUEST_FAILED\t{str(e)}\n")
        return None, None

    # 1) HTTP 상태 코드 확인 (특히 403 처리)
    if resp.status_code == 403:
        # 403 Forbidden인 경우, error_log.txt에 URL과 상태 코드 기록
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{url}\tHTTP_403_FORBIDDEN\n")
        return None, None

    # 403 이외의 오류는 raise_for_status로 예외 처리
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # 404, 500 등 다른 HTTP 오류가 발생하면 로그에 저장
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{url}\tHTTP_ERROR_{resp.status_code}\t{str(e)}\n")
        return None, None

    # —————————————————————————————————————————————
    # 2) BeautifulSoup으로 파싱
    soup = BeautifulSoup(resp.text, "html.parser")

    # 2.1) 본문(content) 추출: 태그 없이 페이지 내 모든 텍스트를 가져옴
    #      (줄바꿈을 활용해 어느 정도 문단 단위 유지)
    content = soup.get_text(separator="\n", strip=True)

    # 2.2) 게시일(date) 추출 시도
    date_str = None

    # (1) <time datetime="…">
    time_tag = soup.find("time", datetime=True)
    if time_tag:
        date_str = time_tag.get("datetime", None)

    # (2) <meta> 태그에서 표준 속성
    if not date_str:
        meta_keys = [
            ("property", "article:published_time"),
            ("property", "og:published_time"),
            ("name", "pubdate"),
            ("name", "publishdate"),
            ("itemprop", "datePublished"),
        ]
        for attr, value in meta_keys:
            meta = soup.find("meta", attrs={attr: value})
            if meta and meta.get("content"):
                date_str = meta["content"]
                break

    # 3) 파싱 후 “YYYY-MM-DD” 형태로 포맷
    pub_date = None
    if date_str:
        try:
            dt = parser.parse(date_str, fuzzy=True)
            pub_date = dt.strftime("%Y-%m-%d")
        except Exception:
            pub_date = None

    return content, pub_date


# ————————————————————————————————————————
if __name__ == "__main__":
    # 1) CSV 읽어오기 (최신 통합본)
    df = pd.read_csv(os.path.join(base_url, "자료 수집 통합본_최신.csv"))

    # (샘플 테스트를 위해 원하시면 .sample(n)으로 축소 가능)
    # df = df.sample(2)

    # 2) 에러 로그 파일이 이미 있으면 삭제하거나 초기화
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # 3) 각 행 순회하면서 텍스트/날짜 추출 및 파일로 저장
    for row in tqdm(df.itertuples(), total=len(df)):
        # URL이 비어 있거나(NA) 이미 생성된 파일이 있으면 건너뜀
        if pd.isna(row.url):
            continue

        # 저장할 디렉토리 및 파일명 설정
        fpath = os.path.join(base_url, "text", str(row.sector), str(row.code).zfill(6))
        os.makedirs(fpath, exist_ok=True)
        fname = os.path.join(fpath, f"{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt")

        if os.path.exists(fname):
            # 이미 파일이 있으면 스킵
            print(f"{fname}: 파일 이미 존재함\n")
            continue

        # 실제 추출 시도
        content, pub_date = extract_news(row.url)
        if content is None and pub_date is None:
            # (extract_news 내부에서 403 등 오류를 이미 로그로 남겼으므로 여기서는 넘어감)
            continue
        if pd.isna(row.upload_dt) and (pub_date is not None):
            df.at[row.Index, 'upload_dt'] = pub_date
        with open(fname, "w", encoding='utf-8') as file:
            file.write(content)
    df.to_csv(f"{base_url}/자료 수집 최종본.csv", index=False, encoding='utf-8-sig')