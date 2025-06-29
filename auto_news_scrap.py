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
    return None

def fetch_google_news(keyword: str,
                      start_date: str,
                      end_date: str):
    service = build("customsearch", "v1", developerKey=os.getenv("API_KEY"))

    try:
        res = service.cse().list(q=f"{keyword}", cx=os.getenv("CX"), num=1, hl="ko", lr="lang_ko",
                                 sort=f"date:r:{start_date.replace('-', '')}:{end_date.replace('-', '')}").execute()
    except HttpError as e:
        # ── 404 (Requested entity was not found) → CX 잘못 등: 바로 반환
        if e.resp.status == 404:
            return None, None
        # ── 다른 HTTP 오류는 호출자에게 넘겨서 로그 등 확인
        raise

    if len(res.get("items", [])) == 0:
        return None, None
    item = res.get("items", [])[0]
    upload_dt = _extract_publish_date(item)
    if upload_dt is None:
        print(f"{keyword} {start_date} ~ {end_date} 업로드일 결측!")
    return item["link"], upload_dt

if __name__ == "__main__":
    # 1. 파일 경로 설정
    # 원본 파일과 작업 내용을 계속 덮어쓸 출력 파일 경로를 지정합니다.
    input_file = 'data_kr/video/자료 수집 통합본.csv'
    output_file = 'data_kr/video/뉴스 기사 수집본.csv'

    # 2. 데이터 로딩 (작업 이어가기 기능)
    # 출력 파일이 이미 존재하면 해당 파일에서 작업을 이어서 시작합니다.
    # 없다면 원본 파일로 새로 시작합니다.
    if os.path.exists(output_file):
        print(f"'{output_file}' 파일에서 작업을 이어갑니다.")
        df = pd.read_csv(output_file, encoding='utf-8-sig')
    else:
        print(f"'{input_file}' 파일에서 새로운 작업을 시작합니다.")
        df = pd.read_csv(input_file, encoding='utf-8-sig')

    # 3. 날짜 열 추출 (최초 한 번만 실행)
    if 'after' not in df.columns or 'before' not in df.columns:
        df['after'] = df['filtering'].str.extract(r'after:([0-9\-]+)')
        df['before'] = df['filtering'].str.extract(r'before:([0-9\-]+)')

    # 4. 처리할 행의 인덱스 식별
    # 'url' 열이 비어있는 행들만 대상으로 지정합니다.
    mask = df['url'].notna()
    indices_to_process = df[~mask].index

    if indices_to_process.empty:
        print("모든 행에 URL이 채워져 있습니다. 작업을 종료합니다.")
    else:
        print(f"총 {len(indices_to_process)}개의 항목에 대한 스크래핑을 시작합니다.")

        # 5. 한 행씩 반복 처리 및 즉시 저장
        for index in tqdm(indices_to_process, desc="스크래핑 진행 중"):
            # 현재 행의 정보를 가져옴
            row = df.loc[index]

            try:
                company = row['name']
                if company in ['LS', '두산', 'LG', 'SK']:
                    company = company + '그룹'
                # 스크래핑 함수 호출
                url, upload_dt = fetch_google_news(
                    keyword=company,
                    start_date=row['after'],
                    end_date=row['before'],
                )

                # 성공적으로 데이터와 업로드일을 가져왔을 경우에만 값 업데이트
                if url and upload_dt:
                    df.loc[index, 'url'] = url
                    df.loc[index, 'upload_dt'] = upload_dt

                    # [핵심] 데이터프레임 전체를 CSV 파일에 즉시  저장
                    df.to_csv(output_file, index=False, encoding='utf-8-sig')

            except Exception as e:
                # 특정 행에서 오류가 발생해도 전체 스크립트가 멈추지 않도록 처리
                print(f"\n오류 발생 (행 인덱스: {index}, 키워드: {row['name']}): {e}")
                continue  # 다음 행으로 넘어감

    # 6. 최종 정리
    # 모든 작업 완료 후 filtering 열을 삭제하고 최종 저장
    if os.path.exists(output_file):
        final_df = pd.read_csv(output_file, encoding='utf-8-sig')
        if 'after' in final_df.columns or 'before' in final_df.columns:
            final_df.drop('filtering', axis=1, inplace=True, errors='ignore')
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("\n모든 작업이 완료되었습니다.")

