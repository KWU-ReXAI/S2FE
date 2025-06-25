import pandas as pd
import os
import sys
import logging
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    IpBlocked,
)

# --- 로깅 설정 --------------------------------------------------
path = '../data_kr/video'
log_file = os.path.join(path, 'error.log')
os.makedirs(path, exist_ok=True)

logging.basicConfig(
    filename=log_file,
    filemode='a',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# ----------------------------------------------------------------

def get_video_id_simple(url: str) -> str:
    return url.rstrip('/').split('/')[-1]

df = pd.read_csv(f"{path}/뉴스 영상 수집본.csv", encoding='utf-8-sig')
ytt_api = YouTubeTranscriptApi()
text_formatter = TextFormatter()

df = df[df['url'].notna()]

# 에러가 난 행 정보를 모아둘 리스트
error_rows = []

for row in tqdm(df.itertuples(), total=len(df)):
    fpath = os.path.join(path, "script", str(row.sector), str(row.code).zfill(6))
    os.makedirs(fpath, exist_ok=True)
    fname = os.path.join(
        fpath,
        f"{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt"
    )

    if os.path.exists(fname):
        continue
    video_id = get_video_id_simple(row.url)
    try:
        transcript = ytt_api.fetch(video_id, languages=['ko', 'en'])
        text_formatted = text_formatter.format_transcript(transcript)
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(text_formatted)

    except (RequestBlocked, IpBlocked) as e:
        err_msg = f"요청 차단됨: {e}"
        logger.error(f"{video_id} - {err_msg}")
        sys.exit(f"프로그램을 종료합니다: {err_msg}")

    except (TranscriptsDisabled, NoTranscriptFound) as e:
        err_msg = f"자막 없음: {e}"
        logger.error(f"{video_id} - {err_msg}")
        error_rows.append({
            'year': row.year,
            'quarter': row.quarter,
            'month': row.month,
            'week': row.week,
            'sector': row.sector,
            'code': row.code,
            'url': row.url,
            'error': err_msg
        })
        df_errors = pd.DataFrame(error_rows)
        error_csv = os.path.join(path, 'error_rows.csv')
        df_errors.to_csv(error_csv, index=False, encoding='utf-8-sig')
        continue

    except VideoUnavailable as e:
        err_msg = f"동영상 없음: {e}"
        logger.error(f"{video_id} - {err_msg}")
        error_rows.append({
            'year': row.year,
            'quarter': row.quarter,
            'month': row.month,
            'week': row.week,
            'sector': row.sector,
            'code': row.code,
            'url': row.url,
            'error': err_msg
        })
        df_errors = pd.DataFrame(error_rows)
        error_csv = os.path.join(path, 'error_rows.csv')
        df_errors.to_csv(error_csv, index=False, encoding='utf-8-sig')
        continue

    except Exception as e:
        err_msg = f"알 수 없는 에러: {e}"
        logger.error(f"{video_id} - {err_msg}", exc_info=True)
        error_rows.append({
            'year': row.year,
            'quarter': row.quarter,
            'month': row.month,
            'week': row.week,
            'sector': row.sector,
            'code': row.code,
            'url': row.url,
            'error': err_msg
        })
        df_errors = pd.DataFrame(error_rows)
        error_csv = os.path.join(path, 'error_rows.csv')
        df_errors.to_csv(error_csv, index=False, encoding='utf-8-sig')
        continue
