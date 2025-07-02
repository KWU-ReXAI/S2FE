import pandas as pd
import webbrowser

base_url = "../data_kr/video"
file_path = f"{base_url}/error_log.txt"
df = pd.read_csv(f"{base_url}/자료 수집 최종본.csv")
# 파일에서 URL 읽기
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# URL만 추출해서 리스트로 저장
urls = [line.split()[0] for line in lines if line.strip()]

# 하나씩 열고 사용자 입력 대기
for url in urls:
    webbrowser.open(url)
    row = (df.loc[df['url'] == url, ['sector', 'code', 'year', 'quarter', 'month', 'week']]).iloc[0]
    fname = f"{base_url}/text/{row.sector}/{str(row.code).zfill(6)}/{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt"
    content = input(f"\n페이지를 열었습니다. 내용을 '{fname}'에 복사한 후 엔터를 누르세요: ")