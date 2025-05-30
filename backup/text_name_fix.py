import os
import pandas as pd

df_text = pd.read_csv(f"../data_kr/video/동영상 수집 통합본 최신.csv")
df_date = pd.read_csv(f"../data_kr/video/자료 수집 통합본_최신.csv")
df_date['after'] = df_date['filtering'].str.extract(r'after:([0-9\-]+)')
df_date['before'] = df_date['filtering'].str.extract(r'before:([0-9\-]+)')

root_dir = '../data_kr/video/text'  # ← 실제 텍스트 폴더 경로로 수정하세요

for dirpath, dirnames, filenames in os.walk(root_dir):
    # dirpath 예시: text/산업재/000120
    rel = os.path.relpath(dirpath, root_dir)
    parts = rel.split(os.sep)
    if len(parts) != 2:
        continue  # 최하위(code) 폴더만 처리

    sector, code = parts
    for fname in filenames:
        if not fname.endswith('.txt'):
            continue

        name, ext = os.path.splitext(fname)  # name 예시: '2016-Q1'
        try:
            year, quarter = name.split('-')   # ['2016', 'Q1']
        except ValueError:
            print(f"파일명 형식 오류, 건너뜀: {os.path.join(dirpath, fname)}")
            continue

        # CSV에서 일치하는 행 찾기
        mask = (
            (df_text['year'] == int(year)) &
            (df_text['quarter'] == quarter) &
            (df_text['code'] == int(code)) &
            (df_text['sector'] == sector)
        )
        matched = df_text.loc[mask, 'upload_date']
        if matched.empty:
            print(f"매칭되는 CSV 없음: {sector}/{code}/{fname}")
            continue

        date = matched.iloc[0]  # 예: '2016-05-15'
        need_to_rename = df_date.loc[((df_date['after'] <= date) & (df_date['before'] >= date)), ['year', 'quarter','month', 'week']]
        if need_to_rename.empty:
            continue
        year, quarter, month, week = need_to_rename.iloc[0]
        new_fname = f"{year}-{quarter}-{str(month).zfill(2)}-{week}.txt"
        src = os.path.join(dirpath, fname)
        dst = os.path.join(dirpath, new_fname)

        # 이미 같은 이름의 파일이 있으면 건너뜀
        if os.path.exists(dst):
            print(f"변경할 파일명 이미 존재, 건너뜀: {dst}")
            continue

        os.rename(src, dst)
        print(f"Renamed: {src} → {dst}")