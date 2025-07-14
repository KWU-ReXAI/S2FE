from predict_stock_outlook import *
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd

llm_dir = 'predict_mix_ta'  # preprocessed_data/llm/ 하위폴더명 (llm predict 저장될 곳)

df_v = pd.read_csv('data_kr/video/뉴스 영상 수집본.csv', encoding='utf-8')
df_v.rename(columns={'url': 'v_url', 'upload_dt': 'v_upload_dt'}, inplace=True)
df_v = df_v[["year", "quarter", "month", "week", "code", "name", "sector", "after", "before", "v_url", "v_upload_dt"]]

df_a = pd.read_csv('data_kr/video/뉴스 기사 수집본.csv', encoding='utf-8')
df_a.rename(columns={'url': 'a_url', 'upload_dt': 'a_upload_dt'}, inplace=True)
df_a = df_a[["a_url", "a_upload_dt"]]

df = pd.concat([df_v, df_a], axis=1)

"""# 1. 'code' 열의 모든 값을 문자열로 변환합니다.
df['code'] = df['code'].astype(str)

# 2. (권장) 모든 코드 값을 6자리 문자열로 통일합니다. (예: 6260 -> '006260')
df['code'] = df['code'].str.zfill(6)"""

# 3. 이제 양쪽 모두 문자열이므로 정상적으로 비교됩니다.
df = df[df['code'] >= 6400]

for code in df["code"].unique():
    df_ = df[df["code"] == code].reset_index(drop=True)

    for row in tqdm(df_.itertuples(), total=len(df_), desc=f"{code}LLM predicting"):
        if pd.isna(row.v_url) or pd.isna(row.a_url):
            continue

        code = str(row.code).zfill(6)
        name = row.name
        v_summary_dir = f'data_kr/video/script/{row.sector}/{code}/'
        t_summary_dir = f'data_kr/video/text/{row.sector}/{code}/'
        predict_dir = f'preprocessed_data/llm/{llm_dir}/{row.sector}/{code}/'
        os.makedirs(predict_dir, exist_ok=True)

        try:
            filename = f'{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt'
            stock = f'{name}({code})'

            df_news = pd.read_csv('./data_kr/video/뉴스 영상 수집본.csv', encoding='utf-8')

            # 주어진 모든 조건에 맞는 행을 필터링합니다.
            filtered_row = df_news[
                (df_news['code'] == row.code) &
                (df_news['year'] == row.year) &
                (df_news['quarter'] == row.quarter) &
                (df_news['month'] == row.month) &
                (df_news['week'] == row.week)
                ]

            # 필터링된 결과에서 'upload_dt' 값을 가져옵니다.
            upload_date = None  # 기본값을 None으로 설정
            if not filtered_row.empty:
                # 조건에 맞는 행이 하나 이상 있을 경우, 첫 번째 행의 'upload_dt' 값을 가져옵니다.
                upload_date = filtered_row.iloc[0]['upload_dt']

            csv_file_path = f'./ta_jonyeok/ta_preprocessed_data/{code}.csv'

            try:
                # 기술적 지표가 포함된 CSV 파일을 읽어옵니다.
                df_ta = pd.read_csv(csv_file_path)

                # '날짜' 열을 datetime 객체로 변환합니다.
                df_ta['날짜'] = pd.to_datetime(df_ta['날짜'])

                # 필터링할 날짜 범위를 계산합니다.
                upload_date = datetime.strptime(upload_date, '%Y-%m-%d')
                start_date = upload_date - timedelta(days=8)
                end_date = upload_date - timedelta(days=1)

                # 정의된 기간 내에 있는 행들을 필터링합니다.
                filtered_df = df_ta[
                    (df_ta['날짜'] >= start_date) & (df_ta['날짜'] <= end_date)
                    ].copy()  # .copy()를 사용하여 SettingWithCopyWarning 방지

                # --- 2. TXT 파일로 저장 ---
                technical_data = ""  # 최종 문자열을 담을 변수 초기화

                if not filtered_df.empty:
                    # 데이터프레임의 헤더(열 이름)를 쉼표로 구분된 문자열로 만듭니다.
                    header = ','.join(filtered_df.columns)

                    # 데이터 부분을 CSV 형태의 문자열로 변환합니다. (헤더와 인덱스 제외)
                    data_string = filtered_df.to_csv(header=False, index=False).strip()

                    # 헤더와 데이터 문자열을 합쳐 최종 문자열을 생성합니다.
                    technical_data = header + '\n' + data_string

                else:
                    print(f"⚠️ 경고: {start_date.date()} ~ {end_date.date()} 기간에 해당하는 데이터가 없습니다.")

            except FileNotFoundError:
                print(f"❌ 오류: '{csv_file_path}' 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
            with open(f'{v_summary_dir}{filename}', "r", encoding="utf-8") as file:
                script = file.read()
            with open(f'{t_summary_dir}{filename}', "r", encoding="utf-8") as file:
                article = file.read()

            data = predict_market_from_mix_with_ta(news_article=article, video_script=script, ta=technical_data,
                                                   stock=f'{name}({code})')

            with open(f'{predict_dir}{filename}', "w", encoding="utf-8") as file:
                file.write(data)

            with open(f'preprocessed_data/llm/{llm_dir}/log.txt', "a", encoding="utf-8") as log_file:
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                log_file.write(f"{timestamp} predict completed: {predict_dir}{filename}\n")
        except Exception as e:
            with open(f'preprocessed_data/llm/{llm_dir}/log.txt', "a", encoding="utf-8") as log_file:
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                log_file.write(f"{timestamp} predict error: {predict_dir}{filename}\n")
                print(e)

df = pd.read_csv('data_kr/video/뉴스 영상 수집본.csv', encoding='utf-8')
llm_dir = 'predict_video_ta'  # preprocessed_data/llm/ 하위폴더명 (llm predict 저장될 곳)

for code in df["code"].unique():
    df_ = df[df["code"] == code].reset_index(drop=True)

    for row in tqdm(df_.itertuples(), total=len(df_), desc=f"{code}LLM predicting"):
        if pd.isna(row.url) or row.url == '':
            continue

        code = str(row.code).zfill(6)
        name = row.name
        script_dir = f'data_kr/video/script/{row.sector}/{code}/'
        predict_dir = f'preprocessed_data/llm/{llm_dir}/{row.sector}/{code}/'
        os.makedirs(predict_dir, exist_ok=True)

        try:
            filename = f'{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt'
            stock = f'{name}({code})'

            df_news = pd.read_csv('./data_kr/video/뉴스 영상 수집본.csv', encoding='utf-8')

            # 주어진 모든 조건에 맞는 행을 필터링합니다.
            filtered_row = df_news[
                (df_news['code'] == row.code) &
                (df_news['year'] == row.year) &
                (df_news['quarter'] == row.quarter) &
                (df_news['month'] == row.month) &
                (df_news['week'] == row.week)
                ]

            # 필터링된 결과에서 'upload_dt' 값을 가져옵니다.
            upload_date = None  # 기본값을 None으로 설정
            if not filtered_row.empty:
                # 조건에 맞는 행이 하나 이상 있을 경우, 첫 번째 행의 'upload_dt' 값을 가져옵니다.
                upload_date = filtered_row.iloc[0]['upload_dt']

            csv_file_path = f'./ta_jonyeok/ta_preprocessed_data/{code}.csv'

            try:
                # 기술적 지표가 포함된 CSV 파일을 읽어옵니다.
                df_ta = pd.read_csv(csv_file_path)

                # '날짜' 열을 datetime 객체로 변환합니다.
                df_ta['날짜'] = pd.to_datetime(df_ta['날짜'])

                # 필터링할 날짜 범위를 계산합니다.
                upload_date = datetime.strptime(upload_date, '%Y-%m-%d')
                start_date = upload_date - timedelta(days=8)
                end_date = upload_date - timedelta(days=1)

                # 정의된 기간 내에 있는 행들을 필터링합니다.
                filtered_df = df_ta[
                    (df_ta['날짜'] >= start_date) & (df_ta['날짜'] <= end_date)
                    ].copy()  # .copy()를 사용하여 SettingWithCopyWarning 방지

                # --- 2. TXT 파일로 저장 ---
                technical_data = ""  # 최종 문자열을 담을 변수 초기화

                if not filtered_df.empty:
                    # 데이터프레임의 헤더(열 이름)를 쉼표로 구분된 문자열로 만듭니다.
                    header = ','.join(filtered_df.columns)

                    # 데이터 부분을 CSV 형태의 문자열로 변환합니다. (헤더와 인덱스 제외)
                    data_string = filtered_df.to_csv(header=False, index=False).strip()

                    # 헤더와 데이터 문자열을 합쳐 최종 문자열을 생성합니다.
                    technical_data = header + '\n' + data_string

                else:
                    print(f"⚠️ 경고: {start_date.date()} ~ {end_date.date()} 기간에 해당하는 데이터가 없습니다.")

            except FileNotFoundError:
                print(f"❌ 오류: '{csv_file_path}' 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

            with open(f'{script_dir}{filename}', "r", encoding="utf-8") as file:
                script = file.read()
            prediction = predict_market_from_summary_with_ta(summary=script,ta=technical_data, stock=f'{name}({code})')

            with open(f'{predict_dir}{filename}', "w", encoding="utf-8") as file:
                file.write(prediction)

            with open(f'preprocessed_data/llm/{llm_dir}/log.txt', "a", encoding="utf-8") as log_file:
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                log_file.write(f"{timestamp} predict completed: {predict_dir}{filename}\n")
        except Exception as e:
            with open(f'preprocessed_data/llm/{llm_dir}/log.txt', "a", encoding="utf-8") as log_file:
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                log_file.write(f"{timestamp} predict error: {predict_dir}{filename}\n")
                print(e)

df = pd.read_csv('data_kr/video/뉴스 기사 수집본.csv', encoding='utf-8')
llm_dir = 'predict_text_ta'  # preprocessed_data/llm/ 하위폴더명 (llm predict 저장될 곳)

for code in df["code"].unique():
    df_ = df[df["code"] == code].reset_index(drop=True)

    for row in tqdm(df_.itertuples(), total=len(df_), desc=f"{code}LLM predicting"):
        if pd.isna(row.url) or row.url == '':
            continue

        code = str(row.code).zfill(6)
        name = row.name
        article_dir = f'data_kr/video/text/{row.sector}/{code}/'
        predict_dir = f'preprocessed_data/llm/{llm_dir}/{row.sector}/{code}/'
        os.makedirs(predict_dir, exist_ok=True)

        try:
            filename = f'{row.year}-{row.quarter}-{str(row.month).zfill(2)}-{row.week}.txt'
            stock = f'{name}({code})'
            df_news = pd.read_csv('./data_kr/video/뉴스 기사 수집본.csv', encoding='utf-8')

            # 주어진 모든 조건에 맞는 행을 필터링합니다.
            filtered_row = df_news[
                (df_news['code'] == row.code) &
                (df_news['year'] == row.year) &
                (df_news['quarter'] == row.quarter) &
                (df_news['month'] == row.month) &
                (df_news['week'] == row.week)
                ]

            # 필터링된 결과에서 'upload_dt' 값을 가져옵니다.
            upload_date = None  # 기본값을 None으로 설정
            if not filtered_row.empty:
                # 조건에 맞는 행이 하나 이상 있을 경우, 첫 번째 행의 'upload_dt' 값을 가져옵니다.
                upload_date = filtered_row.iloc[0]['upload_dt']

            csv_file_path = f'./ta_jonyeok/ta_preprocessed_data/{code}.csv'

            try:
                # 기술적 지표가 포함된 CSV 파일을 읽어옵니다.
                df_ta = pd.read_csv(csv_file_path)

                # '날짜' 열을 datetime 객체로 변환합니다.
                df_ta['날짜'] = pd.to_datetime(df_ta['날짜'])

                # 필터링할 날짜 범위를 계산합니다.
                upload_date = datetime.strptime(upload_date, '%Y-%m-%d')
                start_date = upload_date - timedelta(days=8)
                end_date = upload_date - timedelta(days=1)

                # 정의된 기간 내에 있는 행들을 필터링합니다.
                filtered_df = df_ta[
                    (df_ta['날짜'] >= start_date) & (df_ta['날짜'] <= end_date)
                    ].copy()  # .copy()를 사용하여 SettingWithCopyWarning 방지

                # --- 2. TXT 파일로 저장 ---
                technical_data = ""  # 최종 문자열을 담을 변수 초기화

                if not filtered_df.empty:
                    # 데이터프레임의 헤더(열 이름)를 쉼표로 구분된 문자열로 만듭니다.
                    header = ','.join(filtered_df.columns)

                    # 데이터 부분을 CSV 형태의 문자열로 변환합니다. (헤더와 인덱스 제외)
                    data_string = filtered_df.to_csv(header=False, index=False).strip()

                    # 헤더와 데이터 문자열을 합쳐 최종 문자열을 생성합니다.
                    technical_data = header + '\n' + data_string

                else:
                    print(f"⚠️ 경고: {start_date.date()} ~ {end_date.date()} 기간에 해당하는 데이터가 없습니다.")

            except FileNotFoundError:
                print(f"❌ 오류: '{csv_file_path}' 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

            with open(f'{article_dir}{filename}', "r", encoding="utf-8") as file:
                article = file.read()
            data = predict_market_from_summary_with_ta(summary=article,ta=technical_data, stock=f'{name}({code})')

            with open(f'{predict_dir}{filename}', "w", encoding="utf-8") as file:
                file.write(data)

            with open(f'preprocessed_data/llm/{llm_dir}/log.txt', "a", encoding="utf-8") as log_file:
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                log_file.write(f"{timestamp} predict completed: {predict_dir}{filename}\n")
        except Exception as e:
            with open(f'preprocessed_data/llm/{llm_dir}/log.txt', "a", encoding="utf-8") as log_file:
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                log_file.write(f"{timestamp} predict error: {predict_dir}{filename}\n")
                print(e)

