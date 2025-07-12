from predict_stock_outlook import *
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd

df = pd.read_csv('data_kr/video/뉴스 영상 수집본.csv', encoding='utf-8')
llm_dir = 'predict_video_self_consistency'  # preprocessed_data/llm/ 하위폴더명 (llm predict 저장될 곳)

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
            with open(f'{script_dir}{filename}', "r", encoding="utf-8") as file:
                script = file.read()
            prediction = predict_market_from_summary_self_consistency(script, f'{name}({code})')

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
llm_dir = 'predict_text_self_consistency'  # preprocessed_data/llm/ 하위폴더명 (llm predict 저장될 곳)

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
            with open(f'{article_dir}{filename}', "r", encoding="utf-8") as file:
                article = file.read()
            data = predict_market_from_summary_self_consistency(article, f'{name}({code})')

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

llm_dir = 'predict_mix_self_consistency'  # preprocessed_data/llm/ 하위폴더명 (llm predict 저장될 곳)

df_v = pd.read_csv('data_kr/video/뉴스 영상 수집본.csv', encoding='utf-8')
df_v.rename(columns={'url': 'v_url', 'upload_dt': 'v_upload_dt'}, inplace=True)
df_v = df_v[["year", "quarter", "month", "week", "code", "name", "sector", "after", "before", "v_url", "v_upload_dt"]]

df_a = pd.read_csv('data_kr/video/뉴스 기사 수집본.csv', encoding='utf-8')
df_a.rename(columns={'url': 'a_url', 'upload_dt': 'a_upload_dt'}, inplace=True)
df_a = df_a[["a_url", "a_upload_dt"]]

df = pd.concat([df_v, df_a], axis=1)

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
            with open(f'{v_summary_dir}{filename}', "r", encoding="utf-8") as file:
                script = file.read()
            with open(f'{t_summary_dir}{filename}', "r", encoding="utf-8") as file:
                article = file.read()

            data = predict_market_from_mix_self_consistency(article, script, f'{name}({code})')

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