import pandas as pd

df = pd.read_csv('../collected_news.csv')

df_text_filtered = df[(df['category'] == 'article') & df['text'].isna()].copy(deep=True)
df_date_filtered = df[(df['category'] == 'article') & df['upload_dt'].isna()].copy(deep=True)

df_date_filtered.drop('text', axis=1, inplace=True)

df_text_filtered.to_csv('no text.csv', encoding='utf-8-sig', index=False)
df_date_filtered.to_csv('no upload_dt.csv', encoding='utf-8-sig', index=False)
