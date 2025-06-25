import pandas as pd

path = '../data_kr/video/'
df = pd.read_csv(f'{path}뉴스 영상 수집본.csv', encoding='utf-8-sig')

data = pd.DataFrame(columns=['code', 'name', 'sector', 'total_rows', 'missing_rows', 'missing_ratio'])

code_list = df['code'].unique().tolist()
for code in code_list:
	df_code = df[df['code'] == code]
	df_na = df_code[df_code['url'].isna()]
	data.loc[len(data)] = [str(int(code)).zfill(6), df_code.iloc[0]['name'], df_code.iloc[0]['sector'], len(df_code), len(df_na), len(df_na)/len(df_code)]
df_na = df[df['url'].isna()]
print('total: ', len(df_na), len(df_na)/len(df))

# df.loc[df['url'] == 'X', 'url'] = None
# df.loc[df['upload_dt'] == 'X', 'upload_dt'] = None
data.to_csv(f"{path}Nans.csv", encoding='utf-8-sig', index=False)