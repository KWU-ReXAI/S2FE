import webbrowser
import pandas as pd

chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe %s"
origin = pd.read_csv("김태완_25540_진짜_검토끝.csv")

# code = 6260
# code = 10120
# code = 11200
code = 25540
# code = 47050
# code = 51600

mask = (origin['code'] == code)

df = origin[mask].copy(deep=True)
df_pub = origin[mask].copy(deep=True)
for row in df.itertuples():
	if pd.isna(row.url) or row.category == 'video':
		continue
	webbrowser.get(chrome_path).open(row.url)  # 기본 브라우저에서 열기
	text = input("기사를 유지할까요?(y/n) ")
	if text == 'n':
		df_pub.loc[(df_pub['year'] == row.year)
					& (df_pub['quarter'] == row.quarter)
					& (df_pub['month'] == row.month)
					& (df_pub['week'] == row.week), ['url', 'category', 'upload_dt']] = pd.NA, pd.NA, pd.NA
	ticker = row.code

df_pub.to_csv(f"김태완_{code}_진짜진짜_검토끝.csv", index=False, encoding='utf-8-sig')