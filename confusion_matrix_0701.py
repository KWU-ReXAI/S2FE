import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import platform
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_disclosure_date(strdate, folder_path="./data_kr/date_regression/"):
	"""
	strdate 예: "2020_Q4"
	isStart: True면 가장 빠른 날짜, False면 가장 늦은 날짜
	"""
	try:
		file_path = os.path.join(folder_path, f"{strdate}.csv")
		df = pd.read_csv(file_path)

		if "disclosure_date" not in df.columns:
			return f"{strdate}.csv 파일에 'disclosure_date' 열이 없습니다."

		# 날짜 컬럼을 datetime 형식으로 변환
		df["disclosure_date"] = pd.to_datetime(df["disclosure_date"], errors='coerce')
		df = df.dropna(subset=["disclosure_date"])

		if df.empty:
			return f"{strdate}.csv 파일에 유효한 disclosure_date가 없습니다."

		return df["disclosure_date"].max().strftime("%Y-%m-%d")

	except Exception as e:
		return f"오류 발생: {e}"


def get_rows_by_date_range(code: str, start_date_str: str, end_date_str: str, data: int = 0) -> pd.DataFrame:
	# 실제 존재하는 파일 경로를 찾기
	file_path = None
	if data == 0:
		file_path = "./preprocessed_data/llm/predict_total/predict_video.csv"
	elif data == 1:
		file_path = "./preprocessed_data/llm/predict_total/predict_text.csv"
	elif data == 2:
		file_path = "./preprocessed_data/llm/predict_total/predict_total.csv"
	else:
		raise ValueError('data 파라미터가 범위를 벗어남.')

	# CSV 읽기
	df = pd.read_csv(file_path, encoding='utf-8')
	df = df[df['code'] == code]

	# upload_dt를 datetime으로 변환
	df['upload_dt'] = pd.to_datetime(df['upload_dt'])

	# 문자열을 datetime으로 변환
	start_dt = pd.to_datetime(start_date_str) - relativedelta(months=1)
	end_dt = pd.to_datetime(end_date_str) - relativedelta(months=1)

	# 업로드날짜가 기간 내에 존재할 경우
	mask = df['score'].notna() & (start_dt <= df['upload_dt']) & (df['upload_dt'] < end_dt)

	# 겹치는 행 반환
	return df.loc[mask].reset_index(drop=True)

def LLM_accuracy(code, start_date, end_date, data):

	df_score = pd.DataFrame(columns=[
		"date", "code", "score", "return"
	])

	# 2) 해당 분기의 가장 늦는 공시일 얻기
	start_datetime = get_disclosure_date(start_date)
	end_datetime = get_disclosure_date(end_date)

	df = get_rows_by_date_range(code, start_datetime, end_datetime, data)

	start_dt = datetime.strptime(start_datetime, '%Y-%m-%d') + timedelta(days=1)  # 점검 필요
	end_dt = datetime.strptime(end_datetime, '%Y-%m-%d') - timedelta(days=1)
	current = start_dt  ## 거래일

	while current <= end_dt:
		trade_date = current.strftime("%Y-%m-%d")
		df['upload_dt'] = pd.to_datetime(df['upload_dt'])

		### 자료 수집 기간: upload_start ~ upload_end ###
		upload_start = current - relativedelta(months=1)
		upload_end = current - timedelta(days=1)
		chunk = df.loc[(upload_start <= df['upload_dt']) & (df['upload_dt'] <= upload_end)].copy()

		chunk.reset_index(drop=True, inplace=True)

		chunk["score"] *= (chunk.index + 1) #가중치 넣음! 우선 임의로
		score_sum = chunk["score"].sum()
		threshold = 2
		score = 1 if score_sum >= threshold else -1
		df_score.loc[len(df_score)] = [
			trade_date, code, score, None
		]

		###### 구매 코드 구현하기 ######
		buy_dt = current
		sell_dt = current + relativedelta(months=1)
		sell_dt = sell_dt if sell_dt <= end_dt else end_dt

		df_price = pd.read_csv(f"data_kr/price/{str(int(code)).zfill(6)}.csv")
		df_price['날짜'] = pd.to_datetime(df_price['날짜'])
		buy_price = df_price[df_price['날짜'] >= buy_dt].iloc[0]['종가']
		sell_price = df_price[df_price['날짜'] <= sell_dt].iloc[-1]['종가']
		df_score.loc[len(df_score) - 1, "return"] = sell_price / buy_price - 1

		current += relativedelta(months=1)
	return df_score

if __name__ == '__main__':
	for idx, data in enumerate(tqdm(['video', 'text', 'total'], desc="데이터 별 진행상황")):
		fpath = f"./preprocessed_data/llm/confusion_matrix_0701/{data}"
		os.makedirs(fpath, exist_ok=True)

		df = pd.DataFrame(columns=[
			"date", "code", "score", "return"
		])
		df_code = pd.read_csv(f"./preprocessed_data/llm/predict_total/predict_{data}.csv", encoding='utf-8-sig')
		code_list = df_code['code'].unique().tolist()
		for code in tqdm(code_list, desc="코드별 진행상황"):
			dates = ['2020_Q4', '2021_Q1', '2021_Q2', '2021_Q3', '2021_Q4',
					 '2022_Q1', '2022_Q2', '2022_Q3', '2022_Q4',
					 '2023_Q1', '2023_Q2', '2023_Q3', '2023_Q4',
					 '2024_Q1', '2024_Q2', '2024_Q3', '2024_Q4']
			for index, _ in enumerate(dates):
				if index == len(dates) - 1:
					break
				df = pd.concat([df, LLM_accuracy(code, dates[index], dates[index+1], idx)])
		df.to_csv(f"{fpath}/score_and_return.csv", encoding='utf-8-sig', index=False)
		df = df[~(df['score'] == 0.0)]
		# 1. 부호를 기준으로 예측(y_pred)과 정답(y_true) 생성
		y_pred = np.sign(df['score'].tolist()).astype(int)
		y_true = np.sign(df['return'].tolist()).astype(int)

		# 실제 가격이 변화가 없을 때(y_true가 0)는 안 사는 게 이득이므로(수수료), -1로 수정
		y_true = [-1 if x == 0 else x for x in y_true]

		# 2. Confusion Matrix 생성 및 시각화
		# 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic)
		if platform.system() == 'Windows':
			plt.rc('font', family='Malgun Gothic')
		elif platform.system() == 'Darwin':
			plt.rc('font', family='AppleGothic')
		else:
			plt.rc('font', family='NanumGothic')
		plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

		# 라벨 순서를 [1 (양수), -1 (음수)]로 지정
		cm = confusion_matrix(y_true, y_pred, labels=[1, -1])

		# 시각화
		plt.figure(figsize=(8, 6))
		labels = ['주가 상승', '주가 하락']
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
		plt.title('LLM 예측 결과 Confusion Matrix', fontsize=16)
		plt.xlabel('Predicted', fontsize=12)
		plt.ylabel('Actual', fontsize=12)
		plt.savefig(f"{fpath}/confusion_matrix_{data}.png")

		# 3. 결과 분석 및 DataFrame 생성
		# PP, PN, NP, NN 값 추출 (PP: P예측-P정답, PN: N예측-P정답 ...)
		# cm[0,0] = TP (실제 Positive, 예측 Positive) -> PP
		# cm[0,1] = FN (실제 Positive, 예측 Negative) -> PN
		# cm[1,0] = FP (실제 Negative, 예측 Positive) -> NP
		# cm[1,1] = TN (실제 Negative, 예측 Negative) -> NN
		pp_count = cm[0, 0]
		pn_count = cm[0, 1]
		np_count = cm[1, 0]
		nn_count = cm[1, 1]
		total = pp_count + pn_count + np_count + nn_count

		# 성능 지표 계산
		# zero_division=0: 분모가 0일 경우 0으로 처리하여 경고 방지
		accuracy = accuracy_score(y_true, y_pred)
		precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
		recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
		f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

		# DataFrame 생성
		results_data = {
			'label': ['PP (TP)', 'PN (FN)', 'NP (FP)', 'NN (TN)', 'Total', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
			'count': [pp_count, pn_count, np_count, nn_count, total, np.nan, np.nan, np.nan, np.nan],
			'rate(%)': [
				(pp_count / total) * 100,
				(pn_count / total) * 100,
				(np_count / total) * 100,
				(nn_count / total) * 100,
				np.nan,
				accuracy * 100,
				precision * 100,
				recall * 100,
				f1 * 100
			]
		}

		df_results = pd.DataFrame(results_data)
		df_results['rate(%)'] = df_results['rate(%)'].round(2)  # 소수점 2자리까지 표시

		# 4. CSV 파일로 출력
		# encoding='utf-8-sig'로 지정해야 Excel에서 한글이 깨지지 않습니다.
		df_results.to_csv(f'{fpath}/prediction_results_{data}.csv', index=False, encoding='utf-8-sig')

		print("Confusion Matrix 결과 요약:")
		print(df_results)
		print(f"\n결과가 'prediction_results_{data}.csv' 파일로 저장되었습니다.")
