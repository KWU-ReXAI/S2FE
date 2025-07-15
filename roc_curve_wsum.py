import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import platform
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression
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
		file_path = "./preprocessed_data/llm/predict_total_CoT/predict_video_CoT.csv"
	elif data == 1:
		file_path = "./preprocessed_data/llm/predict_total_CoT/predict_text_CoT.csv"
	elif data == 2:
		file_path = "./preprocessed_data/llm/predict_total_CoT/predict_mix_CoT.csv"
	elif data == 3:
		file_path = "./preprocessed_data/llm/predict_total_CoT/predict_total_CoT.csv"
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

		total_period_days = (upload_end - upload_start).days
		# 'Series' 전체에 .days를 적용할 수 없으므로 .dt 접근자 사용
		chunk["score"] *= 1 + ((chunk['upload_dt'] - upload_start).dt.days / total_period_days)
		model = LinearRegression()
		score_sum = chunk["score"].sum()
		threshold = 0.3
		score = score_sum
		df_score.loc[len(df_score)] = [
			trade_date, code, score, None
		]

		###### 구매 코드 구현하기 ######
		buy_dt = current
		sell_dt = current + relativedelta(weeks=1)
		sell_dt = sell_dt if sell_dt <= end_dt else end_dt

		df_price = pd.read_csv(f"data_kr/price/{str(int(code)).zfill(6)}.csv")
		df_price['날짜'] = pd.to_datetime(df_price['날짜'])
		buy_price = df_price[df_price['날짜'] >= buy_dt].iloc[0]['종가']
		sell_price = df_price[df_price['날짜'] <= sell_dt].iloc[-1]['종가']
		df_score.loc[len(df_score) - 1, "return"] = sell_price / buy_price - 1

		current += relativedelta(months=1)
	return df_score


if __name__ == '__main__':
	# 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic)
	if platform.system() == 'Windows':
		plt.rc('font', family='Malgun Gothic')
	elif platform.system() == 'Darwin':
		plt.rc('font', family='AppleGothic')
	else:
		plt.rc('font', family='NanumGothic')
	plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지
	fpath = f"./preprocessed_data/llm/confusion_matrix_0701"
	for idx, data in enumerate(tqdm(['video', 'text', 'mix', 'total'], desc="데이터 별 진행상황")):

		df = pd.DataFrame(columns=[
			"date", "code", "score", "return"
		])
		for testNum in tqdm(range(1, 11), desc='모델 별 진행상황'):
			for phase in ['p1', 'p2', 'p3', 'p4']:
				df_phase = pd.read_csv(f"./result/test_result_dir/test_selected_stocks_model_{phase}_{testNum}.csv")
				df_dedup = df_phase.drop_duplicates(subset=['0'], keep='first')
				selected = df_dedup.drop('1', axis=1)
				result_list = selected.values.tolist()

				for codes in result_list:
					date = None
					dates = ['2020_Q4', '2021_Q1', '2021_Q2', '2021_Q3', '2021_Q4',
							 '2022_Q1', '2022_Q2', '2022_Q3', '2022_Q4',
							 '2023_Q1', '2023_Q2', '2023_Q3', '2023_Q4',
							 '2024_Q1', '2024_Q2', '2024_Q3', '2024_Q4']
					for code in codes:
						if code in dates:
							date = dates.index(code)
							continue
						df = pd.concat([df, LLM_accuracy(code, dates[date], dates[date+1], idx)])
		df = df[~(df['score'] == 0.0)]
		# 1. 부호를 기준으로 예측(y_pred)과 정답(y_true) 생성
		y_pred = df['score'].tolist()
		y_true = [1 if x > 0.005 else 0 for x in df['return'].tolist()]

		# ROC 곡선 계산
		fpr, tpr, thresholds = roc_curve(y_true, y_pred)

		# AUC 계산
		roc_auc = auc(fpr, tpr)

		# ROC 곡선 그리기
		plt.figure()
		plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
		plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.0])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver Operating Characteristic')
		plt.legend(loc='lower right')

		num_thresholds_to_show = 20
		indices_to_show = np.linspace(0, len(thresholds) - 1, num_thresholds_to_show).astype(int)

		for i in indices_to_show:
			threshold_value = thresholds[i]
			fpr_point = fpr[i]
			tpr_point = tpr[i]

			# 임계값을 그래프 위에 텍스트로 표시
			# text() 함수의 x, y 좌표는 데이터 좌표계입니다.
			# ha='center', va='bottom'은 텍스트 정렬을 지정합니다.
			# round() 함수를 사용하여 소수점 자릿수를 조정합니다.
			plt.text(fpr_point, tpr_point,
					 f'{threshold_value:.2f}',
					 color='red', fontsize=9, ha='right', va='bottom')
			# 선택된 임계값 지점에 작은 마커를 추가하여 시각적으로 강조
			plt.plot(fpr_point, tpr_point, 'o', color='red', markersize=5)

		plt.savefig(f'{fpath}/roc_curve_{data}.png')
		plt.close()