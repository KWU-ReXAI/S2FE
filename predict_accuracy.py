from datetime import timedelta, datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def get_pred_accuracy():
    plt.figure(figsize=(24, 5))
    
    datas = ['video', 'text', 'mix', 'total']
    
    for idx, data in enumerate(datas):
        df = pd.read_csv(f'preprocessed_data/llm/predict_total/predict_{data}.csv', encoding='utf-8')
        df_d = pd.read_csv(f'data_kr/video/뉴스 영상 수집본.csv', encoding='utf-8')[['after', 'before']]
        df = pd.concat([df, df_d], axis=1)
        df[['upload_dt', 'after', 'before']] = df[['upload_dt', 'after', 'before']].apply(pd.to_datetime)
        
        ### LLM 예측에 실제 등락 라벨 추가 ###
        df["code"] = df["code"].astype(str).str.zfill(6)
        
        price_upload = [] # 업로드 당일 종가
        price_end = [] # 공시 당일 종가
        for row in tqdm(df.itertuples(), total=len(df)):
            df_price = pd.read_csv(f"data_kr/price/{row.code}.csv")
            df_price['날짜'] = pd.to_datetime(df_price['날짜'])
            ### 업로드 날짜 직전 종가
            price_upload.append(df_price.loc[df_price["날짜"] < row.upload_dt, "종가"].iloc[-1] if not pd.isna(row.upload_dt) else None)
            ### 데이터 업로드 주간 마지막 날 종가
            price_end.append(df_price.loc[df_price["날짜"] < row.before, "종가"].iloc[-1] if not pd.isna(row.before) else None)
        
        df["price_upload"] = price_upload
        df["price_end"] = price_end
    
        def check_change(row):
            rate = (row["price_end"] / row["price_upload"] - 1) * 100
            if rate > 0:
                return "상승"
            else:
                return "하락"
        df["label"] = df.apply(check_change, axis=1)

        ###############################################################
        ### 유효한 데이터만 남기기 ###
        valid_bool = ~(pd.isna(df['prediction']) | (df['prediction'] == '중립'))
        prediction = df[valid_bool]['prediction']
        label = df[valid_bool]['label']
        
        mapping = {'매우 긍정': '상승', '긍정': '상승', '중립': '횡보', '부정': '하락', '매우 부정': '하락'}
        prediction = [mapping[l] for l in prediction]
        
        # 혼동 행렬
        labels = ["상승", "하락"]
        cm = confusion_matrix(label, prediction, labels=labels)
        
        # 한글 폰트 설정 (Windows, Mac, Linux 환경에 맞게 설정)
        # 윈도우
        plt.rc('font', family='Malgun Gothic')
    
        # 시각화
        plt.subplot(1, 4, idx + 1)  # 1행 3열 중 i번째 위치에 subplot을 생성
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.title(f'{data} Accuracy Confusion Matrix')
        
    plt.tight_layout()
    plt.savefig(f"preprocessed_data/llm/predict_total/predict_accuracy.png")
    plt.close()

get_pred_accuracy()