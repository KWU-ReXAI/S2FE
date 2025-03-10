import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings('ignore')

from datamanager import DataManager
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import joblib
import argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser() # 입력 받을 하이퍼파라미터 설정
parser.add_argument('--train_dir',type=str,nargs='?',default="train_result_dir") # 결과 파일명
parser.add_argument('--test_dir',type=str,nargs='?',default="test_result_dir") # 결과 디렉토리 명
args = parser.parse_args()
DM = DataManager(4) # 특징 개수 4개로 설정하여 데이터 매니저 초기화
device = torch.device('cpu')

phase_list = DM.phase_list.keys()

dir = f"./result/{args.test_dir}"
if os.path.isdir(dir) == False:
    os.mkdir(dir)
dir = f"./result"

use_all_list =["All","Sector","SectorAll"] # 모델 평가 방식

for K in range(1,6): # 한번만 실행
    list_num_stocks = [] # 각 실험에서 선택된 주식 개수를 저장할 리스트
    for m in range(2,3): # 20번 실행
        result = {} # 백테스팅 결과를 저장할 딕셔너리
        num_stocks = [] # 각 phase에서 선택된 주식 개수를 저장하는 리스트
        for phase in tqdm(phase_list): # 각 phase 별 진행상태를 시각적으로 출력
            model = joblib.load(f"{dir}/{args.train_dir}_{K}/train_result_model_{K}_{phase}/model.joblib")  # 저장된 모델 불러옴
            cagr, sharpe, mdd, num_stock_tmp = model.backtest(verbose=True,agg="inter",use_all="SectorAll",inter_n=0.2,isTest=True, testNum=m, dir=args.test_dir) # 백테스팅 실행
            # 상위 20% 주식만을 선택
            num_stocks.append(num_stock_tmp) # 선택된 주식 개수를 저장
            result[phase] = {"CAGR":cagr,"Sharpe Ratio":sharpe,"MDD":mdd} # 백테스팅 결과 저장
        pd.DataFrame(result).to_csv(f"{dir}/test_result_dir/test_result_file_{K}.csv") # result 딕셔너리를 csv 파일로 저장

        list_num_stocks.append(num_stocks) # 각 실험에서 선택된 주식 개수 저장
    DM.create_date_list()
    num_stocks_x = [DM.pno2date(i) for i in range(19,35,4)] # 백테스팅 수행 시점의 날짜 리스트
    print("\n각 Phase 별 평균 선택 주식 개수:")
    print(np.mean(list_num_stocks,axis=0))