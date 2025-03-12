import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from datamanager import DataManager
import pandas as pd
from tqdm import tqdm
import torch
import joblib
import argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser() # 입력 받을 하이퍼파라미터 설정
parser.add_argument('--train_dir',type=str,nargs='?',default="train_result_dir") # 결과 파일명
parser.add_argument('--test_dir',type=str,nargs='?',default="test_result_dir") # 결과 디렉토리 명
args = parser.parse_args()
DM = DataManager(6) # 특징 개수 4개로 설정하여 데이터 매니저 초기화
DM.create_date_list()
device = torch.device('cpu')

phase_list = DM.phase_list.keys()

dir = f"./result/{args.test_dir}"
if os.path.isdir(dir) == False:
    os.mkdir(dir)
dir = f"./result"

use_all_list =["All","Sector","SectorAll"] # 모델 평가 방식
all_results={}
for K in range(1,7): # 한번만 실행
    print(f"\nTest for Train_Model_{K}",flush=True)
    list_num_stocks = [] # 각 실험에서 선택된 주식 개수를 저장할 리스트
    for m in range(2,3): # 20번 실행
        result = {} # 백테스팅 결과를 저장할 딕셔너리
        result_ks = {}
        num_stocks = [] # 각 phase에서 선택된 주식 개수를 저장하는 리스트
        for phase in tqdm(phase_list): # 각 phase 별 진행상태를 시각적으로 출력
            model = joblib.load(f"{dir}/{args.train_dir}_{K}/train_result_model_{K}_{phase}/model.joblib")  # 저장된 모델 불러옴
            cagr, sharpe, mdd, num_stock_tmp,cagr_ks,sharpe_ks,mdd_ks = model.backtest(verbose=True,agg="inter",use_all="SectorAll",inter_n=0.2,isTest=True, testNum=K, dir=args.test_dir) # 백테스팅 실행
            # 상위 20% 주식만을 선택
            num_stocks.append(num_stock_tmp) # 선택된 주식 개수를 저장
            result[phase] = {"CAGR":cagr,"Sharpe Ratio":sharpe,"MDD":mdd} # 백테스팅 결과 저장
            result_ks[phase] = {"CAGR":cagr_ks,"Sharpe Ratio":sharpe_ks,"MDD":mdd_ks}

        result_df = pd.DataFrame(result)
        result_df_ks = pd.DataFrame(result_ks)
        all_results[f"Test {K}"] = result_df
        list_num_stocks.append(num_stocks) # 각 실험에서 선택된 주식 개수 저장

final_df = pd.concat(all_results, axis=1)
tests = [col for col in final_df.columns.levels[0] if col.startswith("Test")]
num_tests = len(tests)
plt.rcParams['font.family'] = 'Malgun Gothic' # 음수 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
# 6개의 테스트에 대해 2행 3열의 서브플롯 생성 (테스트 수에 맞게 조정)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axs = axs.flatten()

# 각 테스트별로 그래프 그리기
for i, test in enumerate(tests):
    ax = axs[i]
    # 해당 테스트의 결과 데이터 추출 (행: 성능 지표, 열: phase)
    df_test = final_df[test]

    # 각 성능 지표에 대해 선 그래프로 표시
    for metric in df_test.index:
        ax.plot(df_test.columns, df_test.loc[metric], marker='o', label=metric)

    # 각 서브플롯에 개별 제목과 레이블 설정
    ax.set_title(f"Phase 별 평가 지표 변화(Model {i+1})")
    ax.set_xlabel("Phase")
    ax.set_ylabel("평가 지표 값")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig(f"{dir}/test_result_dir/test_result_graph.png")
final_df["Average"] = final_df.mean(axis=1)
final_df.to_csv(f"{dir}/test_result_dir/test_result_file.csv", encoding='utf-8-sig')