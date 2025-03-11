import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
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
    print(f"\nTest for Train_Model_{K}")
    list_num_stocks = [] # 각 실험에서 선택된 주식 개수를 저장할 리스트
    for m in range(2,3): # 20번 실행
        result = {} # 백테스팅 결과를 저장할 딕셔너리
        num_stocks = [] # 각 phase에서 선택된 주식 개수를 저장하는 리스트
        for phase in tqdm(phase_list): # 각 phase 별 진행상태를 시각적으로 출력
            model = joblib.load(f"{dir}/{args.train_dir}_{K}/train_result_model_{K}_{phase}/model.joblib")  # 저장된 모델 불러옴
            cagr, sharpe, mdd, num_stock_tmp = model.backtest(verbose=True,agg="inter",use_all="SectorAll",inter_n=0.2,isTest=True, testNum=K, dir=args.test_dir) # 백테스팅 실행
            # 상위 20% 주식만을 선택
            num_stocks.append(num_stock_tmp) # 선택된 주식 개수를 저장
            result[phase] = {"CAGR":cagr,"Sharpe Ratio":sharpe,"MDD":mdd} # 백테스팅 결과 저장
        result_df = pd.DataFrame(result)
        all_results[f"Test {K}"] = result_df

        list_num_stocks.append(num_stocks) # 각 실험에서 선택된 주식 개수 저장
final_df = pd.concat(all_results, axis=1)
final_df["Average"] = final_df.mean(axis=1)
final_df.to_csv(f"{dir}/test_result_dir/test_result_file_all.csv", encoding='utf-8-sig')


# 한글 폰트 및 음수 기호 깨짐 방지 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# final_df는 pd.concat(all_results, axis=1) 후, "Average" 열이 추가된 상태입니다.
# final_df.columns가 MultiIndex인 경우와 아닌 경우를 처리합니다.
if isinstance(final_df.columns, pd.MultiIndex):
    # MultiIndex의 첫 번째 레벨에서 "Test"로 시작하는 항목만 추출합니다.
    test_names = [name for name in final_df.columns.levels[0] if isinstance(name, str) and name.startswith("Test")]
    # 정렬: "Test 1", "Test 2", ... 형식으로 정렬
    test_names = sorted(test_names, key=lambda x: int(x.split()[1]))
else:
    # 단일 Index인 경우 "Test"로 시작하는 항목만 추출
    test_names = [name for name in final_df.columns if isinstance(name, str) and name.startswith("Test")]

# 서브플롯 생성 (2행 x 3열, 총 6개의 그래프)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, test in enumerate(test_names):
    # MultiIndex인 경우, final_df[test]를 통해 해당 Test에 해당하는 DataFrame 추출
    if isinstance(final_df.columns, pd.MultiIndex):
        df_test = final_df[test]
    else:
        df_test = final_df[[col for col in final_df.columns if col.startswith(test)]]

    # 그래프에서는 "Average" 열은 제외하도록 합니다.
    plot_cols = [col for col in df_test.columns if col != "Average"]

    # 각 평가 지표(CAGR, Sharpe Ratio, MDD 등)를 선 그래프로 그립니다.
    for metric in df_test.index:
        axes[i].plot(plot_cols, df_test.loc[metric, plot_cols], marker='o', label=metric)

    axes[i].set_title(f"{test} - 각 Phase 별 평가 지표 변화")
    axes[i].set_xlabel("Phase")
    axes[i].set_ylabel("평가 지표 값")
    axes[i].legend(fontsize=8)
    axes[i].grid(True)

plt.tight_layout()
plt.show()
