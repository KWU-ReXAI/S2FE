import os
import sys
os.environ["PYTHONWARNINGS"] = "ignore"
import math
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from datamanager import DataManager
import pandas as pd
from tqdm import tqdm
import torch
import joblib
import argparse
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser() # 입력 받을 하이퍼파라미터 설정
parser.add_argument('--train_dir',type=str,nargs='?',default="train_result_dir") # 결과 파일명
parser.add_argument('--test_dir',type=str,nargs='?',default="test_result_dir") # 결과 디렉토리 명
parser.add_argument('--testNum',type=int,nargs='?',default=1) # 클러스터링 여부
parser.add_argument('--ensemble',type=str,nargs="?",default="S3CE")
parser.add_argument('--use_all',type=str,nargs="?",default="SectorAll")
parser.add_argument('--agg',type=str,nargs='?',default="inter") # inter
parser.add_argument('--inter_n',type=float,nargs='?',default=0.1) # 0.1
args = parser.parse_args()

if isinstance(args.inter_n, float) and args.inter_n.is_integer():
    args.inter_n = int(args.inter_n)

cluster_n = 5

DM = DataManager(features_n=6,cluster_n=cluster_n) # 특징 개수 4개로 설정하여 데이터 매니저 초기화
DM.create_date_list()
device = torch.device('cpu')
Impute = "SoftImpute"
phase_list = DM.phase_list.keys()
new_data = [{"Parameter": "inter_n", "Value": args.inter_n},{"Parameter": "aggregate", "Value": args.agg},
            {"Parameter": "Impute", "Value": Impute}]

# 기존 train_parameter.csv 파일 읽어오기
train_file_path = f"./result/result_{args.ensemble}/train_parameter.csv"
df_train = pd.read_csv(train_file_path)
# 새로운 데이터를 DataFrame으로 변환 후, 기존 DataFrame과 합치기
df_new = pd.DataFrame(new_data)
df_combined = pd.concat([df_train, df_new], ignore_index=True)
# 합쳐진 데이터를 test_parameter.csv로 저장하기
test_file_path = f"./result/result_{args.ensemble}/test_parameter.csv"
df_combined.to_csv(test_file_path, index=False, encoding='utf-8-sig')

dir = f"./result/result_{args.ensemble}/{args.test_dir}"
if os.path.isdir(dir) == False:
    os.mkdir(dir)
dir = f"./result/result_{args.ensemble}/"
test_dir = f"./result/result_{args.ensemble}/{args.test_dir}"

use_all_list =["All","Sector","SectorAll"] # 모델 평가 방식
all_results={}
for K in range(1,args.testNum+1): # 한번만 실행
    print(f"\nTest for Train_Model_{K}")
    list_num_stocks = [] # 각 실험에서 선택된 주식 개수를 저장할 리스트
    for m in range(2,3): # 20번 실행
        result = {} # 백테스팅 결과를 저장할 딕셔너리
        result_ks = {}
        num_stocks = [] # 각 phase에서 선택된 주식 개수를 저장하는 리스트
        for phase in tqdm(phase_list): # 각 phase 별 진행상태를 시각적으로 출력
            model = joblib.load(f"{dir}/{args.train_dir}_{K}/train_result_model_{K}_{phase}/model.joblib")  # 저장된 모델 불러옴
            if args.ensemble ==  "buyhold":
                cagr, sharpe, mdd, num_stock_tmp,cagr_ks,sharpe_ks,mdd_ks = model.backtest_BuyHold(verbose=True,withValidation= True) # 백테스팅 실행
            else:
                cagr, sharpe, mdd, num_stock_tmp, cagr_ks, sharpe_ks, mdd_ks = model.backtest(verbose=True,
                                                                                              agg=args.agg,
                                                                                              use_all=args.use_all,
                                                                                              inter_n=args.inter_n,
                                                                                              withValidation=True,
                                                                                              isTest=True, testNum=K,
                                                                                              dir=test_dir)  # 백테스팅 실행
            # 상위 20% 주식만을 선택
            num_stocks.append(num_stock_tmp) # 선택된 주식 개수를 저장
            result[phase] = {"CAGR":cagr,"Sharpe Ratio":sharpe,"MDD":mdd} # 백테스팅 결과 저장
            result_ks[phase] = {"CAGR":cagr_ks,"Sharpe Ratio":sharpe_ks,"MDD":mdd_ks}

        result_df = pd.DataFrame(result)
        result_df_ks = pd.DataFrame(result_ks)
        all_results[f"Test {K}"] = result_df
        list_num_stocks.append(num_stocks) # 각 실험에서 선택된 주식 개수 저장

final_df = pd.concat(all_results, axis=1)
avg_df = final_df.groupby(level=1, axis=1).mean()
avg_df.columns = pd.MultiIndex.from_product([["Average"], avg_df.columns])
avg_df.to_csv(f"{test_dir}/test_result_average.csv", encoding='utf-8-sig')
final_df_combined = pd.concat([final_df, avg_df], axis=1)
final_df_combined.to_csv(f"{test_dir}/test_result_file.csv", encoding='utf-8-sig')


# 평가지표 리스트
metrics = ["CAGR", "Sharpe Ratio", "MDD"]

# x축 값: avg_df의 MultiIndex 하위 컬럼(Phase)
phases = avg_df["Average"].columns

# 각 평가지표마다 하나의 그래프로 나타내도록 subplot 생성 (가로로 배열)
fig, axs = plt.subplots(nrows=1, ncols=len(metrics), figsize=(6 * len(metrics), 6))



for i, metric in enumerate(metrics):
    ax = axs[i] if len(metrics) > 1 else axs

    ax.plot(phases, avg_df["Average"].loc[metric], 'r-', marker='o', label='Average')
    ax.plot(phases, result_df_ks.loc[metric], 'y--', marker='o', label='KOSPI200')
    ax.set_title(metric)
    ax.set_xlabel("Phase",labelpad=-0.5)
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend(loc='upper left')

    avg_values = [f"{val:.4f}" for val in avg_df["Average"].loc[metric]]
    ks_values = [f"{val:.4f}" for val in result_df_ks.loc[metric]]

    # Phase별 실수 배열
    phase_vals = avg_df["Average"].loc[metric].values
    phase_vals_ks = result_df_ks.loc[metric].values

    if metric == "CAGR":
        # 1) 산술평균
        arith_avg = phase_vals.mean()
        arith_ks = phase_vals_ks.mean()
        # 2) 기하평균 (기존 방식 유지)
        # 원시 returns 대신 (1 + returns) 에서 기하평균을 구하고 다시 1을 빼기
        geo_avg = np.prod(1 + phase_vals) ** (1.0 / len(phases)) - 1
        geo_ks = np.prod(1 + phase_vals_ks) ** (1.0 / len(phases)) - 1

        # 두 개 모두 리스트에 추가
        avg_values.extend([f"{arith_avg:.4f}", f"{geo_avg:.4f}"])
        ks_values.extend([f"{arith_ks:.4f}", f"{geo_ks:.4f}"])
        col_labels = list(phases) + ["Average", "Total"]
    else:
        # Sharpe, MDD는 기존 산술평균만
        overall_avg = phase_vals.mean()
        overall_ks = phase_vals_ks.mean()
        avg_values.append(f"{overall_avg:.4f}")
        ks_values.append(f"{overall_ks:.4f}")
        col_labels = list(phases) + ["Average"]

    # 테이블 생성
    table = ax.table(
        cellText=[avg_values, ks_values],
        rowLabels=["Model", "KOSPI200"],
        colLabels=col_labels,
        cellLoc='center',
        bbox=[0, -0.40, 1, 0.30]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

# 전체 서브플롯 간의 좌우 간격 및 하단 여백 조정
plt.subplots_adjust(bottom=0.5, wspace=0.3)
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig(f"{dir}/test_result_graph.png")




