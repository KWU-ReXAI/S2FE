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

folder_name = f"result_{args.ensemble}_{args.use_all}"
dir = f"./result/{folder_name}/{args.test_dir}"
if os.path.isdir(dir) == False:
    os.mkdir(dir)
dir = f"./result/{folder_name}/"
test_dir = f"./result/{folder_name}/{args.test_dir}"
test_dir = f"./result/{folder_name}/{args.test_dir}"
test_dir = f"./result/{folder_name}/{args.test_dir}"



# header=[0, 1]로 멀티 인덱스 컬럼을, index_col=0으로 인덱스를 지정합니다.
avg_df_All= pd.read_csv(f"./result/result_S3CE_All/test_result_dir/test_result_average.csv", header=[0, 1], index_col=0, encoding='utf-8-sig')
avg_df_Sector= pd.read_csv(f"./result/result_S3CE_Sector/test_result_dir/test_result_average.csv", header=[0, 1], index_col=0, encoding='utf-8-sig')
avg_df_SectorAll= pd.read_csv(f"./result/result_S3CE_SectorAll/test_result_dir/test_result_average.csv", header=[0, 1], index_col=0, encoding='utf-8-sig')


metrics = ["CAGR", "Sharpe Ratio", "MDD"]


phases = avg_df_SectorAll["Average"].columns

fig, axs = plt.subplots(nrows=1, ncols=len(metrics), figsize=(6 * len(metrics), 6))

print("Merge Result Processing . . .")

for i, metric in enumerate(metrics):
    ax = axs[i] if len(metrics) > 1 else axs

    ax.plot(phases, avg_df_SectorAll["Average"].loc[metric], 'r-', marker='o', label='SectorCluster+All')
    ax.plot(phases, avg_df_All["Average"].loc[metric], 'g-', marker='o', label='All')
    ax.plot(phases, avg_df_Sector["Average"].loc[metric], 'y-', marker='o', label='SectorCluster')
    ax.set_title(metric)
    ax.set_xlabel("Phase",labelpad=-0.5)
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend(loc='upper left')

    avg_values_SectorAll = [f"{val:.4f}" for val in avg_df_SectorAll["Average"].loc[metric]]
    avg_values_All = [f"{val:.4f}" for val in avg_df_All["Average"].loc[metric]]
    avg_values_Sector = [f"{val:.4f}" for val in avg_df_Sector["Average"].loc[metric]]

    # Phase별 실수 배열
    phase_vals_SectorAll = avg_df_SectorAll["Average"].loc[metric].values
    phase_vals_All = avg_df_All["Average"].loc[metric].values
    phase_vals_Sector = avg_df_Sector["Average"].loc[metric].values

    if metric == "CAGR":
        avg_values_SectorAll = [f"{val * 100:.2f}%" for val in phase_vals_SectorAll]
        avg_values_All = [f"{val * 100:.2f}%" for val in phase_vals_All]
        avg_values_Sector = [f"{val * 100:.2f}%" for val in phase_vals_Sector]
        # 1) 산술평균
        arith_avg_SectorAll = phase_vals_SectorAll.mean()
        arith_avg_All = phase_vals_All.mean()
        arith_avg_Sector = phase_vals_Sector.mean()
        # 2) 기하평균 (기존 방식 유지)
        # 원시 returns 대신 (1 + returns) 에서 기하평균을 구하고 다시 1을 빼기
        geo_avg_SectorAll = np.prod(1 + phase_vals_SectorAll) ** (1.0 / len(phases)) - 1
        geo_avg_All = np.prod(1 + phase_vals_All) ** (1.0 / len(phases)) - 1
        geo_avg_Sector = np.prod(1 + phase_vals_Sector) ** (1.0 / len(phases)) - 1

        # 두 개 모두 리스트에 추가
        avg_values_SectorAll.extend([f"{arith_avg_SectorAll*100:.2f}%", f"{geo_avg_SectorAll*100:.2f}%"])
        avg_values_All.extend([f"{arith_avg_All * 100:.2f}%", f"{geo_avg_All * 100:.2f}%"])
        avg_values_Sector.extend([f"{arith_avg_Sector * 100:.2f}%", f"{geo_avg_Sector * 100:.2f}%"])
        col_labels = list(phases) + ["Average", "Total"]
    else:
        overall_avg_SectorAll = phase_vals_SectorAll.mean()
        overall_avg_All = phase_vals_All.mean()
        overall_avg_Sector = phase_vals_Sector.mean()

        avg_values_SectorAll.append(f"{overall_avg_SectorAll:.4f}")
        avg_values_All.append(f"{overall_avg_All:.4f}")
        avg_values_Sector.append(f"{overall_avg_Sector:.4f}")
        col_labels = list(phases) + ["Average"]

    # 테이블 생성
    table = ax.table(
        cellText=[avg_values_SectorAll, avg_values_All, avg_values_Sector],
        rowLabels=["SectorCluster+All", "All","SectorCluster"],
        colLabels=col_labels,
        cellLoc='center',
        bbox=[0, -0.40, 1, 0.30]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

# 전체 서브플롯 간의 좌우 간격 및 하단 여백 조정
plt.subplots_adjust(bottom=0.5, wspace=0.3)
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig(f"./result/SectorAll_Sector_All_CompareGraph.png")
print("Done")