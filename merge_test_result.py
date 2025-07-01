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
from matplotlib.ticker import PercentFormatter  # Y축을 퍼센트로 포맷팅하기 위해 추가

os.chdir(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser()  # 입력 받을 하이퍼파라미터 설정
parser.add_argument('--train_dir', type=str, nargs='?', default="train_result_dir")  # 결과 파일명
parser.add_argument('--test_dir', type=str, nargs='?', default="test_result_dir")  # 결과 디렉토리 명
parser.add_argument('--testNum', type=int, nargs='?', default=1)  # 클러스터링 여부
parser.add_argument('--ensemble', type=str, nargs="?", default="S3CE")
parser.add_argument('--use_all', type=str, nargs="?", default="SectorAll")
parser.add_argument('--agg', type=str, nargs='?', default="inter")  # inter
parser.add_argument('--inter_n', type=float, nargs='?', default=0.1)  # 0.1
args = parser.parse_args()

folder_name = f"result_{args.ensemble}_{args.use_all}"
dir = f"./result/{folder_name}/{args.test_dir}"
if os.path.isdir(dir) == False:
    os.mkdir(dir)
dir = f"./result/{folder_name}/"
test_dir = f"./result/{folder_name}/{args.test_dir}"

# header=[0, 1]로 멀티 인덱스 컬럼을, index_col=0으로 인덱스를 지정합니다.
avg_df_All = pd.read_csv(f"./result/result_S3CE_All/test_result_dir/test_result_average.csv", header=[0, 1],
                         index_col=0, encoding='utf-8-sig')
avg_df_Sector = pd.read_csv(f"./result/result_S3CE_Sector/test_result_dir/test_result_average.csv", header=[0, 1],
                            index_col=0, encoding='utf-8-sig')
avg_df_SectorAll = pd.read_csv(f"./result/result_S3CE_SectorAll/test_result_dir/test_result_average.csv", header=[0, 1],
                               index_col=0, encoding='utf-8-sig')

metrics = ["CAGR", "Sharpe Ratio", "MDD"]
phases = avg_df_SectorAll["Average"].columns

fig, axs = plt.subplots(nrows=1, ncols=len(metrics), figsize=(6 * len(metrics), 6))

print("Merge Result Processing . . .")

# === 추가된 부분 1: 기하평균 값을 저장할 딕셔너리 초기화 ===
cagr_geo_means = {}
# =======================================================

for i, metric in enumerate(metrics):
    ax = axs[i] if len(metrics) > 1 else axs

    ax.plot(phases, avg_df_SectorAll["Average"].loc[metric], 'r-', marker='o', label='SectorCluster+All')
    ax.plot(phases, avg_df_All["Average"].loc[metric], 'g-', marker='o', label='All')
    ax.plot(phases, avg_df_Sector["Average"].loc[metric], 'b-', marker='o', label='SectorCluster')
    ax.set_title(metric)
    ax.set_xlabel("Phase", labelpad=-0.5)
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend(loc='upper left')

    avg_values_SectorAll = [f"{val:.4f}" for val in avg_df_SectorAll["Average"].loc[metric]]
    avg_values_All = [f"{val:.4f}" for val in avg_df_All["Average"].loc[metric]]
    avg_values_Sector = [f"{val:.4f}" for val in avg_df_Sector["Average"].loc[metric]]

    phase_vals_SectorAll = avg_df_SectorAll["Average"].loc[metric].values
    phase_vals_All = avg_df_All["Average"].loc[metric].values
    phase_vals_Sector = avg_df_Sector["Average"].loc[metric].values

    if metric == "CAGR":
        avg_values_SectorAll = [f"{val * 100:.2f}%" for val in phase_vals_SectorAll]
        avg_values_All = [f"{val * 100:.2f}%" for val in phase_vals_All]
        avg_values_Sector = [f"{val * 100:.2f}%" for val in phase_vals_Sector]

        arith_avg_SectorAll = phase_vals_SectorAll.mean()
        arith_avg_All = phase_vals_All.mean()
        arith_avg_Sector = phase_vals_Sector.mean()

        geo_avg_SectorAll = np.prod(1 + phase_vals_SectorAll) ** (1.0 / len(phases)) - 1
        geo_avg_All = np.prod(1 + phase_vals_All) ** (1.0 / len(phases)) - 1
        geo_avg_Sector = np.prod(1 + phase_vals_Sector) ** (1.0 / len(phases)) - 1

        # === 추가된 부분 2: 계산된 기하평균 값을 딕셔너리에 저장 ===
        cagr_geo_means['SectorCluster+All'] = geo_avg_SectorAll
        cagr_geo_means['All'] = geo_avg_All
        cagr_geo_means['SectorCluster'] = geo_avg_Sector
        # ========================================================

        avg_values_SectorAll.extend([f"{arith_avg_SectorAll * 100:.2f}%", f"{geo_avg_SectorAll * 100:.2f}%"])
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

    table = ax.table(
        cellText=[avg_values_SectorAll, avg_values_All, avg_values_Sector],
        rowLabels=["SectorCluster+All", "All", "SectorCluster"],
        colLabels=col_labels,
        cellLoc='center',
        bbox=[0, -0.40, 1, 0.30]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

plt.subplots_adjust(bottom=0.5, wspace=0.3)
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig(f"./result/SectorAll_Sector_All_CompareGraph.png")
print("Original comparison graph saved successfully.")

# === 추가된 부분 3: CAGR 기하평균 비교 막대그래프 생성 및 저장 ===
print("Creating CAGR Geometric Mean bar chart . . .")

# 그래프에 사용할 데이터 준비
models = list(cagr_geo_means.keys())
values = list(cagr_geo_means.values())
colors = ['r', 'g', 'b']  # 기존 라인 그래프와 색상 통일

# 새 그래프 생성
fig_bar, ax_bar = plt.subplots(figsize=(7, 7))

# 막대그래프 그리기
bars = ax_bar.bar(models, values, color=colors,width=0.5,zorder=3)
# ... (그래프 생성 코드 윗부분) ...

# Y축 범위와 눈금 조절
ax_bar.set_ylim(-0.08, 0.08) # Y축 범위를 -8% ~ +8%로 설정
ax_bar.set_yticks(np.arange(-0.08, 0.09, 0.04)) # -8%, -4%, 0, 4%, 8% 위치에 눈금 표시

ax_bar.tick_params(axis='y', labelsize=10) # Y축 눈금 숫자 크기
# ... (이하 코드 동일) ...
ax_bar.set_ylabel('CAGR', fontsize=12)
ax_bar.set_xlabel('Model Type', fontsize=12)

# Y축을 퍼센트 형식으로 표시
ax_bar.yaxis.set_major_formatter(PercentFormatter(1.0))
# y=0 지점에 회색 점선으로 수평선을 추가합니다 (zorder=0으로 막대 뒤에 배치).
ax_bar.grid(axis='y', color='gray', linestyle='--',linewidth=0.5,zorder=0)

ax_bar.axhline(0, color='black', linestyle='-', linewidth=0.5, zorder=0)

for bar in bars:
    yval = bar.get_height()

# --- 표 데이터 준비 ---
formatted_values = [f'{val * 100:.2f}%' for val in values]
cell_data = [formatted_values]
row_labels = ['CAGR']

# --- 표 생성 ---
table = ax_bar.table(
    cellText=cell_data,
    rowLabels=row_labels,
    colLabels=models,
    cellLoc='center',
    bbox=[0, -0.3, 1, 0.15] # [left, bottom, width, height] Bbox로 위치와 크기 미세 조정
)
table.auto_set_font_size(False)
table.set_fontsize(10)

# --- 레이아웃 조정 (표를 위한 공간 확보) ---
fig_bar.subplots_adjust(bottom=0.1)

# 레이아웃 조정 및 그래프 저장
plt.tight_layout()
plt.savefig(f"./result/SectorAll_Sector_All_CAGR_Comparison.png")
# ===============================================================

print("Done. All graphs have been saved.")