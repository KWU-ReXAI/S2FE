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
parser.add_argument('--agg',type=str,nargs='?',default="inter") # inter
parser.add_argument('--inter_n',type=float,nargs='?',default=0.2) # 0.2
parser.add_argument('--LLM',action="store_true") # 클러스터링 여부
#parser.add_argument('--data',type=int,nargs='?',default=0) # LLM 예측 데이터 선택 (0:video, 1:text, 2:video+text)
#parser.add_argument('--LLMagg',type=str,nargs='?',default="False") # inter
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
train_file_path = "./result/train_parameter.csv"
df_train = pd.read_csv(train_file_path)
# 새로운 데이터를 DataFrame으로 변환 후, 기존 DataFrame과 합치기
df_new = pd.DataFrame(new_data)
df_combined = pd.concat([df_train, df_new], ignore_index=True)
# 합쳐진 데이터를 test_parameter.csv로 저장하기
test_file_path = "./result/test_parameter.csv"
df_combined.to_csv(test_file_path, index=False, encoding='utf-8-sig')

dir = f"./result/{args.test_dir}"
if os.path.isdir(dir) == False:
    os.mkdir(dir)
dir = f"./result"

use_all_list =["All","Sector","SectorAll"] # 모델 평가 방식
all_results={}
all_results_video={}
all_results_article={}
all_results_mix={}
for K in range(1,args.testNum+1): # 한번만 실행
    print(f"\nTest for Train_Model_{K}")
    list_num_stocks = [] # 각 실험에서 선택된 주식 개수를 저장할 리스트
    for m in range(2,3): # 20번 실행
        result = {} # 백테스팅 결과를 저장할 딕셔너리
        # result_ks = {}
        result_video = {}
        result_article = {}
        result_mix = {}
        num_stocks = [] # 각 phase에서 선택된 주식 개수를 저장하는 리스트
        for phase in tqdm(phase_list): # 각 phase 별 진행상태를 시각적으로 출력
            model = joblib.load(f"{dir}/{args.train_dir}_{K}/train_result_model_{K}_{phase}/model.joblib")  # 저장된 모델 불러옴
            cagr, sharpe, mdd, num_stock_tmp, cagr_video, sharpe_video, mdd_video, cagr_article, sharpe_article, mdd_article,cagr_mix, sharpe_mix, mdd_mix \
                = model.backtest(verbose=True,agg=args.agg,use_all="Sector",inter_n=args.inter_n,withValidation= True,
                                 isTest=True, testNum=K, dir=args.test_dir, withLLM=args.LLM) # 백테스팅 실행
                
            # 상위 20% 주식만을 선택
            num_stocks.append(num_stock_tmp) # 선택된 주식 개수를 저장
            result[phase] = {"CAGR":cagr,"Sharpe Ratio":sharpe,"MDD":mdd} # 백테스팅 결과 저장
            # result_ks[phase] = {"CAGR":cagr_ks,"Sharpe Ratio":sharpe_ks,"MDD":mdd_ks}
            result_video[phase] = {"CAGR": cagr_video, "Sharpe Ratio": sharpe_video, "MDD": mdd_video}  # 백테스팅 결과 저장
            result_article[phase] = {"CAGR": cagr_article, "Sharpe Ratio": sharpe_article, "MDD": mdd_article}  # 백테스팅 결과 저장
            result_mix[phase] = {"CAGR": cagr_mix, "Sharpe Ratio": sharpe_mix, "MDD": mdd_mix}  # 백테스팅 결과 저장

        result_df = pd.DataFrame(result)
        # result_df_ks = pd.DataFrame(result_ks)
        result_df_video = pd.DataFrame(result_video)
        result_df_article = pd.DataFrame(result_article)
        result_df_mix = pd.DataFrame(result_mix)
        all_results[f"Test {K}"] = result_df
        all_results_video[f"Test {K}"] = result_df_video
        all_results_article[f"Test {K}"] = result_df_article
        all_results_mix[f"Test {K}"] = result_df_mix
        list_num_stocks.append(num_stocks) # 각 실험에서 선택된 주식 개수 저장

final_df = pd.concat(all_results, axis=1)
avg_df = final_df.groupby(level=1, axis=1).mean()
avg_df.columns = pd.MultiIndex.from_product([["Average"], avg_df.columns])
avg_df.to_csv(f"{dir}/test_result_dir/test_result_average.csv", encoding='utf-8-sig')
final_df_combined = pd.concat([final_df, avg_df], axis=1)
final_df_combined.to_csv(f"{dir}/test_result_dir/test_result_file.csv", encoding='utf-8-sig')

# LLM 모델에 대한 테스트별 평균
final_df_video = pd.concat(all_results_video, axis=1)
avg_df_video = final_df_video.groupby(level=1, axis=1).mean()
avg_df_video.columns = pd.MultiIndex.from_product([["Average"], avg_df_video.columns])
avg_df_video.to_csv(f"{dir}/test_result_dir/video_test_result_average.csv", encoding='utf-8-sig')
final_df_video_combined = pd.concat([final_df_video, avg_df_video], axis=1)
final_df_video_combined.to_csv(f"{dir}/test_result_dir/video_test_result_file.csv", encoding='utf-8-sig')

# LLM 모델에 대한 테스트별 평균
final_df_article = pd.concat(all_results_article, axis=1)
avg_df_article= final_df_article.groupby(level=1, axis=1).mean()
avg_df_article.columns = pd.MultiIndex.from_product([["Average"], avg_df_article.columns])
avg_df_article.to_csv(f"{dir}/test_result_dir/article_test_result_average.csv", encoding='utf-8-sig')
final_df_article_combined = pd.concat([final_df_article, avg_df_article], axis=1)
final_df_article_combined.to_csv(f"{dir}/test_result_dir/article_test_result_file.csv", encoding='utf-8-sig')

# LLM 모델에 대한 테스트별 평균
final_df_mix = pd.concat(all_results_mix, axis=1)
avg_df_mix = final_df_mix.groupby(level=1, axis=1).mean()
avg_df_mix.columns = pd.MultiIndex.from_product([["Average"], avg_df_mix.columns])
avg_df_mix.to_csv(f"{dir}/test_result_dir/mix_test_result_average.csv", encoding='utf-8-sig')
final_df_mix_combined = pd.concat([final_df_mix, avg_df_mix], axis=1)
final_df_mix_combined.to_csv(f"{dir}/test_result_dir/mix_test_result_file.csv", encoding='utf-8-sig')

# 평가지표 리스트
metrics = ["CAGR", "Sharpe Ratio", "MDD"]

# x축 값: avg_df의 MultiIndex 하위 컬럼(Phase)
phases = avg_df["Average"].columns

# 각 평가지표마다 하나의 그래프로 나타내도록 subplot 생성 (가로로 배열)
fig, axs = plt.subplots(nrows=1, ncols=len(metrics), figsize=(6 * len(metrics), 6))

"""for i, metric in enumerate(metrics):
    ax = axs[i] if len(metrics) > 1 else axs

    # 그래프 출력: Average는 빨간 실선, KOSPI200은 노란 점선으로 표시
    ax.plot(phases, avg_df["Average"].loc[metric], 'r-', marker='o', label='Average')
    ax.plot(phases, result_df_ks.loc[metric], 'y--', marker='o', label='KOSPI200')
    ax.set_title(metric)
    ax.set_xlabel("Phase")
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend(loc='upper left')

    # 소수점 4자리로 포맷팅한 값을 생성
    avg_values = [f"{val:.4f}" for val in avg_df["Average"].loc[metric]]
    ks_values = [f"{val:.4f}" for val in result_df_ks.loc[metric]]
    cell_text = [avg_values, ks_values]

    # bbox 인자를 사용하여 표와 그래프 사이의 간격을 넓게 조정 (bbox=[x, y, width, height])
    table = ax.table(cellText=cell_text,
                     rowLabels=["Average", "KOSPI200"],
                     colLabels=phases,
                     cellLoc='center',
                     bbox=[0, -0.35, 1, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # y축 하한값 조정하여 그래프와 표가 겹치지 않도록 함
    lower_bound = min(min(avg_df["Average"].loc[metric]), min(result_df_ks.loc[metric]))
    ax.set_ylim(bottom=lower_bound - 0.3 * abs(lower_bound))"""

for i, metric in enumerate(metrics):
    ax = axs[i] if len(metrics) > 1 else axs

    ax.plot(phases, avg_df_video["Average"].loc[metric], 'r-', marker='o', label='Average with Video')
    ax.plot(phases, avg_df_article["Average"].loc[metric], 'g-', marker='o', label='Average with Article')
    ax.plot(phases, avg_df_mix["Average"].loc[metric], 'b-', marker='o', label='Average with Mix')
    ax.plot(phases, avg_df["Average"].loc[metric], 'y--', marker='o', label='Average without LLM')
    # ax.plot(phases, result_df_ks.loc[metric], 'y--', marker='o', label='KOSPI200')
    # ax.plot(phases, avg_df_llm["Average"].loc[metric], 'b-', marker='o', label='Average with LLM')
    ax.set_title(metric)
    ax.set_xlabel("Phase",labelpad=-0.5)
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend(loc='upper left')

    avg_values = [f"{val:.4f}" for val in avg_df["Average"].loc[metric]]
    # ks_values = [f"{val:.4f}" for val in result_df_ks.loc[metric]]
    avg_values_video = [f"{val:.4f}" for val in avg_df_video["Average"].loc[metric]]
    avg_values_article = [f"{val:.4f}" for val in avg_df_article["Average"].loc[metric]]
    avg_values_mix = [f"{val:.4f}" for val in avg_df_mix["Average"].loc[metric]]

    # Phase별 실수 배열
    phase_vals = avg_df["Average"].loc[metric].values
    # phase_vals_ks = result_df_ks.loc[metric].values
    phase_vals_video = avg_df_video["Average"].loc[metric].values
    phase_vals_article = avg_df_article["Average"].loc[metric].values
    phase_vals_mix = avg_df_mix["Average"].loc[metric].values

    """if metric == "CAGR":
        # 1) 산술평균
        arith_avg = phase_vals.mean()
        # arith_ks = phase_vals_ks.mean()
        arith_avg_llm = phase_vals_llm.mean()
        # 2) 기하평균 (기존 방식 유지)
        # 원시 returns 대신 (1 + returns) 에서 기하평균을 구하고 다시 1을 빼기
        geo_avg = np.prod(1 + phase_vals) ** (1.0 / len(phases)) - 1
        # geo_ks = np.prod(1 + phase_vals_ks) ** (1.0 / len(phases)) - 1
        geo_avg_llm = np.prod(1 + phase_vals_llm) ** (1.0 / len(phases)) - 1

        # 두 개 모두 리스트에 추가
        avg_values.extend([f"{arith_avg:.4f}", f"{geo_avg:.4f}"])
        # ks_values.extend([f"{arith_ks:.4f}", f"{geo_ks:.4f}"])
        avg_values_llm.extend([f"{arith_avg_llm:.4f}", f"{geo_avg_llm:.4f}"])
        col_labels = list(phases) + ["Average", "Total"]"""
    if metric == "CAGR":
        # CAGR 값들을 퍼센트(%) 형식으로 변환 (소수점 둘째 자리까지)
        avg_values = [f"{val * 100:.2f}%" for val in phase_vals]
        avg_values_video = [f"{val * 100:.2f}%" for val in phase_vals_video]
        avg_values_article = [f"{val * 100:.2f}%" for val in phase_vals_article]
        avg_values_mix = [f"{val * 100:.2f}%" for val in phase_vals_mix]

        # 1) 산술평균
        arith_avg = phase_vals.mean()
        arith_avg_video = phase_vals_video.mean()
        arith_avg_article = phase_vals_article.mean()
        arith_avg_mix = phase_vals_mix.mean()
        # 2) 기하평균
        geo_avg = np.prod(1 + phase_vals) ** (1.0 / len(phases)) - 1
        geo_avg_video = np.prod(1 + phase_vals_video) ** (1.0 / len(phases)) - 1
        geo_avg_article = np.prod(1 + phase_vals_article) ** (1.0 / len(phases)) - 1
        geo_avg_mix = np.prod(1 + phase_vals_mix) ** (1.0 / len(phases)) - 1

        # 평균값들도 퍼센트(%) 형식으로 변환하여 리스트에 추가
        avg_values.extend([f"{arith_avg * 100:.2f}%", f"{geo_avg * 100:.2f}%"])
        avg_values_video.extend([f"{arith_avg_video * 100:.2f}%", f"{geo_avg_video * 100:.2f}%"])
        avg_values_article.extend([f"{arith_avg_article * 100:.2f}%", f"{geo_avg_article * 100:.2f}%"])
        avg_values_mix.extend([f"{arith_avg_mix * 100:.2f}%", f"{geo_avg_mix * 100:.2f}%"])
        col_labels = list(phases) + ["Average", "Total"]
    else:
        # Sharpe, MDD는 기존 산술평균만
        overall_avg = phase_vals.mean()
        # overall_ks = phase_vals_ks.mean()
        overall_avg_video = phase_vals_video.mean()
        overall_avg_article = phase_vals_article.mean()
        overall_avg_mix = phase_vals_mix.mean()
        avg_values.append(f"{overall_avg:.4f}")
        # ks_values.append(f"{overall_ks:.4f}")
        avg_values_video.append(f"{overall_avg_video:.4f}")
        avg_values_article.append(f"{overall_avg_article:.4f}")
        avg_values_mix.append(f"{overall_avg_mix:.4f}")
        col_labels = list(phases) + ["Average"]

    # 테이블 생성
    table = ax.table(
        cellText=[avg_values_video, avg_values_article, avg_values_mix, avg_values],
        rowLabels=["Average with Video", "Average with Article", "Average with Mix", "Average without LLM"],
        # cellText=[avg_values, ks_values, avg_values_llm],
        # rowLabels=["Model without LLM", "KOSPI200", "Model with LLM"],
        colLabels=col_labels,
        cellLoc='center',
        bbox=[0, -0.40, 1, 0.30]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

# 전체 서브플롯 간의 좌우 간격 및 하단 여백 조정
plt.subplots_adjust(bottom=0.5, wspace=0.3)
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig(f"{dir}/test_result_dir/test_result_graph.png")




