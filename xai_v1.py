import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings('ignore')
from datamanager import DataManager
from tqdm import tqdm
import torch
import pandas as pd
import joblib
import argparse
import shap
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

os.chdir(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser() # 입력 받을 하이퍼파라미터 설정
parser.add_argument('--train_dir',type=str,nargs='?',default="train_result_dir") # 불러올 모델 폴더
parser.add_argument('--xai_dir',type=str,nargs='?',default="xai_result_dir") # 결과 디렉토리 명
parser.add_argument('--trainNum',type=int,nargs='?',default=5) # 훈련 횟수
parser.add_argument('--cluster_n',type=int,nargs='?',default=5)

args = parser.parse_args()
trainNum = args.trainNum
cluster_n = args.cluster_n

DM = DataManager(features_n=6,cluster_n=cluster_n)
DM.create_date_list()
phase_list = DM.phase_list.keys()

dir = f"./result/{args.xai_dir}"
if not os.path.isdir(dir):
    os.mkdir(dir)

feature_names = [] # shape: (1 + cluster_n, )
data_list = [] # shape: (1 + cluster_n, phase)
# 특징 이름 및 데이터 추출
for sector in ["ALL"] + DM.cluster_list:
    # feature_names 추출
    column_path = f"./preprocessed_data/{sector}/{sector}_feature_imp.csv"
    df = pd.read_csv(column_path, index_col=0)
    index = df.index.tolist()
    index.insert(0, "Relative Return")  # RR 제외하고 feature selection 했을 때
    feature_names.append(index)

    # 데이터 추출
    phase_data = []
    for phase in phase_list:
        _, _, test_tmp = DM.data_phase(sector, phase)
        test_data = test_tmp.reshape(test_tmp.shape[0] * test_tmp.shape[1], -1)
        data = test_data[:, :-1]
        phase_data.append(data)
    data_list.append(phase_data)

sectors_shap = [] # shape: (1 + cluster_n, phase)
for num in tqdm(range(1, trainNum+1), desc="Test Progress"):
    models = [] # 페이즈 별 모델 저장
    for phase in phase_list:
        model = joblib.load(f"./result/{args.train_dir}_{num}/train_result_model_{num}_{phase}/model.joblib")  # 페이즈 별 저장된 모델 불러옴
        models.append(model)
    for sector_idx, sector in enumerate(["ALL"] + DM.cluster_list):
        phase_shap = []
        for phase_idx, phase in enumerate(phase_list):
            data = data_list[sector_idx][phase_idx]
            if sector == "ALL":
                sector_model = models[phase_idx].all_sector_model
            else:
                sector_model = models[phase_idx].sector_models[sector]

            explainer_mlp = shap.DeepExplainer(sector_model.mlp, torch.Tensor(data).to(sector_model.device))
            explainer_anfis = shap.GradientExplainer(sector_model.anfis, torch.Tensor(data).to(sector_model.device))
            explainer_rf = shap.TreeExplainer(sector_model.rf, data)

            shap_values_mlp = explainer_mlp.shap_values(torch.Tensor(data).to(sector_model.device))
            shap_values_anfis = explainer_anfis.shap_values(torch.Tensor(data).to(sector_model.device))
            shap_values_rf = explainer_rf.shap_values(data)

            shap_values = shap_values_mlp[:, :, 0] + shap_values_anfis[:, :, 0] + shap_values_rf
            shap_values /= 3
            phase_shap.append(shap_values)

        if len(sectors_shap) != 1 + cluster_n:
            sectors_shap.append(phase_shap)
        else:
            for idx in range(len(phase_shap)):
                sectors_shap[sector_idx][idx] += phase_shap[idx]

# 데이터 정규화
eps = 1e-8  # 0 나눔 방지
for idx in range(len(data_list)):
     data_list[idx] = [
         (arr - np.mean(arr, axis=0)) / (np.std(arr, axis=0) + eps)
         for arr in data_list[idx]
     ]

# vstack for graph
features = [np.vstack(tuple(x)) for x in data_list]
shap_values = [np.vstack(tuple(x)) for x in sectors_shap]

# 이상치 대체(IQR)
shap_values = [np.clip(
                x,
                np.percentile(x, 25, axis=0) - 1.5 * (np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)),
                np.percentile(x, 75, axis=0) + 1.5 * (np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0))
                ) for x in shap_values]

for idx, sector in enumerate(["ALL"] + DM.cluster_list):
    shap.plots.violin(shap_values=shap_values[idx], features=features[idx], feature_names=feature_names[idx],show=False)
    plt.title(f"{sector} Analysis")
    plt.savefig(f"{dir}/{sector}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.clf()