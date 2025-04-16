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
from shap.plots.colors import red_blue
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

os.chdir(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser() # 입력 받을 하이퍼파라미터 설정
parser.add_argument('--train_dir',type=str,nargs='?',default="train_result_dir") # 불러올 모델 폴더
parser.add_argument('--xai_dir',type=str,nargs='?',default="xai_v2_result_dir") # 결과 디렉토리 명
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
    df = pd.read_csv(column_path, index_col=0 if sector == "ALL" else 1)
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

            shap_values_mlp = explainer_mlp.shap_values(torch.Tensor(data).to(sector_model.device), check_additivity=False)
            shap_values_anfis = explainer_anfis.shap_values(torch.Tensor(data).to(sector_model.device))
            shap_values_rf = explainer_rf.shap_values(data, check_additivity=False)

            shap_values = shap_values_mlp[:, :, 0] + shap_values_anfis[:, :, 0] + shap_values_rf
            shap_values /= 3
            phase_shap.append(shap_values)

        if len(sectors_shap) != 1 + cluster_n:
            sectors_shap.append(phase_shap)
        else:
            for idx in range(len(phase_shap)):
                sectors_shap[sector_idx][idx] += phase_shap[idx]

# 이상치 대체(IQR)
for i in range(len(sectors_shap)):
    for j in range(len(sectors_shap[i])):
        Q1 = np.percentile(sectors_shap[i][j], 25, axis=0)
        Q3 = np.percentile(sectors_shap[i][j], 75, axis=0)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        sectors_shap[i][j] = np.clip(sectors_shap[i][j], lower, upper)

# 그래프 포맷 조정 필요
for idx, sector in enumerate(["ALL"] + DM.cluster_list):
    fig, axs = plt.subplots(2, 2, figsize=(30, 16), constrained_layout=True)
    for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        phase = x * 2 + y
        plt.sca(axs[x, y])  # axs[x, y]를 현재 active axis로 설정
        shap.plots.violin(shap_values=sectors_shap[idx][phase], features=data_list[idx][phase], feature_names=feature_names[idx], show=False, color_bar=False)
        axs[x,y].set_title(f"Phase {phase+1}")

        # 폰트 크기 조절때문에 직접 그린 color bar
        m = cm.ScalarMappable(cmap=red_blue)
        m.set_array([0, 1])
        cb = plt.colorbar(m, ax=plt.gca(), ticks=[0, 1], aspect=80)
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Feature Value", size=8, labelpad=0)
        cb.ax.tick_params(labelsize=8, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)

        axs[x, y].tick_params(axis='x', labelsize=8)
        axs[x, y].tick_params(axis='y', labelsize=8)
        axs[x, y].set_xlabel("SHAP value (impact on model output)", fontsize=8)
    fig.suptitle(f"{sector} Analysis", fontsize=20)
    plt.savefig(f"{dir}/{sector}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.clf()