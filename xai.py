import os
from datamanager import DataManager
from tqdm import tqdm
import torch
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import argparse

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def xai_value(testNum=5, cluster_n=5, train_dir="train_result_dir", xai_dir="xai_result_dir"):
    DM = DataManager(features_n=6,cluster_n=cluster_n)
    DM.create_date_list()
    phase_list = DM.phase_list.keys()

    dir = f"./result/{xai_dir}"
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
    for num in tqdm(range(1, testNum+1), desc="Test Progress"):
        models = [] # 페이즈 별 모델 저장
        for phase in phase_list:
            model = joblib.load(f"./result/{train_dir}_{num}/train_result_model_{num}_{phase}/model.joblib")  # 페이즈 별 저장된 모델 불러옴
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

                shap_values_mlp = explainer_mlp.shap_values(torch.Tensor(data).to(sector_model.device), check_additivity=True)
                shap_values_anfis = explainer_anfis.shap_values(torch.Tensor(data).to(sector_model.device))
                shap_values_rf = explainer_rf.shap_values(data, check_additivity=True)

                shap_values = shap_values_mlp[:, :, 0] + shap_values_anfis[:, :, 0] + shap_values_rf
                phase_shap.append(shap_values)

            if num == 1:
                sectors_shap.append(phase_shap)
            else: # 값을 더해가며 동시에 평균 업데이트
                for idx in range(len(sectors_shap[sector_idx])):
                    sectors_shap[sector_idx][idx] *= num - 1
                    sectors_shap[sector_idx][idx] += phase_shap[idx]
                    sectors_shap[sector_idx][idx] /= num

    # 이상치 대체(IQR)
    for i in range(len(sectors_shap)):
        for j in range(len(sectors_shap[i])):
            Q1 = np.percentile(sectors_shap[i][j], 25, axis=0)
            Q3 = np.percentile(sectors_shap[i][j], 75, axis=0)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            sectors_shap[i][j] = np.clip(sectors_shap[i][j], lower, upper)

    # 각 모델별로 페이즈에 대한 그래프 저장
    for idx, sector in enumerate(["ALL"] + DM.cluster_list):
        sector_dir = os.path.join(dir, sector)
        if not os.path.isdir(sector_dir):
            os.mkdir(sector_dir)
        for phase in range(len(phase_list)):
            # plot 작성
            np.save(f"{sector_dir}/phase{phase+1}", sectors_shap[idx][phase])
            shap.plots.violin(shap_values=sectors_shap[idx][phase], features=data_list[idx][phase],
                              feature_names=feature_names[idx], show=False)
            plt.title(f"Analysis for {sector} Phase {phase+1}")
            plt.savefig(f"{sector_dir}/phase_{phase+1}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.clf()
        # all phase
        all_shap_value = np.concatenate(sectors_shap[idx], axis=0)
        all_features = np.concatenate(data_list[idx], axis=0)
        np.save(f"{sector_dir}/phaseAll", all_shap_value)
        shap.plots.violin(shap_values=all_shap_value, features=all_features,
                          feature_names=feature_names[idx], show=False)
        plt.title(f"Analysis for {sector}")
        plt.savefig(f"{sector_dir}/phase_All.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.clf()

def xai_edited_plot(cluster_n=5, xai_dir="xai_result_dir"):
    DM = DataManager(features_n=6, cluster_n=cluster_n)
    DM.create_date_list()
    phase_list = DM.phase_list.keys()

    dir = f"./result/{xai_dir}"
    if not os.path.isdir(dir):
        os.mkdir(dir)

    for sector in ["ALL"] + DM.cluster_list:
        cluster_dir = f"{dir}/{sector}"
        if not os.path.isdir(dir):
            raise Exception('먼저 전체 plot을 출력하십시오.')
        column_path = f"./preprocessed_data/{sector}/{sector}_feature_imp.csv"
        df = pd.read_csv(column_path, index_col=0 if sector == "ALL" else 1)
        feature_names = df.index.tolist()
        feature_names.insert(0, "Relative Return")  # RR 제외하고 feature selection 했을 때
        for idx, phase in enumerate(phase_list):
            _, _, test_tmp = DM.data_phase(sector, phase)
            test_data = test_tmp.reshape(test_tmp.shape[0] * test_tmp.shape[1], -1)
            data = test_data[:, :-1]
            shap_v = np.load(f"{cluster_dir}/phase{idx+1}.npy")

            if sector == "cluster_4" and idx == 3:
                indices = [1, 5]
                selected = [feature_names[i] for i in indices]
                shap.plots.violin(shap_values=shap_v[:, [1,5]], features=data[:, [1,5]],
                                  feature_names=selected, show=False)
                # plt.title()
                plt.savefig(f"sample_result.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
                plt.clf()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--testNum', type=int, nargs='?', default=5)
    args = parser.parse_args()
    # xai_value(testNum=args.testNum)
    xai_edited_plot()