import os
import shap
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datamanager import DataManager
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def xai(dm, model, phase, test_dir):
    feature_names = [] # shape: (1 + cluster_n)
    data_list = [] # shape: (1 + cluster_n)
    # 특징 이름 및 데이터 추출
    for sector in ["ALL"] + dm.cluster_list:
        # feature_names 추출
        column_path = f"./preprocessed_data/{sector}/{sector}_feature_imp.csv"
        df = pd.read_csv(column_path, index_col=0 if sector == "ALL" else 1)
        index = df.index.tolist()
        index.insert(0, "Relative Return")  # RR 제외하고 feature selection 했을 때
        feature_names.append(index)

        # 데이터 추출
        cluster = False if sector == 'ALL' else True
        _, _, test_data = dm.data_phase(sector, phase, cluster=cluster, model="S3CE")
        data_list.append(test_data)

    sectors_shap = [] # shape: (1 + cluster_n * quarters)
    test_start = dm.phase_list[phase][2]
    test_end = dm.phase_list[phase][3]
    for sector_idx, sector in enumerate(["ALL"] + dm.cluster_list):
        for pno in range(test_start, test_end):
            i = pno - test_start
            data = data_list[sector_idx][i, :, :-1]
            if sector == "ALL":
                sector_model = model.all_sector_model
            else:
                sector_model = model.sector_models[sector]

            explainer_mlp = shap.DeepExplainer(sector_model.mlp, torch.Tensor(data).to(sector_model.device))
            explainer_anfis = shap.GradientExplainer(sector_model.anfis, torch.Tensor(data).to(sector_model.device))
            explainer_rf = shap.TreeExplainer(sector_model.rf, data)

            shap_values_mlp = explainer_mlp(torch.Tensor(data).to(sector_model.device))
            shap_values_anfis = explainer_anfis(torch.Tensor(data).to(sector_model.device))
            shap_values_rf = explainer_rf(data)

            shap_values = shap_values_mlp[:, :, 0] + shap_values_anfis[:, :, 0] + shap_values_rf
            shap_values.feature_names = feature_names[sector_idx]
            sectors_shap.append(shap_values)

    # 각 모델별로 페이즈에 대한 그래프 저장
    for idx, sector in enumerate(["ALL"] + dm.cluster_list):
        for pno in range(test_start, test_end):
            date_str = dm.pno2date(pno)
            quarter_dir = os.path.join(test_dir, 'shap_result', phase, date_str)
            if not os.path.isdir(quarter_dir):
                os.makedirs(quarter_dir)
            i = pno - test_start
            quarter_len = test_end - test_start
            # plot 작성
            shap.plots.beeswarm(sectors_shap[idx * quarter_len + i], show=False)
            plt.xscale('symlog')
            plt.xticks(fontname="dejavu sans")
            plt.title(f"Analysis for {sector} quarter {date_str}")
            plt.savefig(f"{quarter_dir}/{date_str}_{sector}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.clf()

if __name__ == "__main__":
    DM = DataManager(features_n=6, cluster_n=5)
    DM.create_date_list()
    model = joblib.load(f"./result/result_S3CE_SectorAll/train_result_dir_1/train_result_model_1_p4/model.joblib")
    xai(DM, model, 'p4', "./result/result_S3CE_SectorAll/test_result_dir")