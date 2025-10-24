import os
import ast
import shap
import torch
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datamanager import DataManager
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def concat_explanations(expl_list):
    """
    Explanation 객체 리스트를 안전하게 결합합니다.
    (PyTorch 텐서 및 None인 base_values 처리)
    """

    # 1. .values와 .data는 NumPy 배열로 변환
    all_values_np = [
        (e.values.cpu().numpy() if hasattr(e.values, 'cpu') else e.values)
        for e in expl_list
    ]
    all_data_np = [
        (e.data.cpu().numpy() if hasattr(e.data, 'cpu') else e.data)
        for e in expl_list
    ]

    # 2. NumPy 배열끼리 결합
    combined_values = np.concatenate(all_values_np, axis=0)
    combined_data = np.concatenate(all_data_np, axis=0)

    # 3. base_values 처리 (모두 None이면 None, 아니면 결합)
    combined_base_values = None
    if not any(e.base_values is None for e in expl_list):
        all_base_values_np = [
            (e.base_values.cpu().numpy() if hasattr(e.base_values, 'cpu') else e.base_values)
            for e in expl_list
        ]
        combined_base_values = np.concatenate(all_base_values_np, axis=0)

    # 4. 새 Explanation 객체 반환
    return shap.Explanation(
        values=combined_values,
        data=combined_data,
        base_values=combined_base_values,
        feature_names=expl_list[0].feature_names
    )

def xai(dm, model, phase):
    feature_names = [] # shape: (1 + cluster_n)
    train_datas = [] # shape: (1 + cluster_n)
    test_datas = [] # shape: (1 + cluster_n)
    # 특징 이름 및 데이터 추출
    for sector in dm.cluster_list + ["ALL"]:
        # feature_names 추출
        column_path = f"./preprocessed_data/{sector}/{sector}_feature_imp.csv"
        df = pd.read_csv(column_path, index_col=0 if sector == "ALL" else 1)
        index = df.index.tolist()
        index.insert(0, "Relative Return")  # RR 제외하고 feature selection 했을 때
        feature_names.append(index)

        # 데이터 추출
        cluster = False if sector == 'ALL' else True
        train_data, valid_data, test_data = dm.data_phase(sector, phase, cluster=cluster, model="S3CE")
        train_data = np.concatenate((train_data, valid_data), axis=0)
        a, b = train_data.shape[0], train_data.shape[1]
        train_data = train_data.reshape(a * b, -1)
        a, b = test_data.shape[0], test_data.shape[1]
        test_data = test_data.reshape(a * b, -1)
        train_datas.append(train_data)
        test_datas.append(test_data)

    sectors_shap = [] # shape: (1 + cluster_n)
    for sector_idx, sector in enumerate(tqdm(dm.cluster_list + ["ALL"], desc='클러스터 모델 별 XAI')):
        train = train_datas[sector_idx][:, :-1]
        test = test_datas[sector_idx][:, :-1]
        if sector == "ALL":
            sector_model = model.all_sector_model
        else:
            sector_model = model.sector_models[sector]

        explainer_mlp = shap.DeepExplainer(sector_model.mlp, torch.Tensor(train).to(sector_model.device))
        explainer_anfis = shap.GradientExplainer(sector_model.anfis, torch.Tensor(train).to(sector_model.device))
        explainer_rf = shap.TreeExplainer(sector_model.rf, train)

        explaination_mlp= explainer_mlp(torch.Tensor(test).to(sector_model.device))
        explaination_anfis = explainer_anfis(torch.Tensor(test).to(sector_model.device))
        explaination_rf = explainer_rf(test)

        explaination_all = explaination_mlp[:, :, 0] + explaination_anfis[:, :, 0] + explaination_rf
        explaination_all.feature_names = feature_names[sector_idx]
        sectors_shap.append(explaination_all)

    return sectors_shap

def xai_plots(dm, explanations, test_dir, testnum):
    # 각 모델별로 그래프 저장
    shap_path = os.path.join(test_dir, f'shap_result_{testnum}')
    if not os.path.isdir(shap_path):
        os.makedirs(shap_path)

    fpath = './preprocessed_data/cluster_result/cluster_result.txt'
    cluster_list = None
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
            data = ast.literal_eval(content)
            if isinstance(data, list):
                cluster_list = data
            else:
                print("오류: 파일 내용이 리스트 형식이 아닙니다.")
    except FileNotFoundError:
        print(f"오류: '{fpath}' 파일을 찾을 수 없습니다.")
    except (ValueError, SyntaxError):
        print(f"오류: '{fpath}' 파일 내용의 형식이 잘못되었습니다.")

    for idx in range(len(dm.cluster_list + ["ALL"])):
        # plot 작성
        shap.plots.beeswarm(explanations[idx], show=False)
        plt.xscale('symlog')
        plt.xticks(fontname="dejavu sans")
        sector = '_'.join(cluster_list[idx]) if idx < len(dm.cluster_list) else "ALL"
        plt.title(f"SHAP Analysis of {sector}")
        plt.savefig(f"{shap_path}/{sector}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.clf()

if __name__ == "__main__":
    DM = DataManager(features_n=6, cluster_n=5)
    DM.create_date_list()
    xai_explanations = []
    model = joblib.load(f"./result/result_S3CE_SectorAll/train_result_dir_1/train_result_model_1_p3/model.joblib")
    xai_explanations.append(xai(DM, model, 'p3'))
    model = joblib.load(f"./result/result_S3CE_SectorAll/train_result_dir_1/train_result_model_1_p4/model.joblib")
    xai_explanations.append(xai(DM, model, 'p4'))
    combined_expl_list = [
        concat_explanations(expl_tuple)
        for expl_tuple in zip(*xai_explanations)
    ]
    xai_plots(DM, combined_expl_list, "./result/result_S3CE_SectorAll/test_result_dir", 1)