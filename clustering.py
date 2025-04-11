import numpy as np
import pandas as pd
from datamanager import DataManager
import os

cluster_n = 5

DM = DataManager(features_n= 6, cluster_n= cluster_n)
sector_list = DM.sector_list
DM.create_date_list()
features = {}

# 저장할 폴더 생성
output_folder = "./preprocessed_data/cluster_result"
os.makedirs(output_folder, exist_ok=True)

cluster_result = None  # 첫 번째 phase의 결과를 저장할 변수

for phase in DM.phase_list:
    dup_features = np.zeros((10, 10))
    for sector in sector_list:
        df_data, _, _ = DM.data_phase(sector, phase, True, isall=True)
        sector_features = df_data.columns[:-1]
        features[sector] = sector_features

    for i, sector_i in enumerate(sector_list):
        for j, sector_j in enumerate(sector_list):
            if sector_i == sector_j:
                continue
            if i >= j:
                continue
            dup_n = len(features[sector_i].intersection(features[sector_j]))
            dup_features[i][j] = dup_n

    df_dup_features = pd.DataFrame(dup_features, index=sector_list, columns=sector_list)

    # 클러스터 생성 과정
    clusters = []
    visited = set()

    # 최대 겹치는 feature 기준으로 클러스터링
    while len(clusters) < cluster_n:
        max_overlap = 0
        max_pair = (None, None)

        # 최대 겹치는 feature를 가진 섹터 쌍 찾기
        for i in range(len(sector_list)):
            if i in visited:
                continue
            for j in range(i + 1, len(sector_list)):
                if j in visited:
                    continue
                if dup_features[i][j] > max_overlap:
                    max_overlap = dup_features[i][j]
                    max_pair = (i, j)

        # 클러스터에 추가하고 방문 처리
        if max_pair[0] is not None:
            cluster = set([max_pair[0], max_pair[1]])
            visited.update(cluster)
            clusters.append(cluster)

    # 남은 섹터들을 가까운 클러스터에 추가
    for i in range(len(sector_list)):
        if i in visited:
            continue
        closest_cluster = max(clusters, key=lambda c: sum(dup_features[i][j] for j in c))
        closest_cluster.add(i)

    # 섹터 인덱스를 이름으로 변환하여 클러스터 출력
    sector_clusters = [[sector_list[idx] for idx in cluster] for cluster in clusters]
    print(f"{phase}: {sector_clusters}")

    # 첫 번째 phase의 결과만 저장 (모든 phase 결과가 동일하므로)
    if cluster_result is None:
        cluster_result = sector_clusters

    # CSV 파일 저장 기능 추가 (각 클러스터별 dup_features 저장)
    for idx, cluster in enumerate(clusters):
        # 클러스터에 속하는 섹터 인덱스를 이름으로 변환
        cluster_sectors = [sector_list[i] for i in cluster]
        # 클러스터 내 섹터들 간의 dup_features 추출
        cluster_dup_features = df_dup_features.loc[cluster_sectors, cluster_sectors]
        dup_filename = f"{output_folder}/dup_features_{phase}_{idx}.csv"
        cluster_dup_features.to_csv(dup_filename)

# cluster_result를 텍스트 파일로 저장
result_file = os.path.join(output_folder, "cluster_result.txt")
with open(result_file, "w", encoding="utf-8") as f:
    f.write(str(cluster_result))
