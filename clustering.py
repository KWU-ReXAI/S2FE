import numpy as np
import pandas as pd
from datamanager import DataManager
import os

DM = DataManager(6)
sector_list = DM.sector_list
DM.create_date_list()
features = {}
cluster_n = 3

# 저장할 폴더 생성
output_folder = "./data_kr/clusters_output"
os.makedirs(output_folder, exist_ok=True)

for phase in DM.phase_list:
    sector_len = len(sector_list)
    dup_features = np.zeros((sector_len, sector_len))

    for sector in sector_list:
        df_data, _, _ = DM.data_phase(sector, phase, True,isall=True)
        features[sector] = set(df_data.columns[:])

    for i, sector_i in enumerate(sector_list):
        for j, sector_j in enumerate(sector_list):
            if sector_i == sector_j or i >= j:
                continue
            dup_n = len(features[sector_i].intersection(features[sector_j]))
            dup_features[i][j] = dup_n

    df_dup_features = pd.DataFrame(dup_features, index=sector_list, columns=sector_list)

    clusters = []  # 최종 클러스터 리스트
    visited = set()

    while len(clusters) < min(cluster_n, sector_len):
        min_overlap = np.inf
        min_pair = None

        for i in range(sector_len):
            if sector_list[i] in visited:
                continue
            for j in range(i + 1, sector_len):
                if sector_list[j] in visited:
                    continue
                if dup_features[i][j] < min_overlap:
                    min_overlap = dup_features[i][j]
                    min_pair = (i, j)

        if min_pair:
            sector_a, sector_b = sector_list[min_pair[0]], sector_list[min_pair[1]]
            cluster = {sector_a, sector_b}
            clusters.append(cluster)
            visited.add(sector_a)
            visited.add(sector_b)
        else:
            break

    for i in range(sector_len):
        if sector_list[i] in visited:
            continue

        if not clusters:
            clusters.append({sector_list[i]})
        else:
            farthest_cluster = min(clusters, key=lambda c: sum(
                dup_features[i][sector_list.index(j)] for j in c if j in sector_list
            ))
            farthest_cluster.add(sector_list[i])

        visited.add(sector_list[i])

    print(f"{phase}: {clusters}")

    for idx, cluster in enumerate(clusters):
        cluster_sectors = list(cluster)
        cluster_indices = [sector_list.index(sector) for sector in cluster_sectors]

        # 클러스터 내 섹터들 간의 dup_features 추출
        cluster_dup_features = df_dup_features.loc[cluster_sectors, cluster_sectors]

        # CSV 파일 저장
        dup_filename = f"{output_folder}/dup_features_{phase}_{idx}.csv"
        cluster_dup_features.to_csv(dup_filename)
        #print(f"✅ 클러스터 {idx} (Phase {phase}) dup_features 저장 완료: {dup_filename}")


