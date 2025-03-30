import numpy as np
import pandas as pd
from datamanager import DataManager
import os

DM = DataManager(6)
sector_list = DM.sector_list
DM.create_date_list()
features = {}
cluster_n = 4

# 저장할 폴더 생성
output_folder = "./data_kr/clusters_output"
os.makedirs(output_folder, exist_ok=True)

'''
for phase in DM.phase_list:
    sector_len = len(sector_list)
    dup_features = np.zeros((sector_len, sector_len))

    for sector in sector_list:
        df_data, _, _ = DM.data_phase(sector, phase, True)
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


'''
for phase in DM.phase_list:
    dup_features = np.zeros((20, 20))
    for sector in sector_list:
        df_data, _, _ = DM.data_phase(sector, phase, True,isall=True)
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
    print(f"{phase}:{sector_clusters}")

    # CSV 파일 저장 기능 추가 (각 클러스터별 dup_features 저장)
    for idx, cluster in enumerate(clusters):
        # 클러스터에 속하는 섹터 인덱스를 이름으로 변환
        cluster_sectors = [sector_list[i] for i in cluster]
        # 클러스터 내 섹터들 간의 dup_features 추출
        cluster_dup_features = df_dup_features.loc[cluster_sectors, cluster_sectors]
        dup_filename = f"{output_folder}/dup_features_{phase}_{idx}.csv"
        cluster_dup_features.to_csv(dup_filename)

