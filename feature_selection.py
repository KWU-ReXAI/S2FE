from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm # 진행상황 표시를 위해 추가
import numpy as np
import random

def forward_selection(X, y, n_features_to_select, n_runs=10):
    seeds = [random.randint(1, 100) for _ in range(n_runs)]
    selected_features = []
    remaining_features = list(X.columns)

    # n_features_to_select 만큼 특징을 선택할 때까지 반복
    while len(selected_features) < n_features_to_select and remaining_features:
        mse_with_candidates = []

        # 남아있는 특징들을 하나씩 후보로 테스트
        for feature in remaining_features:
            current_features = selected_features + [feature]
            mses = []

            # n_runs 만큼 반복하여 평균 MSE 계산
            for i in range(n_runs):
                model = RandomForestRegressor(random_state=seeds[i])
                model.fit(X[current_features], y)
                y_pred = model.predict(X[current_features])
                mse = mean_squared_error(y, y_pred)
                mses.append(mse)

            avg_mse = np.mean(mses)
            mse_with_candidates.append((feature, avg_mse))

        # 평균 MSE가 가장 낮은 특징을 베스트 특징으로 선택
        mse_with_candidates.sort(key=lambda x: x[1])
        best_feature, best_mse_for_step = mse_with_candidates[0]

        # 베스트 특징을 선택된 특징 리스트에 추가하고, 남은 특징 리스트에서 제거
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        print(f"Step {len(selected_features)}: Added '{best_feature}' with avg_mse: {best_mse_for_step:.4f}")

    # 최종 선택된 특징들로 마지막 모델을 훈련하여 중요도 계산
    final_model = RandomForestRegressor()
    final_model.fit(X[selected_features], y)
    feature_importances = final_model.feature_importances_

    # 결과를 DataFrame으로 정리
    result_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return result_df


def random_forest_feature_selection(X, y, n_features, n_runs=10):
    importances_df = pd.DataFrame(index=X.columns)
    seeds = [random.randint(1, 100) for _ in range(n_runs)]
    for i in tqdm(range(n_runs)):
        # 매번 다른 random_state로 모델 생성 및 학습
        rgr = RandomForestRegressor(random_state=seeds[i])
        rgr.fit(X, y)
        importances_df[f'run_{i}'] = rgr.feature_importances_

    # 각 특징의 평균 중요도 계산
    mean_importances = importances_df.mean(axis=1)

    # 평균 중요도를 기준으로 정렬하고 상위 n개 선택
    selected_features = mean_importances.sort_values(ascending=False).head(n_features)

    result = pd.DataFrame({
        'Feature': selected_features.index,
        'Average_Importance': selected_features.values
    })

    return result


def backward_elimination(X, y, n_features_to_keep, n_runs=10):
    seeds = [random.randint(1, 100) for _ in range(n_runs)]
    remaining_features = list(X.columns)

    # 특징의 개수가 원하는 개수가 될 때까지 반복
    while len(remaining_features) > n_features_to_keep:
        # 현재 남은 특징들로 n_runs 만큼 훈련하여 평균 중요도 계산
        total_importances = pd.Series(np.zeros(len(remaining_features)), index=remaining_features)

        for i in range(n_runs):
            model = RandomForestRegressor(random_state=seeds[i])
            model.fit(X[remaining_features], y)
            total_importances += model.feature_importances_

        avg_importances = total_importances / n_runs

        # 평균 중요도가 가장 낮은 특징을 찾아서 제거
        feature_to_remove = avg_importances.idxmin()
        remaining_features.remove(feature_to_remove)
        print(f"Removed '{feature_to_remove}'. Remaining features: {len(remaining_features)}")

    # 최종 남은 특징들로 모델을 훈련하여 중요도 계산
    final_model = RandomForestRegressor()
    final_model.fit(X[remaining_features], y)
    final_importances = final_model.feature_importances_

    # 결과를 DataFrame으로 정리
    result_df = pd.DataFrame({
        'Feature': remaining_features,
        'Importance': final_importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return result_df