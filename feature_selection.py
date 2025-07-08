from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm # 진행상황 표시를 위해 추가
import numpy as np


def forward_selection(X, y, n_features_t, n_runs=20):
    """
    Forward Selection을 여러 번 실행하여 각 특징의 평균 중요도와 선택 빈도를 집계합니다.
    """
    print(f"Running Stable Forward Selection for {n_runs} iterations...")

    all_runs_results = []

    for i in tqdm(range(n_runs)):
        selected_features = []
        remaining_features = list(X.columns)

        while len(selected_features) < n_features_t and remaining_features:
            mse_with_candidates = []

            for feature in remaining_features:
                current_features = selected_features + [feature]
                # 각 run 내에서는 동일한 random_state를 사용해 일관성 유지
                model = RandomForestRegressor()
                model.fit(X[current_features], y)
                y_pred = model.predict(X[current_features])
                mse = mean_squared_error(y, y_pred)
                mse_with_candidates.append((feature, mse))

            mse_with_candidates.sort(key=lambda x: x[1])
            best_feature, _ = mse_with_candidates[0]

            selected_features.append(best_feature)
            remaining_features.remove(best_feature)

        # 최종 선택된 특징들의 중요도 계산
        final_model = RandomForestRegressor()
        final_model.fit(X[selected_features], y)

        importance_dict = {selected_features[j]: final_model.feature_importances_[j] for j in
                           range(len(selected_features))}
        run_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
        all_runs_results.append(run_df)

    # 모든 실행 결과 취합
    combined_df = pd.concat(all_runs_results)

    # 특징별 평균 중요도와 선택 횟수 계산
    avg_importance = combined_df.groupby('Feature')['Importance'].mean()
    selection_count = combined_df.groupby('Feature').size()

    result_df = pd.DataFrame({
        'Average_Importance': avg_importance
    }).reset_index()

    # 평균 중요도 기준으로 정렬
    result_df = result_df.sort_values(by='Average_Importance', ascending=False)

    return result_df.head(n_features_t)


def random_forest_feature_selection(X, y, n_features, n_runs=10):
    importances_df = pd.DataFrame(index=X.columns)

    for i in tqdm(range(n_runs)):
        # 매번 다른 random_state로 모델 생성 및 학습
        rgr = RandomForestRegressor()
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

def backward_elimination(X, y, n_features,n_runs=5):
    all_runs_results = []

    for i in tqdm(range(n_runs)):
        x_with_features = X.copy()

        # 각 run 내에서는 동일한 random_state를 사용
        model = RandomForestRegressor()

        while len(x_with_features.columns) > n_features:
            model.fit(x_with_features, y)
            feature_importances = pd.Series(model.feature_importances_, index=x_with_features.columns)
            # 가장 중요도가 낮은 특징 제거
            remove = feature_importances.idxmin()
            x_with_features = x_with_features.drop(columns=[remove])

        # 최종 선택된 특징과 중요도 저장
        model.fit(x_with_features, y)
        final_importances = pd.Series(model.feature_importances_, index=x_with_features.columns)

        run_df = pd.DataFrame({
            'Feature': final_importances.index,
            'Importance': final_importances.values
        })
        all_runs_results.append(run_df)

    # 모든 실행 결과 취합
    combined_df = pd.concat(all_runs_results)

    # 특징별 평균 중요도와 선택 횟수 계산
    avg_importance = combined_df.groupby('Feature')['Importance'].mean()
    selection_count = combined_df.groupby('Feature').size()

    result_df = pd.DataFrame({
        'Average_Importance': avg_importance
    }).reset_index()

    # 평균 중요도 기준으로 정렬
    result_df = result_df.sort_values(by='Average_Importance', ascending=False)

    return result_df.head(n_features)