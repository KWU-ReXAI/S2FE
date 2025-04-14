from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import statsmodels.api as sm

def preprocess_and_calculate_vif(df, threshold=10.0):
    df_filtered = df.copy()
    feature_cols = df_filtered.columns.tolist()

    excluded_cols_for_vif = []

    for col in feature_cols:
        if (df_filtered[col] == 0).mean() > 0.5:  # 0 값이 50% 이상인 열 제외
            excluded_cols_for_vif.append(col)

    df_filtered = df_filtered.drop(columns=excluded_cols_for_vif)

    feature_cols_for_vif = df_filtered.columns.tolist()
    print(f"VIF 계산을 위한 변수 수: {len(feature_cols_for_vif)}")

    X = df_filtered[feature_cols_for_vif]
    X = sm.add_constant(X)  # 상수항을 추가

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data.sort_values(by='Variable', ascending=False)

    print(vif_data)

    high_vif_features = vif_data[vif_data["VIF"] > threshold]["Variable"].tolist()


    if "const" in high_vif_features:
        high_vif_features.remove("const")

    remove_features = high_vif_features
    print(f"VIF가 {threshold} 이상인 변수들: {remove_features}")

    return remove_features, vif_data

def remove_highly_correlated_features(df, correlation_threshold):
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = df_numeric.corr()

    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                colname = corr_matrix.columns[i]
                to_drop.add(colname)

    print(f"제거할 변수 목록 (상관계수 {correlation_threshold} 이상):")
    print(to_drop)


    return to_drop, corr_matrix