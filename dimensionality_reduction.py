from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def preprocess_and_calculate_vif(df, threshold=10.0):
    df_filtered = df.copy()
    feature_cols = df_filtered.columns.tolist()

    excluded_cols = []
    for col in feature_cols:
        if df_filtered[col].isnull().any():
            excluded_cols.append((col, 'NaN'))
        elif (df_filtered[col] == 0).any():
            excluded_cols.append((col, '0 값'))

    if excluded_cols:
        print("제외된 열 목록 (NaN 또는 0 포함):")
        for col, reason in excluded_cols:
            missing_ratio = df_filtered[col].isnull().mean() if reason == 'NaN' else (df_filtered[col] == 0).mean()
            print(f" - {col} ({reason}, 비율: {missing_ratio:.2%})")

    feature_cols = [col for col in feature_cols if col not in [c[0] for c in excluded_cols]]
    df_filtered = df_filtered[feature_cols]

    while True:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = feature_cols
        vif_data["VIF"] = [variance_inflation_factor(df_filtered[feature_cols].values, i) for i in range(len(feature_cols))]

        for col in vif_data["Feature"]:
            if (df_filtered[col] == 0).any():
                vif_data.loc[vif_data["Feature"] == col, "VIF"] = 0.0

        print("\n현재 VIF 값:")
        print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))

        max_vif = vif_data["VIF"].max()

        if max_vif < threshold:
            print(f"\n모든 VIF가 {threshold} 미만입니다. 변수 선택 완료.")
            break

        drop_feature = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        print(f"다중공선성 제거: {drop_feature} (VIF: {max_vif:.2f})\n")
        feature_cols.remove(drop_feature)

    return feature_cols

def calculate_vif_cluster(df):
    df_clean = df.loc[:, ~df.isnull().any()]

    import pandas as pd
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_clean.columns
    vif_data["VIF"] = [variance_inflation_factor(df_clean.values, i) for i in range(df_clean.shape[1])]

    print(vif_data)
    return vif_data

def apply_pca(X: pd.DataFrame, n_components: int = 30) -> pd.DataFrame:
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    top_features = []
    used_names = set()
    for i, component in enumerate(pca.components_):
        abs_component = np.abs(component)
        top_index = np.argmax(abs_component)
        top_feature = X.columns[top_index]

        if top_feature in used_names:
            top_feature = f"{top_feature}_PC{i+1}"
        used_names.add(top_feature)

        top_features.append(top_feature)

    df_pca = pd.DataFrame(X_pca, columns=top_features, index=X.index)
    return df_pca


def apply_pca_cluster(X: pd.DataFrame, n_components: int = 30) -> pd.DataFrame:
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    top_features = []
    used_names = set()

    for i, component in enumerate(pca.components_):
        abs_component = np.abs(component)
        top_index = np.argmax(abs_component)
        top_feature = X.columns[top_index]

        if top_feature in used_names:
            top_feature = f"{top_feature}_PC{i+1}"
        used_names.add(top_feature)

        top_features.append(top_feature)

    df_pca = pd.DataFrame(X_pca, columns=top_features, index=X.index)
    return df_pca

def remove_highly_correlated_features(df, correlation_threshold=0.9):
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = df_numeric.corr()

    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > correlation_threshold:
                colname = corr_matrix.columns[i]
                to_drop.add(colname)

    print(f"제거할 변수 목록 (상관계수 {correlation_threshold} 이상):")
    print(to_drop)

    df_cleaned = df.drop(columns=to_drop)
    return df_cleaned
