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
