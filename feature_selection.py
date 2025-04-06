from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import mean_squared_error

def random_forest_feature_selection(X, y, n_features_t):
    rgr = RandomForestRegressor()
    rgr.fit(X, y)
    feature_importance = pd.Series(rgr.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected_features = feature_importance.index[:n_features_t]
    return selected_features


def forward_selection(X, y, n_features_t):
    selected_features = []
    remaining_features = list(X.columns)
    best_mse = float("inf")

    while len(selected_features) < n_features_t and remaining_features:
        mse_with_candidates = []

        for feature in remaining_features:
            current_features = selected_features + [feature]
            model = RandomForestRegressor()
            model.fit(X[current_features], y)
            y_pred = model.predict(X[current_features])
            mse = mean_squared_error(y, y_pred)
            mse_with_candidates.append((feature, mse))

        mse_with_candidates.sort(key=lambda x: x[1])
        best_feature, best_mse = mse_with_candidates[0]

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        print(f"선택된 특성: {best_feature}, MSE: {best_mse:.4f}")

    return selected_features


def backward_elimination(X, y, n_features, significance_level):
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    while max(model.pvalues[1:]) > significance_level:
        remove = model.pvalues.idxmax()
        X_with_const = X_with_const.drop(columns=[remove])
        model = sm.OLS(y, X_with_const).fit()

        if len(X_with_const.columns) - 1 <= n_features:
            break

    selected_features = X_with_const.columns[1:]

    if len(selected_features) > n_features:
        selected_features = selected_features[:n_features]

    return selected_features
