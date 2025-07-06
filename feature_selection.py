from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error

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

        model.fit(X[selected_features], y)
        feature_importances = model.feature_importances_


        importance_dict = {selected_features[i]: feature_importances[i] for i in range(len(selected_features))}
        selected_features_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])

    selected_features_df = selected_features_df.sort_values(by='Importance', ascending=False)

    return selected_features_df

def random_forest_feature_selection(X, y, n_features_t):
    rgr = RandomForestRegressor()
    rgr.fit(X, y)
    selected_columns = X.columns
    feature_importance = pd.Series(rgr.feature_importances_, index=selected_columns).sort_values(ascending=False)
    selected_features = feature_importance.head(n_features_t)

    result = pd.DataFrame({
        'Feature': selected_features.index,
        'Importance': selected_features.values
    })

    return result

def backward_elimination(x, y, n_features):
    model = RandomForestRegressor(random_state=42)
    model.fit(x, y)
    feature_importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
    x_with_features = x.copy()

    while len(x_with_features.columns) > n_features:
        remove = feature_importances.idxmin()
        x_with_features = x_with_features.drop(columns=[remove])

        model.fit(x_with_features, y)
        feature_importances = pd.Series(model.feature_importances_, index=x_with_features.columns).sort_values(
            ascending=False)

    final_feature_importances = pd.Series(model.feature_importances_, index=x_with_features.columns).sort_values(
        ascending=False)
    top_features = final_feature_importances.head(n_features)

    top_features_df = pd.DataFrame({
        'Feature': top_features.index,
        'Importance': top_features.values
    }).reset_index(drop=True)

    print("상위 특성 및 중요도:")
    print(top_features_df)

    return top_features_df