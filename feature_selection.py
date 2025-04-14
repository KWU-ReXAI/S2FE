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
