import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


DATA_PATH = "data/processed/final_dataset.csv"
FEATURES_PATH = "models/features.pkl"
OUT_PATH = "models/metrics.csv"
TARGET_COL = "income_median"


def main() -> None:
    with open(FEATURES_PATH, "rb") as f:
        feature_cols = joblib.load(f)

    df = pd.read_csv(DATA_PATH)
    X = df[feature_cols]
    y = df[TARGET_COL]

    valid_idx = X.dropna().index
    X_clean = X.loc[valid_idx]
    y_clean = y.loc[valid_idx]

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for name, model in models.items():
        y_pred = cross_val_predict(model, X_clean, y_clean, cv=kf)
        rows.append(
            {
                "model_key": name,
                "rmse": float(mean_squared_error(y_clean, y_pred) ** 0.5),
                "mae": float(mean_absolute_error(y_clean, y_pred)),
                "r2": float(r2_score(y_clean, y_pred)),
                "cv_method": "KFold",
                "n_splits": 5,
                "random_state": 42,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("rmse")
    os.makedirs("models", exist_ok=True)
    metrics_df.to_csv(OUT_PATH, index=False)
    print(metrics_df.to_csv(index=False))


if __name__ == "__main__":
    main()
