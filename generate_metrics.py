import joblib
import pandas as pd
from model_config import (
    DATA_PATH,
    FEATURES_PATH,
    METRICS_PATH,
    MODELS_DIR,
    TARGET_COL,
    build_models,
    model_labels_by_key,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


def main() -> None:
    with FEATURES_PATH.open("rb") as f:
        feature_cols = joblib.load(f)

    df = pd.read_csv(DATA_PATH)
    X = df[feature_cols]
    y = df[TARGET_COL]

    valid_idx = X.dropna().index
    X_clean = X.loc[valid_idx]
    y_clean = y.loc[valid_idx]

    models = build_models()
    labels_by_key = model_labels_by_key()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for model_key, model in models.items():
        y_pred = cross_val_predict(model, X_clean, y_clean, cv=kf)
        rows.append(
            {
                "model_key": labels_by_key.get(model_key, model_key),
                "rmse": float(mean_squared_error(y_clean, y_pred) ** 0.5),
                "mae": float(mean_absolute_error(y_clean, y_pred)),
                "r2": float(r2_score(y_clean, y_pred)),
                "cv_method": "KFold",
                "n_splits": 5,
                "random_state": 42,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("rmse")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(metrics_df.to_csv(index=False))


if __name__ == "__main__":
    main()
