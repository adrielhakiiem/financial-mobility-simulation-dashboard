from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from model_config import (
    DATA_PATH,
    FEATURE_COLS,
    MODELS_DIR,
    MODEL_SPECS,
    TARGET_COL,
    build_models,
    model_labels_by_key,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the processed dataset used for model training."""
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    return df


def preprocess_data(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Select features/target and apply the notebook's missing-value strategy."""
    X = df[feature_cols]
    y = df[target_col]

    print(f"Features shape before dropping missing: {X.shape}")
    print(f"Target shape before dropping missing: {y.shape}")
    print(f"\nMissing values in features:\n{X.isnull().sum()}")

    valid_indices = X.dropna().index
    X_clean = X.loc[valid_indices]
    y_clean = y.loc[valid_indices]

    print(f"\nFeatures shape after dropping missing: {X_clean.shape}")
    print(f"Target shape after dropping missing: {y_clean.shape}")

    return X_clean, y_clean, feature_cols


def evaluate_models(
    models: dict[str, object],
    X_clean: pd.DataFrame,
    y_clean: pd.Series,
) -> pd.DataFrame:
    """Evaluate models using the notebook's 5-fold cross-validation approach."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    labels_by_key = model_labels_by_key()

    for model_key, model in models.items():
        y_pred_cv = cross_val_predict(model, X_clean, y_clean, cv=kf)
        rmse = np.sqrt(mean_squared_error(y_clean, y_pred_cv))
        mae = mean_absolute_error(y_clean, y_pred_cv)
        r2 = r2_score(y_clean, y_pred_cv)

        results.append(
            {
                "model_key": model_key,
                "model_label": labels_by_key.get(model_key, model_key),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

    results_df = pd.DataFrame(results)

    print("\nModel performance (5-fold CV):")
    for _, row in results_df.iterrows():
        print(f"\n{row['model_label']}")
        print(f"  RMSE: {row['rmse']:.2f}")
        print(f"  MAE: {row['mae']:.2f}")
        print(f"  R2: {row['r2']:.4f}")

    return results_df


def train_models(
    models: dict[str, object],
    X_clean: pd.DataFrame,
    y_clean: pd.Series,
) -> dict[str, object]:
    """Fit all models on the full cleaned dataset for downstream reuse."""
    trained_models: dict[str, object] = {}
    for model_key, model in models.items():
        model.fit(X_clean, y_clean)
        trained_models[model_key] = model
    return trained_models


def save_artifacts(
    trained_models: dict[str, object],
    feature_cols: list[str],
    models_dir: Path = MODELS_DIR,
) -> None:
    """Persist trained models and feature metadata."""
    models_dir.mkdir(parents=True, exist_ok=True)

    for spec in MODEL_SPECS:
        joblib.dump(trained_models[spec.key], models_dir / spec.filename)
    joblib.dump(feature_cols, models_dir / "features.pkl")

    print(f"\nSaved model artifacts to: {models_dir}")
    print("No scaler/encoder artifacts were saved because none are used in this pipeline.")


def run_linear_train_test_check(X_clean: pd.DataFrame, y_clean: pd.Series) -> None:
    """Replicate the notebook's explicit linear train-test split check."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nLinear regression holdout check:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")


def main() -> None:
    df = load_data()
    X_clean, y_clean, feature_cols = preprocess_data(df)

    run_linear_train_test_check(X_clean, y_clean)

    models = build_models()
    evaluate_models(models, X_clean, y_clean)

    trained_models = train_models(models, X_clean, y_clean)
    save_artifacts(trained_models, feature_cols)


if __name__ == "__main__":
    main()
