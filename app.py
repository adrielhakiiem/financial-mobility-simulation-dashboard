from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = Path("data/processed/final_dataset.csv")
MODELS_DIR = Path("models")
TARGET_COL = "income_median"

MODEL_FILES = {
    "Linear": "linear_model.pkl",
    "Ridge": "ridge_model.pkl",
    "Random Forest": "rf_model.pkl",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "year" in df.columns:
        df = df.sort_values(["district", "year"])
    return df


@st.cache_resource
def load_models() -> tuple[dict[str, object], list[str]]:
    features_path = MODELS_DIR / "features.pkl"
    if not features_path.exists():
        raise FileNotFoundError("Missing models/features.pkl. Train models first.")

    feature_cols = load(features_path)
    models = {name: load(MODELS_DIR / filename) for name, filename in MODEL_FILES.items()}
    return models, feature_cols


def predict(model: object, feature_values: dict[str, float], feature_order: list[str]) -> float:
    feature_row = pd.DataFrame([[feature_values[col] for col in feature_order]], columns=feature_order)
    return float(model.predict(feature_row)[0])


@st.cache_data
def compute_model_metrics(
    df: pd.DataFrame,
    _models: dict[str, object],
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    clean = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    X = clean[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = clean[target_col]

    rows = []
    for model_name, model in _models.items():
        y_pred = model.predict(X)
        rows.append(
            {
                "Model": model_name,
                "RMSE": float(np.sqrt(mean_squared_error(y, y_pred))),
                "MAE": float(mean_absolute_error(y, y_pred)),
                "R2": float(r2_score(y, y_pred)),
            }
        )

    return pd.DataFrame(rows).sort_values("RMSE")


def get_latest_district_row(df: pd.DataFrame, district: str) -> pd.Series:
    district_df = df[df["district"] == district].copy()
    if district_df.empty:
        raise ValueError(f"District not found: {district}")

    if "year" in district_df.columns:
        return district_df.sort_values("year").iloc[-1]
    if "date" in district_df.columns:
        return district_df.sort_values("date").iloc[-1]
    return district_df.iloc[-1]


def get_feature_bounds(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for col in feature_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        col_min = float(series.min(skipna=True))
        col_max = float(series.max(skipna=True))
        if np.isnan(col_min) or np.isnan(col_max):
            col_min, col_max = 0.0, 1.0
        if col_min == col_max:
            col_max = col_min + 1.0
        bounds[col] = (col_min, col_max)
    return bounds


def main() -> None:
    st.set_page_config(page_title="Financial Mobility Simulation Dashboard", layout="wide")
    st.title("Financial Mobility Simulation Dashboard")

    df = load_data()
    models, feature_cols = load_models()
    metrics_df = compute_model_metrics(df, models, feature_cols, TARGET_COL)

    if "district" not in df.columns:
        st.error("Dataset must contain a 'district' column.")
        st.stop()

    district_options = sorted(df["district"].dropna().unique().tolist())
    feature_bounds = get_feature_bounds(df, feature_cols)
    feature_medians = df[feature_cols].median(numeric_only=True)

    st.sidebar.header("Controls")
    selected_model_name = st.sidebar.selectbox("Model", list(models.keys()), index=2)
    selected_district = st.sidebar.selectbox("District", district_options)

    selected_row = get_latest_district_row(df, selected_district)

    baseline_values: dict[str, float] = {}
    for feature in feature_cols:
        raw_value = selected_row.get(feature, np.nan)
        if pd.isna(raw_value):
            raw_value = feature_medians.get(feature, 0.0)
        baseline_values[feature] = float(raw_value)

    if st.session_state.get("active_district") != selected_district:
        for feature in feature_cols:
            st.session_state[f"input_{feature}"] = baseline_values[feature]
        st.session_state["active_district"] = selected_district

    st.sidebar.subheader("Simulation Inputs")
    input_values: dict[str, float] = {}
    for feature in feature_cols:
        min_v, max_v = feature_bounds[feature]
        spread = max_v - min_v
        step = max(spread / 200.0, 0.01)
        input_values[feature] = st.sidebar.number_input(
            label=feature,
            min_value=min_v,
            max_value=max_v,
            value=float(st.session_state[f"input_{feature}" ]),
            step=float(step),
            key=f"input_{feature}",
        )

    selected_model = models[selected_model_name]
    predicted_income = predict(selected_model, input_values, feature_cols)
    baseline_prediction = predict(selected_model, baseline_values, feature_cols)
    actual_income = float(selected_row[TARGET_COL]) if TARGET_COL in selected_row and pd.notna(selected_row[TARGET_COL]) else np.nan

    delta_from_actual = predicted_income - actual_income if not np.isnan(actual_income) else np.nan
    delta_from_baseline = predicted_income - baseline_prediction

    st.subheader("Section A - District Info")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("State", str(selected_row.get("state", "N/A")))
    info_col2.metric("District", str(selected_row.get("district", "N/A")))
    info_col3.metric("Year", str(selected_row.get("year", "N/A")))

    district_feature_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Actual District Value": [baseline_values[f] for f in feature_cols],
        }
    )
    st.dataframe(district_feature_df, use_container_width=True)

    st.subheader("Section B - Prediction")
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    pred_col1.metric(
        "Predicted Income (income_median)",
        f"RM {predicted_income:,.2f}",
        delta=f"{delta_from_baseline:+,.2f} vs district baseline",
    )

    if np.isnan(actual_income):
        pred_col2.metric("Actual Income", "N/A")
        pred_col3.metric("Difference from Actual", "N/A")
    else:
        pred_col2.metric("Actual Income", f"RM {actual_income:,.2f}")
        pred_col3.metric(
            "Difference from Actual",
            f"RM {delta_from_actual:+,.2f}",
        )

    changes = []
    for feature in feature_cols:
        current = input_values[feature]
        baseline = baseline_values[feature]
        delta = current - baseline
        if abs(delta) > 1e-12:
            changes.append(
                {
                    "Feature": feature,
                    "Baseline": baseline,
                    "Current": current,
                    "Change": delta,
                }
            )

    if changes:
        st.info(f"Simulation active: {len(changes)} feature(s) changed from district baseline.")
        st.dataframe(pd.DataFrame(changes), use_container_width=True)
    else:
        st.caption("No simulation changes yet. Inputs match district baseline values.")

    st.subheader("Section C - Model Comparison")
    st.caption("Metrics computed on available dataset rows without retraining models.")
    metrics_display = metrics_df.copy()
    for col in ["RMSE", "MAE", "R2"]:
        metrics_display[col] = metrics_display[col].map(lambda x: f"{x:,.4f}")
    st.dataframe(metrics_display, use_container_width=True)

    st.subheader("Section D - Visualization")
    viz_feature = st.selectbox("Feature for scatter plot", feature_cols, index=0)

    scatter_df = df[[viz_feature, TARGET_COL]].dropna().copy()
    st.scatter_chart(scatter_df, x=viz_feature, y=TARGET_COL, use_container_width=True)

    income_series = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()
    hist_counts, bin_edges = np.histogram(income_series, bins=24)
    labels = [f"{int(bin_edges[i])} - {int(bin_edges[i + 1])}" for i in range(len(bin_edges) - 1)]
    hist_df = pd.DataFrame({"Income Bin": labels, "Count": hist_counts}).set_index("Income Bin")
    st.bar_chart(hist_df, use_container_width=True)


if __name__ == "__main__":
    main()
