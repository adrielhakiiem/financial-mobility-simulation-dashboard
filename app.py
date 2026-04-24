from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load


DATA_PATH = Path("data/processed/final_dataset.csv")
MODELS_DIR = Path("models")
METRICS_PATH = MODELS_DIR / "metrics.csv"
TARGET_COL = "income_median"

MODEL_FILES = {
    "Linear": "linear_model.pkl",
    "Ridge": "ridge_model.pkl",
    "Random Forest": "rf_model.pkl",
}

FEATURE_GROUPS = {
    "Economic Indicators": ["poverty_absolute", "poverty_relative", "gini"],
    "Infrastructure": ["piped_water", "sanitation", "electricity"],
}

FEATURE_HELP = {
    "poverty_absolute": "Absolute poverty rate (%) for the selected district and year.",
    "poverty_relative": "Relative poverty rate (%) for the selected district and year.",
    "gini": "Gini coefficient (income inequality index).",
    "piped_water": "Share of households with piped water access (%).",
    "sanitation": "Share of households with improved sanitation (%).",
    "electricity": "Share of households with electricity access (%).",
}


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 1.6rem;
            max-width: 1200px;
        }
        .kpi-card {
            background: linear-gradient(135deg, #f6fafc 0%, #eef4f8 100%);
            border: 1px solid #dbe6ef;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.65rem;
        }
        .kpi-label {
            font-size: 0.82rem;
            color: #4f6475;
            letter-spacing: 0.01em;
            margin-bottom: 0.2rem;
        }
        .kpi-value {
            font-size: 1.2rem;
            color: #15344a;
            font-weight: 650;
            margin: 0;
        }
        .section-note {
            background: #f9fbfd;
            border-left: 4px solid #4C78A8;
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
            margin-top: 0.4rem;
            margin-bottom: 0.8rem;
            color: #29455a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"RM {float(value):,.2f}"


def render_kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <p class="kpi-value">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    missing_model_files = [
        filename for filename in MODEL_FILES.values() if not (MODELS_DIR / filename).exists()
    ]
    if missing_model_files:
        missing_text = ", ".join(missing_model_files)
        raise FileNotFoundError(f"Missing model files: {missing_text}")

    feature_cols = load(features_path)
    models = {name: load(MODELS_DIR / filename) for name, filename in MODEL_FILES.items()}
    return models, feature_cols


def predict(model: object, feature_values: dict[str, float], feature_order: list[str]) -> float:
    feature_row = pd.DataFrame([[feature_values[col] for col in feature_order]], columns=feature_order)
    return float(model.predict(feature_row)[0])


@st.cache_data
def load_metrics(metrics_path: Path = METRICS_PATH) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(
            "Missing models/metrics.csv. Add precomputed cross-validation metrics first."
        )

    metrics_df = pd.read_csv(metrics_path)
    required_cols = {"model_key", "rmse", "mae", "r2"}
    missing_cols = required_cols.difference(metrics_df.columns)
    if missing_cols:
        missing_text = ", ".join(sorted(missing_cols))
        raise ValueError(f"metrics.csv is missing required columns: {missing_text}")

    return metrics_df


def get_latest_district_row(df: pd.DataFrame, district: str) -> pd.Series:
    district_df = df[df["district"] == district].copy()
    if district_df.empty:
        raise ValueError(f"District not found: {district}")

    if "year" in district_df.columns:
        return district_df.sort_values("year").iloc[-1]
    if "date" in district_df.columns:
        return district_df.sort_values("date").iloc[-1]
    return district_df.iloc[-1]


def get_district_row(df: pd.DataFrame, district: str, year: int | None) -> pd.Series:
    district_df = df[df["district"] == district].copy()
    if district_df.empty:
        raise ValueError(f"District not found: {district}")

    if year is not None and "year" in district_df.columns:
        year_df = district_df[district_df["year"] == year]
        if not year_df.empty:
            return year_df.iloc[-1]

    return get_latest_district_row(df, district)


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
    apply_custom_style()
    st.title("Financial Mobility Simulation Dashboard")
    st.caption("Interactive district-level dashboard for income mobility prediction and scenario simulation.")

    try:
        df = load_data()
        models, feature_cols = load_models()
        metrics_df = load_metrics()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    if "district" not in df.columns:
        st.error("Dataset must contain a 'district' column.")
        st.stop()

    district_options = sorted(df["district"].dropna().unique().tolist())
    feature_bounds = get_feature_bounds(df, feature_cols)
    feature_medians = df[feature_cols].median(numeric_only=True)

    st.sidebar.header("Controls")
    selected_model_name = st.sidebar.selectbox("Model", list(models.keys()), index=2)
    selected_district = st.sidebar.selectbox("District", district_options)
    district_years = (
        df.loc[df["district"] == selected_district, "year"].dropna().astype(int).sort_values().unique().tolist()
        if "year" in df.columns
        else []
    )
    selected_year = st.sidebar.selectbox(
        "Year",
        district_years,
        index=len(district_years) - 1 if district_years else 0,
        help="Select the district-year baseline used for actual values and default simulation inputs.",
    ) if district_years else None

    with st.sidebar.expander("How to use this dashboard", expanded=False):
        st.markdown(
            """
            1. Select district, year, and model.
            2. Adjust inputs to run a simulation scenario.
            3. Use Reset to Baseline to restore official district values.
            """
        )

    selected_row = get_district_row(df, selected_district, selected_year)

    baseline_values: dict[str, float] = {}
    for feature in feature_cols:
        raw_value = selected_row.get(feature, np.nan)
        if pd.isna(raw_value):
            raw_value = feature_medians.get(feature, 0.0)
        baseline_values[feature] = float(raw_value)

    if (
        st.session_state.get("active_district") != selected_district
        or st.session_state.get("active_year") != selected_year
    ):
        for feature in feature_cols:
            st.session_state[f"input_{feature}"] = baseline_values[feature]
        st.session_state["active_district"] = selected_district
        st.session_state["active_year"] = selected_year

    if st.sidebar.button("Reset to Baseline", use_container_width=True):
        for feature in feature_cols:
            st.session_state[f"input_{feature}"] = baseline_values[feature]
        st.rerun()

    st.sidebar.subheader("Simulation Inputs")
    input_values: dict[str, float] = {}
    for group_name, group_features in FEATURE_GROUPS.items():
        st.sidebar.markdown(f"**{group_name}**")
        for feature in group_features:
            if feature not in feature_cols:
                continue
            min_v, max_v = feature_bounds[feature]
            spread = max_v - min_v
            step = max(spread / 200.0, 0.01)
            input_values[feature] = st.sidebar.number_input(
                label=feature,
                min_value=min_v,
                max_value=max_v,
                value=float(st.session_state[f"input_{feature}"]),
                step=float(step),
                key=f"input_{feature}",
                help=FEATURE_HELP.get(feature, "Simulated feature input value."),
            )

    for feature in feature_cols:
        if feature in input_values:
            continue
        min_v, max_v = feature_bounds[feature]
        spread = max_v - min_v
        step = max(spread / 200.0, 0.01)
        input_values[feature] = st.sidebar.number_input(
            label=feature,
            min_value=min_v,
            max_value=max_v,
            value=float(st.session_state[f"input_{feature}"]),
            step=float(step),
            key=f"input_{feature}",
            help=FEATURE_HELP.get(feature, "Simulated feature input value."),
        )

    selected_model = models[selected_model_name]
    predicted_income = predict(selected_model, input_values, feature_cols)
    baseline_prediction = predict(selected_model, baseline_values, feature_cols)
    actual_income = float(selected_row[TARGET_COL]) if TARGET_COL in selected_row and pd.notna(selected_row[TARGET_COL]) else np.nan

    delta_from_actual = predicted_income - actual_income if not np.isnan(actual_income) else np.nan
    delta_from_baseline = predicted_income - baseline_prediction

    tabs = st.tabs(["Overview", "Prediction & Simulation", "Model Comparison", "Visualization"])

    with tabs[0]:
        st.markdown(
            "<div class='section-note'><strong>FYP2 Implementation:</strong> "
            "This screen documents system integration of data loading, model inference, user controls, and real-time simulation outputs.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("### What This Model Predicts")
        st.write(
            "The selected model predicts district-level median household income (income_median) "
            "from poverty, inequality, and infrastructure indicators."
        )
        st.markdown("### What Simulation Means")
        st.write(
            "Simulation lets you adjust feature values to test how a district's predicted income "
            "changes under hypothetical scenarios."
        )
        st.markdown("### Data Limitations")
        st.warning(
            "This dataset is relatively small and aggregated at district-year level (not household level), "
            "so predictions are useful for scenario exploration, not causal claims."
        )

        top_col1, top_col2, top_col3, top_col4 = st.columns(4)
        top_col1.metric("State", str(selected_row.get("state", "N/A")))
        top_col2.metric("District", str(selected_row.get("district", "N/A")))
        top_col3.metric("Year", str(selected_row.get("year", "N/A")))
        top_col4.metric("Model", selected_model_name)

        actual_text = format_currency(actual_income)
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            render_kpi_card("Actual Income (Selected Baseline)", actual_text)
        with summary_col2:
            render_kpi_card("Current Predicted Income", format_currency(predicted_income))

        district_feature_df = pd.DataFrame(
            {
                "Feature": feature_cols,
                "Baseline Value": [baseline_values[f] for f in feature_cols],
                "Current Simulation Value": [input_values[f] for f in feature_cols],
            }
        )
        st.dataframe(district_feature_df, use_container_width=True)

    with tabs[1]:
        st.markdown(
            "<div class='section-note'><strong>FYP2 Testing:</strong> "
            "Use this tab to verify prediction responsiveness, baseline reset behavior, and scenario sensitivity to feature changes.</div>",
            unsafe_allow_html=True,
        )
        st.subheader("Prediction Result")
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        pred_col1.metric(
            "Predicted Income (income_median)",
            f"RM {predicted_income:,.2f}",
            delta=f"{delta_from_baseline:+,.2f} vs baseline prediction",
        )

        if np.isnan(actual_income):
            pred_col2.metric("Actual Income", "N/A")
            pred_col3.metric("Difference from Actual", "N/A")
        else:
            pred_col2.metric("Actual Income", format_currency(actual_income))
            pred_col3.metric("Difference from Actual", f"RM {delta_from_actual:+,.2f}")

        st.caption("Adjust sidebar inputs to run scenario simulation dynamically.")

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
            st.success(f"Simulation active: {len(changes)} feature(s) changed from baseline.")
            st.dataframe(pd.DataFrame(changes), use_container_width=True)
        else:
            st.info("No simulation changes yet. Inputs match district baseline values.")

    with tabs[2]:
        st.subheader("Official Model Comparison (Precomputed CV Metrics)")
        st.caption("Metrics are loaded from models/metrics.csv using cross-validation results.")

        metrics_display = metrics_df.copy()
        metrics_display = metrics_display.rename(
            columns={
                "model_key": "Model",
                "rmse": "RMSE",
                "mae": "MAE",
                "r2": "R2",
            }
        )
        metrics_display = metrics_display.sort_values("RMSE")
        best_model_name = str(metrics_display.iloc[0]["Model"])
        st.success(f"Best performing model by RMSE: {best_model_name}")
        st.dataframe(
            metrics_display[["Model", "RMSE", "MAE", "R2", *[c for c in ["cv_method", "n_splits", "random_state"] if c in metrics_display.columns]]],
            use_container_width=True,
        )

        rmse_fig, rmse_ax = plt.subplots(figsize=(8, 4))
        rmse_ax.bar(metrics_display["Model"], metrics_display["RMSE"], color=["#4C78A8", "#72B7B2", "#F58518"])
        rmse_ax.set_title("RMSE Comparison (Lower is Better)")
        rmse_ax.set_ylabel("RMSE")
        rmse_ax.set_xlabel("Model")
        rmse_ax.grid(axis="y", alpha=0.2)
        st.pyplot(rmse_fig)
        plt.close(rmse_fig)

    with tabs[3]:
        st.subheader("Feature-Income Relationship")
        viz_feature = st.selectbox("Feature for scatter plot", feature_cols, index=0)

        scatter_df = df[[viz_feature, TARGET_COL]].dropna().copy()
        scatter_fig, scatter_ax = plt.subplots(figsize=(9, 5))
        scatter_ax.scatter(
            scatter_df[viz_feature],
            scatter_df[TARGET_COL],
            alpha=0.45,
            s=28,
            color="#9AA0A6",
            label="District observations",
        )

        baseline_feature_value = baseline_values[viz_feature]
        if not np.isnan(actual_income):
            scatter_ax.scatter(
                [baseline_feature_value],
                [actual_income],
                color="#E45756",
                s=180,
                marker="*",
                label="Selected district (actual)",
            )

        scatter_ax.scatter(
            [input_values[viz_feature]],
            [predicted_income],
            color="#1F77B4",
            s=120,
            marker="X",
            label="Simulated scenario",
        )
        scatter_ax.set_title(f"{viz_feature} vs {TARGET_COL}")
        scatter_ax.set_xlabel(viz_feature)
        scatter_ax.set_ylabel(TARGET_COL)
        scatter_ax.legend(loc="best")
        scatter_ax.grid(alpha=0.2)
        st.pyplot(scatter_fig)
        plt.close(scatter_fig)

        st.subheader("Income Distribution")
        income_series = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()
        dist_fig, dist_ax = plt.subplots(figsize=(9, 4.8))
        dist_ax.hist(income_series, bins=24, color="#72B7B2", alpha=0.8, edgecolor="white")
        if not np.isnan(actual_income):
            dist_ax.axvline(actual_income, color="#E45756", linestyle="--", linewidth=2, label="Selected district actual")
        dist_ax.axvline(predicted_income, color="#1F77B4", linestyle="-", linewidth=2, label="Simulated prediction")
        dist_ax.set_title("Distribution of District income_median")
        dist_ax.set_xlabel("income_median")
        dist_ax.set_ylabel("Count")
        dist_ax.legend(loc="best")
        dist_ax.grid(alpha=0.2)
        st.pyplot(dist_fig)
        plt.close(dist_fig)


if __name__ == "__main__":
    main()
