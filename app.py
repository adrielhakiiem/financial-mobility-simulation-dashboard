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
TARGET_LABEL = "Median household income"

MODEL_FILES = {
    "Linear": "linear_model.pkl",
    "Ridge": "ridge_model.pkl",
    "Random Forest": "rf_model.pkl",
}

FEATURE_GROUPS = {
    "Economic Indicators": ["poverty_absolute", "poverty_relative", "gini"],
    "Infrastructure": ["piped_water", "sanitation", "electricity"],
}

FEATURE_LABELS = {
    "poverty_absolute": "Absolute poverty rate (%)",
    "poverty_relative": "Relative poverty rate (%)",
    "gini": "Income inequality (Gini)",
    "piped_water": "Piped water access (%)",
    "sanitation": "Improved sanitation access (%)",
    "electricity": "Electricity access (%)",
}

FEATURE_HELP = {
    "poverty_absolute": "Share of households below the absolute poverty line.",
    "poverty_relative": "Share of households below the relative poverty line.",
    "gini": "How uneven income distribution is. Higher means more inequality.",
    "piped_water": "Share of households with piped water access.",
    "sanitation": "Share of households with improved sanitation.",
    "electricity": "Share of households with electricity access.",
}


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0f172a;
            --panel: #111f33;
            --panel-soft: #13243b;
            --border: #1f334d;
            --text: #e5edf5;
            --muted: #9bb0c4;
            --accent: #7fb7b1;
            --accent-2: #b6c7d6;
            --highlight: #f6bd60;
            --danger: #ef6f6c;
            --spacing-unit: 1rem;
        }
        .stApp {
            background: radial-gradient(circle at top left, #14253b 0%, #0f172a 55%, #0d1323 100%);
            color: var(--text);
            padding: var(--spacing-unit);
        }
        .block-container {
            padding: calc(var(--spacing-unit) * 1.5);
            max-width: 1200px;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: calc(var(--spacing-unit) * 1.5);
        }
        h2 {
            font-size: 2rem;
            margin-bottom: calc(var(--spacing-unit) * 1.2);
        }
        h3 {
            font-size: 1.5rem;
            margin-bottom: var(--spacing-unit);
        }
        .section-title {
            font-size: 1.25rem;
            margin-bottom: calc(var(--spacing-unit) * 0.8);
        }
        .section-subtitle {
            font-size: 1rem;
            margin-bottom: calc(var(--spacing-unit) * 0.6);
        }
        .kpi-card {
            padding: calc(var(--spacing-unit) * 0.8);
            margin-bottom: calc(var(--spacing-unit) * 0.8);
            background: linear-gradient(145deg, #14253a 0%, #111f33 100%);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: 0 8px 20px rgba(7, 14, 27, 0.35);
        }
        .kpi-label {
            font-size: 0.78rem;
            color: var(--muted);
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }
        .kpi-value {
            font-size: 1.5rem;
            color: var(--text);
            font-weight: 600;
            margin: 0;
        }
        .section-note {
            background: var(--panel);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            margin-top: 0.6rem;
            margin-bottom: 1rem;
            color: var(--muted);
        }
        .stSidebar {
            background: #0b1220;
            border-right: 1px solid var(--border);
            padding: var(--spacing-unit);
        }
        .stSidebar .sidebar-content {
            margin-bottom: calc(var(--spacing-unit) * 1.2);
        }
        .stDataFrame, .stTable {
            margin-top: var(--spacing-unit);
        }
        .chart-container {
            margin-bottom: calc(var(--spacing-unit) * 1.5);
        }
        .chart-title {
            margin-bottom: calc(var(--spacing-unit) * 0.8);
        }
        .landing-shell {
            padding-top: 0.25rem;
        }
        .hero-panel,
        .preview-panel {
            background: linear-gradient(145deg, rgba(20, 37, 58, 0.96) 0%, rgba(17, 31, 51, 0.98) 100%);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: 0 18px 45px rgba(7, 14, 27, 0.38);
        }
        .hero-panel {
            padding: 1.55rem 1.5rem 1.4rem;
            height: 100%;
        }
        .preview-panel {
            padding: 1.1rem;
            height: 100%;
        }
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.75rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--accent-2);
            margin-bottom: 0.9rem;
        }
        .hero-title {
            font-size: 2.5rem;
            line-height: 1.05;
            font-weight: 700;
            color: var(--text);
            margin: 0 0 0.8rem;
        }
        .hero-subtitle {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.55;
            margin: 0 0 1rem;
            max-width: 42rem;
        }
        .hero-copy {
            color: #bfd0df;
            font-size: 0.95rem;
            line-height: 1.6;
            margin: 0 0 1.1rem;
        }
        .feature-list {
            display: grid;
            gap: 0.6rem;
            margin: 0 0 1.1rem;
        }
        .feature-item {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            background: rgba(10, 19, 34, 0.62);
            border: 1px solid rgba(31, 51, 77, 0.82);
            border-radius: 14px;
            padding: 0.7rem 0.8rem;
            color: var(--text);
            font-size: 0.92rem;
        }
        .feature-dot {
            width: 0.6rem;
            height: 0.6rem;
            border-radius: 999px;
            background: linear-gradient(180deg, var(--accent) 0%, #6ea8fe 100%);
            box-shadow: 0 0 0 4px rgba(127, 183, 177, 0.12);
            flex: 0 0 auto;
        }
        .cta-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.85rem;
            margin-top: 1rem;
        }
        .cta-note {
            color: var(--muted);
            font-size: 0.88rem;
        }
        .cta-link {
            color: var(--accent-2);
            font-size: 0.88rem;
        }
        .preview-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.9rem;
        }
        .preview-kicker {
            color: var(--muted);
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .preview-title {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 600;
        }
        .preview-badge {
            background: rgba(127, 183, 177, 0.12);
            color: #b8e5e0;
            border: 1px solid rgba(127, 183, 177, 0.22);
            border-radius: 999px;
            font-size: 0.76rem;
            padding: 0.42rem 0.7rem;
            white-space: nowrap;
        }
        .preview-kpis {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin-bottom: 0.75rem;
        }
        .preview-kpi,
        .preview-insight,
        .preview-chart,
        .preview-grid-card {
            background: rgba(10, 19, 34, 0.65);
            border: 1px solid rgba(31, 51, 77, 0.84);
            border-radius: 16px;
        }
        .preview-kpi {
            padding: 0.75rem;
            min-height: 5.1rem;
        }
        .preview-kpi-label,
        .preview-grid-label {
            color: var(--muted);
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .preview-kpi-value {
            color: var(--text);
            font-size: 1.08rem;
            font-weight: 600;
            margin: 0;
        }
        .preview-main {
            display: grid;
            gap: 0.7rem;
        }
        .preview-chart {
            padding: 0.9rem;
        }
        .preview-bars {
            display: grid;
            gap: 0.58rem;
            margin-top: 0.65rem;
        }
        .preview-bar-row {
            display: grid;
            grid-template-columns: 92px 1fr 44px;
            gap: 0.6rem;
            align-items: center;
        }
        .preview-bar-track {
            height: 0.55rem;
            background: #0b1423;
            border: 1px solid rgba(31, 51, 77, 0.9);
            border-radius: 999px;
            overflow: hidden;
        }
        .preview-bar-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #7fb7b1 0%, #6ea8fe 100%);
        }
        .preview-bar-label,
        .preview-bar-value {
            color: var(--muted);
            font-size: 0.8rem;
        }
        .preview-grid {
            display: grid;
            grid-template-columns: 1.05fr 0.95fr;
            gap: 0.7rem;
        }
        .preview-grid-card {
            padding: 0.85rem;
        }
        .preview-grid-lines {
            display: grid;
            gap: 0.42rem;
            margin-top: 0.45rem;
        }
        .preview-grid-line {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 0.8rem;
            align-items: center;
            color: #d7e2ec;
            font-size: 0.84rem;
        }
        .preview-grid-chip {
            font-size: 0.74rem;
            color: #b8c8d9;
            background: rgba(148, 163, 184, 0.1);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
        }
        .preview-insight {
            padding: 0.85rem;
            color: #d5e1eb;
            font-size: 0.87rem;
            line-height: 1.55;
        }
        .preview-note {
            color: var(--muted);
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }
        .launch-button button {
            border-radius: 999px !important;
            padding: 0.72rem 1.2rem !important;
            font-weight: 600 !important;
            background: linear-gradient(90deg, #7fb7b1 0%, #6ea8fe 100%) !important;
            color: #07111f !important;
            border: none !important;
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


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#111f33")
    ax.tick_params(colors="#c8d6e5")
    ax.xaxis.label.set_color("#c8d6e5")
    ax.yaxis.label.set_color("#c8d6e5")
    ax.title.set_color("#e5edf5")
    for spine in ax.spines.values():
        spine.set_color("#1f334d")
    ax.grid(color="#22344d", alpha=0.4)


def format_percentile(value: float, series: pd.Series) -> float:
    if series.empty or np.isnan(value):
        return np.nan
    return float((series <= value).mean() * 100.0)


def describe_relative(value: float, median: float, label: str) -> str:
    if np.isnan(value) or np.isnan(median):
        return f"{label} is not available for comparison."
    if value >= median * 1.15:
        return f"{label} is well above the national median."
    if value >= median * 1.03:
        return f"{label} is slightly above the national median."
    if value <= median * 0.85:
        return f"{label} is well below the national median."
    if value <= median * 0.97:
        return f"{label} is slightly below the national median."
    return f"{label} is around the national median."


def describe_scenario_shift(delta: float) -> str:
    if abs(delta) < 1e-6:
        return "The scenario keeps predicted income close to the baseline."
    if delta > 0:
        return "The scenario suggests a higher predicted income than the baseline."
    return "The scenario suggests a lower predicted income than the baseline."


def build_indicator_insight(
    feature: str,
    baseline_value: float,
    scenario_value: float,
    median_value: float,
    predicted_delta: float,
) -> str:
    label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
    base_text = describe_relative(baseline_value, median_value, label)
    change = scenario_value - baseline_value
    if abs(change) < 1e-6:
        change_text = "The scenario keeps this indicator at baseline levels."
    elif change > 0:
        change_text = "The scenario increases this indicator relative to baseline."
    else:
        change_text = "The scenario reduces this indicator relative to baseline."

    if "poverty" in feature or feature == "gini":
        direction = "Higher values are generally linked to lower income outcomes."
    else:
        direction = "Higher values are generally linked to stronger income outcomes."

    shift_text = describe_scenario_shift(predicted_delta)
    return f"{base_text} {change_text} {direction} {shift_text}"


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


def go_to_dashboard() -> None:
    st.session_state["app_mode"] = "dashboard"


def render_homepage() -> None:
    st.markdown("<div class='landing-shell'>", unsafe_allow_html=True)
    left_col, right_col = st.columns([1.05, 0.95], gap="large")

    with left_col:
        st.markdown(
            "<div class='hero-panel'>"
            "<div class='eyebrow'>Civic-tech analytics platform</div>"
            "<div class='hero-title'>Financial Mobility Simulation Dashboard</div>"
            "<div class='hero-subtitle'>"
            "Interactive civic-tech platform for exploring district income conditions, poverty indicators, and infrastructure-related simulations across Malaysia."
            "</div>"
            "<div class='hero-copy'>"
            "The dashboard helps users examine how socioeconomic conditions relate to income outcomes, test policy-style scenarios, and use machine learning outputs to support clearer predictive insight."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='feature-list'>"
            "<div class='feature-item'><span class='feature-dot'></span>Predictive Income Simulation</div>"
            "<div class='feature-item'><span class='feature-dot'></span>District-Level Analytics</div>"
            "<div class='feature-item'><span class='feature-dot'></span>Infrastructure &amp; Poverty Insights</div>"
            "<div class='feature-item'><span class='feature-dot'></span>Interactive Visual Exploration</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        button_col, note_col = st.columns([0.62, 0.38], gap="small")
        with button_col:
            st.markdown("<div class='launch-button'>", unsafe_allow_html=True)
            st.button(
                "Launch Dashboard",
                type="primary",
                use_container_width=True,
                on_click=go_to_dashboard,
                key="launch_dashboard_home",
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with note_col:
            st.markdown(
                "<div class='cta-note'>View Project Overview</div>"
                "<div class='cta-link'>A concise introduction before analytics.</div>",
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown(
            "<div class='preview-panel'>"
            "<div class='preview-header'>"
            "<div>"
            "<div class='preview-kicker'>Dashboard preview</div>"
            "<div class='preview-title'>Civic analytics workspace</div>"
            "</div>"
            "<div class='preview-badge'>Interactive simulation ready</div>"
            "</div>"
            "<div class='preview-kpis'>"
            "<div class='preview-kpi'><div class='preview-kpi-label'>Median income</div><p class='preview-kpi-value'>RM 5,420</p></div>"
            "<div class='preview-kpi'><div class='preview-kpi-label'>District coverage</div><p class='preview-kpi-value'>137 districts</p></div>"
            "<div class='preview-kpi'><div class='preview-kpi-label'>Best model</div><p class='preview-kpi-value'>Random Forest</p></div>"
            "</div>"
            "<div class='preview-main'>"
            "<div class='preview-chart'>"
            "<div class='preview-kicker'>Simulation snapshot</div>"
            "<div class='preview-title' style='font-size:0.95rem;'>Income response by district condition</div>"
            "<div class='preview-bars'>"
            "<div class='preview-bar-row'><div class='preview-bar-label'>Poverty</div><div class='preview-bar-track'><div class='preview-bar-fill' style='width: 62%;'></div></div><div class='preview-bar-value'>62%</div></div>"
            "<div class='preview-bar-row'><div class='preview-bar-label'>Gini</div><div class='preview-bar-track'><div class='preview-bar-fill' style='width: 48%;'></div></div><div class='preview-bar-value'>48%</div></div>"
            "<div class='preview-bar-row'><div class='preview-bar-label'>Sanitation</div><div class='preview-bar-track'><div class='preview-bar-fill' style='width: 84%;'></div></div><div class='preview-bar-value'>84%</div></div>"
            "<div class='preview-bar-row'><div class='preview-bar-label'>Water access</div><div class='preview-bar-track'><div class='preview-bar-fill' style='width: 91%;'></div></div><div class='preview-bar-value'>91%</div></div>"
            "</div>"
            "</div>"
            "<div class='preview-grid'>"
            "<div class='preview-grid-card'>"
            "<div class='preview-grid-label'>Insight summary</div>"
            "<div class='preview-insight'>District conditions can be compared against national patterns to estimate possible income trajectories under different scenarios.</div>"
            "</div>"
            "<div class='preview-grid-card'>"
            "<div class='preview-grid-label'>Visualization grid</div>"
            "<div class='preview-grid-lines'>"
            "<div class='preview-grid-line'><span>Income vs poverty</span><span class='preview-grid-chip'>Scatter</span></div>"
            "<div class='preview-grid-line'><span>District ranking</span><span class='preview-grid-chip'>Distribution</span></div>"
            "<div class='preview-grid-line'><span>Model comparison</span><span class='preview-grid-chip'>Metrics</span></div>"
            "</div>"
            "</div>"
            "</div>"
            "<div class='preview-note'>Designed to feel like the actual dashboard while keeping the opening screen compact and presentation-ready.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


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

    if "app_mode" not in st.session_state:
        st.session_state["app_mode"] = "home"

    if st.session_state["app_mode"] == "home":
        render_homepage()
        return

    st.markdown("<span class='pill'>Civic-tech analytics</span>", unsafe_allow_html=True)
    st.title("Financial Mobility Simulation Dashboard")
    st.caption(
        "Explore district-level income outlooks with simple simulations and clear explanations."
    )

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

    st.sidebar.header("Explore settings")
    st.sidebar.caption("Choose a district and year baseline, then adjust local conditions.")
    selected_model_name = st.sidebar.selectbox("Prediction model", list(models.keys()), index=2)
    selected_district = st.sidebar.selectbox("District", district_options)
    district_years = (
        df.loc[df["district"] == selected_district, "year"].dropna().astype(int).sort_values().unique().tolist()
        if "year" in df.columns
        else []
    )
    selected_year = st.sidebar.selectbox(
        "Baseline year",
        district_years,
        index=len(district_years) - 1 if district_years else 0,
        help="This year provides the baseline values and observed income for the district.",
    ) if district_years else None

    with st.sidebar.expander("Quick guide", expanded=False):
        st.markdown(
            """
            1. Pick a district and baseline year.
            2. Adjust local conditions to explore scenarios.
            3. Reset to Baseline to return to official values.
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

    st.sidebar.subheader("District conditions")
    st.sidebar.caption("Adjust values to see how predicted income responds.")
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
                label=FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
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
            label=FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
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
            "<div class='section-note'><strong>Purpose:</strong> "
            "This dashboard explains how district conditions relate to income outcomes using simple, testable scenarios."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-title'>What this dashboard shows</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>"
            "We estimate district-level median household income based on poverty, inequality, and infrastructure indicators."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-title'>How to read the simulations</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>"
            "Adjust the district conditions to explore possible outcomes. This is a scenario tool, not a causal claim."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-title'>Interpretation cues</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>"
            "Higher poverty is generally associated with lower predicted income, while stronger infrastructure access is often linked to higher outcomes."
            "</div>",
            unsafe_allow_html=True,
        )

        top_col1, top_col2, top_col3, top_col4 = st.columns(4)
        top_col1.metric("State", str(selected_row.get("state", "N/A")))
        top_col2.metric("District", str(selected_row.get("district", "N/A")))
        top_col3.metric("Baseline year", str(selected_row.get("year", "N/A")))
        top_col4.metric("Prediction model", selected_model_name)

        actual_text = format_currency(actual_income)
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            render_kpi_card("Observed income (baseline)", actual_text)
        with summary_col2:
            render_kpi_card("Predicted income (current scenario)", format_currency(predicted_income))

        district_feature_df = pd.DataFrame(
            {
                "Indicator": [FEATURE_LABELS.get(f, f.replace("_", " ").title()) for f in feature_cols],
                "Baseline value": [baseline_values[f] for f in feature_cols],
                "Scenario value": [input_values[f] for f in feature_cols],
            }
        )
        st.dataframe(district_feature_df, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.markdown(
            "<div class='section-note'><strong>Simulation:</strong> "
            "Compare the baseline with your scenario and see how predicted income changes."
            "</div>",
            unsafe_allow_html=True,
        )
        st.subheader("Prediction result")
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        pred_col1.metric(
            "Predicted income",
            f"RM {predicted_income:,.2f}",
            delta=f"{delta_from_baseline:+,.2f} vs baseline",
        )

        if np.isnan(actual_income):
            pred_col2.metric("Observed income", "N/A")
            pred_col3.metric("Difference from observed", "N/A")
        else:
            pred_col2.metric("Observed income", format_currency(actual_income))
            pred_col3.metric("Difference from observed", f"RM {delta_from_actual:+,.2f}")

        st.caption("Adjust the sidebar values to explore a different scenario.")

        changes = []
        for feature in feature_cols:
            current = input_values[feature]
            baseline = baseline_values[feature]
            delta = current - baseline
            if abs(delta) > 1e-12:
                changes.append(
                    {
                        "Indicator": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                        "Baseline": baseline,
                        "Scenario": current,
                        "Change": delta,
                    }
                )

        if changes:
            st.success(f"Scenario active: {len(changes)} indicator(s) adjusted from baseline.")
            st.dataframe(pd.DataFrame(changes), use_container_width=True, hide_index=True)
        else:
            st.info("No scenario changes yet. Inputs match district baseline values.")

    with tabs[2]:
        st.subheader("Prediction accuracy comparison")
        st.caption("Metrics are based on cross-validated performance from the training phase.")

        metrics_display = metrics_df.copy()
        metrics_display = metrics_display.rename(
            columns={
                "model_key": "Model",
                "rmse": "Prediction error",
                "mae": "Average absolute error",
                "r2": "Explained variance (R2)",
            }
        )
        metrics_display = metrics_display.sort_values("Prediction error")
        best_model_name = str(metrics_display.iloc[0]["Model"])
        st.success(f"Best performing model (lowest error): {best_model_name}")
        st.dataframe(
            metrics_display[
                [
                    "Model",
                    "Prediction error",
                    "Average absolute error",
                    "Explained variance (R2)",
                    *[c for c in ["cv_method", "n_splits", "random_state"] if c in metrics_display.columns],
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        rmse_fig, rmse_ax = plt.subplots(figsize=(8, 4))
        rmse_fig.patch.set_facecolor("#0f172a")
        rmse_ax.bar(
            metrics_display["Model"],
            metrics_display["Prediction error"],
            color=["#7fb7b1", "#6ea8fe", "#f6bd60"],
        )
        rmse_ax.set_title("Prediction error (lower is better)")
        rmse_ax.set_ylabel("Prediction error")
        rmse_ax.set_xlabel("Model")
        style_axes(rmse_ax)
        st.pyplot(rmse_fig)
        plt.close(rmse_fig)

    with tabs[3]:
        st.subheader("District Insight Summary")
        income_series = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()
        national_medians = df[feature_cols].median(numeric_only=True)

        insight_items = [
            describe_relative(
                baseline_values.get("poverty_absolute", np.nan),
                national_medians.get("poverty_absolute", np.nan),
                "Absolute poverty",
            ),
            describe_relative(
                baseline_values.get("poverty_relative", np.nan),
                national_medians.get("poverty_relative", np.nan),
                "Relative poverty",
            ),
            describe_relative(
                baseline_values.get("gini", np.nan),
                national_medians.get("gini", np.nan),
                "Income inequality",
            ),
            describe_relative(
                baseline_values.get("electricity", np.nan),
                national_medians.get("electricity", np.nan),
                "Electricity access",
            ),
            describe_relative(
                baseline_values.get("piped_water", np.nan),
                national_medians.get("piped_water", np.nan),
                "Water access",
            ),
            describe_relative(
                baseline_values.get("sanitation", np.nan),
                national_medians.get("sanitation", np.nan),
                "Sanitation access",
            ),
            describe_scenario_shift(delta_from_baseline),
        ]
        for item in insight_items:
            st.markdown(f"- {item}", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Indicator relationships</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Each chart highlights the selected district and your scenario so you can compare typical patterns.</div>",
            unsafe_allow_html=True,
        )

        indicator_grid = [
            "poverty_absolute",
            "poverty_relative",
            "gini",
            "electricity",
            "piped_water",
            "sanitation",
        ]

        for left_feature, right_feature in zip(indicator_grid[::2], indicator_grid[1::2]):
            col_left, col_right = st.columns(2)
            for feature, col in [(left_feature, col_left), (right_feature, col_right)]:
                with col:
                    scatter_df = df[[feature, TARGET_COL]].dropna().copy()
                    scatter_fig, scatter_ax = plt.subplots(figsize=(5, 3.4))
                    scatter_fig.patch.set_facecolor("#0f172a")
                    scatter_ax.scatter(
                        scatter_df[feature],
                        scatter_df[TARGET_COL],
                        alpha=0.45,
                        s=20,
                        color="#8ca3b8",
                        label="All districts",
                    )

                    baseline_feature_value = baseline_values[feature]
                    if not np.isnan(actual_income):
                        scatter_ax.scatter(
                            [baseline_feature_value],
                            [actual_income],
                            color="#ef6f6c",
                            s=120,
                            marker="*",
                            label="Selected district",
                        )

                    scatter_ax.scatter(
                        [input_values[feature]],
                        [predicted_income],
                        color="#7fb7b1",
                        s=80,
                        marker="X",
                        label="Scenario",
                    )
                    scatter_ax.set_title(
                        f"{FEATURE_LABELS.get(feature, feature.replace('_', ' ').title())} vs {TARGET_LABEL}",
                        fontsize=10,
                    )
                    scatter_ax.set_xlabel(FEATURE_LABELS.get(feature, feature.replace("_", " ").title()), fontsize=9)
                    scatter_ax.set_ylabel(f"{TARGET_LABEL} (RM)", fontsize=9)
                    scatter_ax.legend(
                        loc="best",
                        facecolor="#0f172a",
                        edgecolor="#1f334d",
                        labelcolor="#e5edf5",
                        fontsize=8,
                    )
                    style_axes(scatter_ax)
                    st.pyplot(scatter_fig)
                    plt.close(scatter_fig)

                    median_value = national_medians.get(feature, np.nan)
                    insight_text = build_indicator_insight(
                        feature,
                        baseline_values[feature],
                        input_values[feature],
                        median_value,
                        delta_from_baseline,
                    )
                    st.caption(insight_text)

        st.subheader("Where this district stands")
        st.caption("Compare observed income with the full district distribution and the simulated scenario.")
        dist_fig, dist_ax = plt.subplots(figsize=(9, 4.8))
        dist_fig.patch.set_facecolor("#0f172a")
        dist_ax.hist(income_series, bins=24, color="#6ea8fe", alpha=0.8, edgecolor="#0f172a")
        if not np.isnan(actual_income):
            dist_ax.axvline(
                actual_income,
                color="#ef6f6c",
                linestyle="--",
                linewidth=2,
                label="Selected district (observed)",
            )
        dist_ax.axvline(
            predicted_income,
            color="#7fb7b1",
            linestyle="-",
            linewidth=2,
            label="Scenario prediction",
        )
        dist_ax.set_title("District incomes across Malaysia")
        dist_ax.set_xlabel(f"{TARGET_LABEL} (RM)")
        dist_ax.set_ylabel("Count")
        dist_ax.legend(loc="best", facecolor="#0f172a", edgecolor="#1f334d", labelcolor="#e5edf5")
        style_axes(dist_ax)
        st.pyplot(dist_fig)
        plt.close(dist_fig)

        actual_percentile = format_percentile(actual_income, income_series)
        scenario_percentile = format_percentile(predicted_income, income_series)
        if not np.isnan(actual_percentile):
            st.markdown(
                f"- Observed income is higher than {actual_percentile:.0f}% of districts."
            )
        if not np.isnan(scenario_percentile):
            st.markdown(
                f"- The scenario would place the district above {scenario_percentile:.0f}% of districts."
            )
        if not np.isnan(actual_percentile) and not np.isnan(scenario_percentile):
            if scenario_percentile > actual_percentile + 5:
                st.markdown("- The scenario moves the district into a higher income tier.")
            elif scenario_percentile < actual_percentile - 5:
                st.markdown("- The scenario moves the district into a lower income tier.")
            else:
                st.markdown("- The scenario keeps the district in a similar income tier.")


if __name__ == "__main__":
    main()
