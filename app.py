import base64
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from model_config import (
    DATA_PATH,
    DEFAULT_MODEL_LABEL,
    FEATURES_PATH,
    METRICS_PATH,
    MODELS_DIR,
    TARGET_COL,
    model_descriptions_by_label,
    model_files_by_label,
)

TARGET_LABEL = "Median household income"
ASSETS_DIR = Path("assets/images")
MAP_SOURCE_TEXT = "Source: OpenDOSM / Department of Statistics Malaysia (HIES 2022)"

MODEL_FILES = model_files_by_label()
MODEL_EXPLANATIONS = model_descriptions_by_label()
NATIONAL_MAP_SPECS = [
    {
        "title": "Median Household Income",
        "filename": "median_income.png",
        "caption": (
            "Typical household incomes are strongest in major urban and industrial regions."
        ),
    },
    {
        "title": "Mean Household Income",
        "filename": "mean_income.png",
        "caption": (
            "Average incomes are highest where overall earning power is concentrated."
        ),
    },
    {
        "title": "Poverty Rate",
        "filename": "poverty_rate.png",
        "caption": (
            "Poverty distribution varies significantly across districts."
        ),
    },
    {
        "title": "Mean Household Expenditure",
        "filename": "mean_expenditure.png",
        "caption": (
            "Household spending is generally higher in more urban and connected districts."
        ),
    },
    {
        "title": "Gini Coefficient",
        "filename": "gini_coefficient.png",
        "caption": (
            "Income inequality differs across districts even where incomes are relatively high."
        ),
    },
]

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

_DEPRIVATION_FEATURES = frozenset({"poverty_absolute", "poverty_relative", "gini"})
_POVERTY_PROJECTION_FEATURES = ("poverty_absolute", "poverty_relative")
_INFRASTRUCTURE_PROJECTION_FEATURES = ("piped_water", "sanitation", "electricity")


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
            --sidebar-label: #c9d9e8;
            --sidebar-caption: #aabecf;
            --sidebar-group-surface: rgba(17, 31, 51, 0.72);
            --sidebar-input-bg: rgba(10, 19, 34, 0.75);
            --body-support: #aabdd2;
            --body-dim: #9eb6cc;
            --insight-pos: #8ec9a8;
            --insight-neg: #e8a87c;
            --insight-neutral: #b4c4d4;
        }
        .stApp {
            background: radial-gradient(circle at top left, #14253b 0%, #0f172a 55%, #0d1323 100%);
            color: var(--text);
        }
        [data-testid="stHeader"] {
            background: rgba(15, 23, 42, 0.76);
            border-bottom: 1px solid rgba(31, 51, 77, 0.7);
        }
        .block-container {
            max-width: 1288px;
            padding-top: calc(var(--spacing-unit) * 2.5);
            padding-right: calc(var(--spacing-unit) * 1.2);
            padding-bottom: calc(var(--spacing-unit) * 1.65);
            padding-left: calc(var(--spacing-unit) * 1.2);
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
        /* Main column: page and section hierarchy (excludes sidebar) */
        [data-testid="stMain"] h1 {
            font-size: 2.28rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #f4f8fc;
            margin-top: 0.05rem;
            margin-bottom: 0.28rem;
            line-height: 1.08;
        }
        [data-testid="stMain"] h2 {
            font-size: 1.38rem;
            font-weight: 650;
            color: #edf4fb;
            margin-top: 1.45rem;
            margin-bottom: 0.42rem;
            letter-spacing: -0.018em;
        }
        [data-testid="stMain"] h3 {
            font-size: 1.04rem;
            font-weight: 600;
            color: #dfe8f2;
            margin-top: 0.95rem;
            margin-bottom: 0.38rem;
        }
        [data-testid="stMain"] .stCaption {
            color: var(--body-support) !important;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        [data-testid="stMain"] .stCaption p {
            color: inherit !important;
        }
        .page-tagline {
            font-size: 0.98rem;
            color: var(--body-support);
            line-height: 1.56;
            margin: 0 0 1.4rem;
            max-width: 50rem;
        }
        .pill {
            display: inline-block;
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            font-weight: 600;
            color: #b8e5e0;
            background: rgba(127, 183, 177, 0.14);
            border: 1px solid rgba(127, 183, 177, 0.28);
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            margin-bottom: 0.82rem;
        }
        .section-kicker {
            display: inline-block;
            font-size: 0.73rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-weight: 650;
            color: #b8dce0;
            margin: 0.12rem 0 0.42rem;
        }
        .section-title {
            font-size: 1.42rem;
            font-weight: 700;
            color: #f1f7fc;
            margin-top: 1.32rem;
            margin-bottom: 0.38rem;
            letter-spacing: -0.022em;
        }
        .section-title:first-child {
            margin-top: 0.12rem;
        }
        .section-subtitle {
            font-size: 0.97rem;
            font-weight: 400;
            color: var(--body-dim);
            line-height: 1.64;
            margin-bottom: 0.92rem;
            max-width: 47rem;
        }
        .section-subtitle--tight {
            margin-bottom: 0.5rem;
        }
        .viz-live-note {
            font-size: 0.9rem;
            color: var(--body-support);
            background: rgba(17, 31, 51, 0.55);
            border: 1px solid rgba(31, 51, 77, 0.85);
            border-radius: 12px;
            padding: 0.65rem 0.85rem;
            margin: 0.35rem 0 1rem;
            line-height: 1.5;
            max-width: 48rem;
        }
        .interpretation-list {
            margin: 0.35rem 0 1.1rem;
            padding-left: 1.15rem;
            color: var(--body-dim);
            line-height: 1.55;
            font-size: 0.95rem;
        }
        .interpretation-list li {
            margin-bottom: 0.45rem;
        }
        .insight-label {
            color: #dcecf7;
            font-weight: 600;
        }
        .model-compare-lede {
            font-size: 0.94rem;
            color: var(--body-dim);
            line-height: 1.58;
            margin: 0.25rem 0 1rem;
            max-width: 48rem;
        }
        .model-compare-divider-label {
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 600;
            color: var(--accent-2);
            margin: 1.25rem 0 0.5rem;
        }
        .insight-pos {
            color: var(--insight-pos);
            font-weight: 600;
        }
        .insight-neg {
            color: var(--insight-neg);
            font-weight: 600;
        }
        .insight-neutral {
            color: var(--insight-neutral);
            font-weight: 500;
        }
        .insight-context {
            color: var(--body-dim);
            font-weight: 400;
        }
        .insight-direction {
            color: #8a9dad;
            font-size: 0.94em;
            font-weight: 400;
        }
        .chart-insight-caption {
            font-size: 0.88rem;
            line-height: 1.55;
            margin: 0.4rem 0 0.85rem;
            color: var(--body-dim);
        }
        .block-gap-chart {
            display: block;
            height: 0.35rem;
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
            padding: 0.82rem 0.96rem;
            margin-top: 0.45rem;
            margin-bottom: 1rem;
            color: var(--body-dim);
            font-size: 0.92rem;
            line-height: 1.52;
        }
        .section-note strong {
            color: #dcecf4;
            font-weight: 600;
        }
        .national-overview-intro {
            color: var(--body-support);
            font-size: 0.95rem;
            line-height: 1.63;
            margin: 0.08rem 0 1rem;
            max-width: 56rem;
        }
        .national-map-card {
            background: linear-gradient(145deg, rgba(20, 37, 58, 0.94) 0%, rgba(17, 31, 51, 0.98) 100%);
            border: 1px solid rgba(31, 51, 77, 0.92);
            border-radius: 18px;
            box-shadow: 0 9px 22px rgba(7, 14, 27, 0.22);
            padding: 0.82rem 0.82rem 0.76rem;
            margin-bottom: 0.72rem;
            min-height: 100%;
        }
        .national-map-title {
            color: #eef4fb;
            font-size: 1rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            margin-bottom: 0.52rem;
        }
        .national-map-image-wrap {
            background: rgba(10, 19, 34, 0.68);
            border: 1px solid rgba(31, 51, 77, 0.76);
            border-radius: 14px;
            padding: 0.34rem;
            margin-bottom: 0.58rem;
        }
        .national-map-image {
            display: block;
            width: 100%;
            height: auto;
            border-radius: 10px;
        }
        .national-map-caption {
            color: var(--body-dim);
            font-size: 0.86rem;
            line-height: 1.48;
            margin: 0 0 0.48rem;
        }
        .national-map-source {
            color: var(--accent-2);
            font-size: 0.75rem;
            letter-spacing: 0.02em;
            line-height: 1.38;
            margin: 0;
        }
        .stSidebar {
            background: #0b1220;
            border-right: 1px solid var(--border);
            padding: var(--spacing-unit);
        }
        .stSidebar .sidebar-content {
            margin-bottom: calc(var(--spacing-unit) * 1.2);
        }
        /* Sidebar: scoped layout, contrast, and hierarchy (dashboard mode only) */
        [data-testid="stSidebar"] {
            color: var(--text);
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] > div > [data-testid="stVerticalBlock"] {
            gap: 0.62rem;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] .stCaption {
            color: var(--sidebar-caption) !important;
            line-height: 1.45;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] label span,
        [data-testid="stSidebar"] .stWidgetLabel p {
            color: var(--sidebar-label) !important;
            font-weight: 500;
            font-size: 0.9rem;
            letter-spacing: 0.01em;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--text) !important;
            letter-spacing: -0.01em;
            margin-top: 0.15rem;
            margin-bottom: 0.35rem;
        }
        [data-testid="stSidebar"] h1 {
            font-size: 1.15rem;
            font-weight: 700;
        }
        [data-testid="stSidebar"] h2 {
            font-size: 1.05rem;
            font-weight: 600;
        }
        [data-testid="stSidebar"] h3 {
            font-size: 1rem;
            font-weight: 600;
        }
        [data-testid="stSidebar"] hr {
            margin: 1rem 0;
            border: none;
            border-top: 1px solid rgba(31, 51, 77, 0.95);
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] {
            background-color: var(--sidebar-input-bg) !important;
            border-color: rgba(31, 51, 77, 0.95) !important;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] svg,
        [data-testid="stSidebar"] [data-baseweb="input"] svg {
            fill: var(--sidebar-label);
        }
        [data-testid="stSidebar"] .stNumberInput input {
            color: var(--text) !important;
            background-color: var(--sidebar-input-bg) !important;
            border: 1px solid rgba(31, 51, 77, 0.95) !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] .stNumberInput button {
            background-color: rgba(17, 31, 51, 0.9) !important;
            border-color: rgba(31, 51, 77, 0.95) !important;
            color: var(--sidebar-label) !important;
        }
        [data-testid="stSidebar"] details {
            background: var(--sidebar-group-surface);
            border: 1px solid rgba(31, 51, 77, 0.85);
            border-radius: 12px;
            padding: 0.15rem 0.5rem 0.45rem;
        }
        [data-testid="stSidebar"] summary {
            color: var(--text);
            font-weight: 600;
            font-size: 0.9rem;
        }
        [data-testid="stSidebar"] details .stMarkdown p,
        [data-testid="stSidebar"] details .stMarkdown li {
            color: var(--sidebar-caption);
            font-size: 0.88rem;
        }
        .sidebar-section-kicker {
            font-size: 0.68rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--accent-2);
            margin: 0 0 0.2rem;
            font-weight: 600;
        }
        .sidebar-section-kicker--spaced {
            margin-top: 0.15rem;
        }
        .sidebar-group-labelbar {
            margin: 0.65rem 0 0.45rem;
            padding: 0.48rem 0.72rem;
            background: var(--sidebar-group-surface);
            border: 1px solid rgba(31, 51, 77, 0.88);
            border-radius: 10px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #d8e8f2;
            letter-spacing: 0.06em;
            text-transform: uppercase;
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


def _sidebar_section_kicker(text: str, *, spaced: bool = False) -> None:
    classes = "sidebar-section-kicker"
    if spaced:
        classes += " sidebar-section-kicker--spaced"
    st.sidebar.markdown(f"<p class='{classes}'>{html.escape(text)}</p>", unsafe_allow_html=True)


def _sidebar_group_labelbar(title: str) -> None:
    st.sidebar.markdown(
        f"<div class='sidebar-group-labelbar'>{html.escape(title)}</div>",
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


def render_model_explanation_card(model_name: str, description: str) -> None:
    st.markdown(
        (
            "<div class='section-note'>"
            f"<strong>{html.escape(model_name)}:</strong> {html.escape(description)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


@st.cache_data
def load_image_base64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def render_national_map_card(title: str, image_path: Path, caption: str) -> None:
    if not image_path.exists():
        st.warning(f"Missing overview map: {image_path.name}")
        return

    image_b64 = load_image_base64(str(image_path))
    st.markdown(
        (
            "<div class='national-map-card'>"
            f"<div class='national-map-title'>{html.escape(title)}</div>"
            "<div class='national-map-image-wrap'>"
            f"<img class='national-map-image' src='data:image/png;base64,{image_b64}' "
            f"alt='{html.escape(title)}' />"
            "</div>"
            f"<p class='national-map-caption'>{html.escape(caption)}</p>"
            f"<p class='national-map-source'>{html.escape(MAP_SOURCE_TEXT)}</p>"
            "</div>"
        ),
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


def _scenario_indicator_change_html(feature: str, change: float) -> str:
    if abs(change) < 1e-6:
        return '<span class="insight-neutral">The scenario keeps this indicator at baseline levels.</span>'
    if change > 0:
        cls = "insight-neg" if feature in _DEPRIVATION_FEATURES else "insight-pos"
        verb = "increases"
    else:
        cls = "insight-pos" if feature in _DEPRIVATION_FEATURES else "insight-neg"
        verb = "reduces"
    return f'<span class="{cls}">The scenario {verb} this indicator relative to baseline.</span>'


def _predicted_income_shift_html(predicted_delta: float) -> str:
    if abs(predicted_delta) < 1e-6:
        return '<span class="insight-neutral">The scenario keeps predicted income close to the baseline.</span>'
    if predicted_delta > 0:
        return '<span class="insight-pos">The scenario suggests a higher predicted income than the baseline.</span>'
    return '<span class="insight-neg">The scenario suggests a lower predicted income than the baseline.</span>'


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
    change_html = _scenario_indicator_change_html(feature, change)

    if "poverty" in feature or feature == "gini":
        direction = "Higher values are generally linked to lower income outcomes."
    else:
        direction = "Higher values are generally linked to stronger income outcomes."
    direction_html = f'<span class="insight-direction">{html.escape(direction)}</span>'
    shift_html = _predicted_income_shift_html(predicted_delta)
    base_html = f'<span class="insight-context">{html.escape(base_text)}</span>'
    return (
        f'<p class="chart-insight-caption">{base_html} {change_html} {direction_html} '
        f"{shift_html}</p>"
    )


def format_baseline_insight_list_item(feature_key: str, value: float, median: float) -> str:
    label = FEATURE_LABELS.get(feature_key, feature_key.replace("_", " ").title())
    text = describe_relative(value, median, label)
    escaped = html.escape(text)
    escaped_label = html.escape(label)
    if escaped_label in escaped:
        escaped = escaped.replace(
            escaped_label,
            f'<strong class="insight-label">{escaped_label}</strong>',
            1,
        )
    return f'<li><span class="insight-context">{escaped}</span></li>'


def format_scenario_shift_list_item(delta: float) -> str:
    text = describe_scenario_shift(delta)
    if abs(delta) < 1e-6:
        inner = f'<span class="insight-neutral">{html.escape(text)}</span>'
    elif delta > 0:
        inner = f'<span class="insight-pos">{html.escape(text)}</span>'
    else:
        inner = f'<span class="insight-neg">{html.escape(text)}</span>'
    return f"<li>{inner}</li>"


def format_distribution_insights_html(
    actual_percentile: float,
    scenario_percentile: float,
) -> str:
    parts: list[str] = []
    if not np.isnan(actual_percentile):
        parts.append(
            "<li><span class=\"insight-context\">Observed income is higher than "
            f'<span class="insight-label">{actual_percentile:.0f}%</span> of districts.</span></li>'
        )
    if not np.isnan(scenario_percentile):
        parts.append(
            "<li><span class=\"insight-context\">The scenario would place the district above "
            f'<span class="insight-label">{scenario_percentile:.0f}%</span> of districts.</span></li>'
        )
    if not np.isnan(actual_percentile) and not np.isnan(scenario_percentile):
        if scenario_percentile > actual_percentile + 5:
            tier = '<span class="insight-pos">The scenario moves the district into a higher income tier.</span>'
        elif scenario_percentile < actual_percentile - 5:
            tier = '<span class="insight-neg">The scenario moves the district into a lower income tier.</span>'
        else:
            tier = '<span class="insight-neutral">The scenario keeps the district in a similar income tier.</span>'
        parts.append(f"<li>{tier}</li>")
    if not parts:
        return ""
    return '<ul class="interpretation-list">' + "".join(parts) + "</ul>"


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
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Missing models/features.pkl. Train models first.")

    missing_model_files = [
        filename for filename in MODEL_FILES.values() if not (MODELS_DIR / filename).exists()
    ]
    if missing_model_files:
        missing_text = ", ".join(missing_model_files)
        raise FileNotFoundError(f"Missing model files: {missing_text}")

    feature_cols = load(FEATURES_PATH)
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
            "<div class='preview-kpi'><div class='preview-kpi-label'>District coverage</div><p class='preview-kpi-value'>160 districts</p></div>"
            "<div class='preview-kpi'><div class='preview-kpi-label'>Best model</div><p class='preview-kpi-value'>Gradient Boosting</p></div>"
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


def clamp_feature_value(
    feature: str,
    value: float,
    feature_bounds: dict[str, tuple[float, float]],
) -> float:
    min_v, max_v = feature_bounds[feature]
    return float(min(max(value, min_v), max_v))


def apply_projection_changes(
    feature_values: dict[str, float],
    annual_feature_changes: dict[str, float],
    feature_bounds: dict[str, tuple[float, float]],
) -> dict[str, float]:
    updated_values: dict[str, float] = {}
    for feature, value in feature_values.items():
        next_value = value + annual_feature_changes.get(feature, 0.0)
        updated_values[feature] = clamp_feature_value(feature, next_value, feature_bounds)
    return updated_values


def build_projection_rows(
    model: object,
    start_values: dict[str, float],
    annual_feature_changes: dict[str, float],
    feature_order: list[str],
    feature_bounds: dict[str, tuple[float, float]],
    years: int,
) -> list[dict[str, object]]:
    """
    Project the current feature row forward by applying annual indicator changes only.

    Predictions remain standard model inferences on the projected feature row. The
    predicted income is never reused as a future model input, which keeps the
    projection compatible with the existing trained models and feature schema.
    """
    rows: list[dict[str, object]] = []
    current_values = {feature: float(value) for feature, value in start_values.items()}

    for step in range(years + 1):
        rows.append(
            {
                "step": step,
                "prediction": predict(model, current_values, feature_order),
                "feature_values": current_values.copy(),
            }
        )
        if step < years:
            # Clamp projected indicators to the observed training range so the
            # projection stays within the model's known feature support.
            current_values = apply_projection_changes(
                current_values,
                annual_feature_changes,
                feature_bounds,
            )

    return rows


def build_projection_comparison_df(
    model: object,
    baseline_values: dict[str, float],
    scenario_values: dict[str, float],
    annual_feature_changes: dict[str, float],
    feature_order: list[str],
    feature_bounds: dict[str, tuple[float, float]],
    years: int,
    base_year: int | None,
) -> pd.DataFrame:
    baseline_rows = build_projection_rows(
        model,
        baseline_values,
        {feature: 0.0 for feature in feature_order},
        feature_order,
        feature_bounds,
        years,
    )
    scenario_rows = build_projection_rows(
        model,
        scenario_values,
        annual_feature_changes,
        feature_order,
        feature_bounds,
        years,
    )

    projection_rows: list[dict[str, object]] = []
    for baseline_row, scenario_row in zip(baseline_rows, scenario_rows):
        step = int(scenario_row["step"])
        if base_year is None:
            year_label = "Current" if step == 0 else f"Year {step}"
        else:
            year_label = str(base_year + step)

        row: dict[str, object] = {
            "Projection step": step,
            "Year": year_label,
            "Baseline projection (RM)": float(baseline_row["prediction"]),
            "Scenario projection (RM)": float(scenario_row["prediction"]),
        }
        row["Δ vs baseline (RM)"] = (
            row["Scenario projection (RM)"] - row["Baseline projection (RM)"]
        )
        for feature in feature_order:
            row[FEATURE_LABELS.get(feature, feature.replace("_", " ").title())] = float(
                scenario_row["feature_values"][feature]
            )
        projection_rows.append(row)

    return pd.DataFrame(projection_rows)


def compute_prediction_ranking(
    df: pd.DataFrame,
    model: object,
    feature_cols: list[str],
    predicted_income: float,
    year: int | None,
) -> tuple[float, int]:
    if year is not None and "year" in df.columns:
        ranking_df = df[df["year"] == year].copy()
    else:
        ranking_df = df.copy()

    ranking_df = ranking_df.dropna(subset=feature_cols)
    if ranking_df.empty:
        return np.nan, 0

    predictions = model.predict(ranking_df[feature_cols])
    predictions_series = pd.Series(predictions).astype(float)
    if predictions_series.empty:
        return np.nan, 0

    percentile = float((predictions_series <= predicted_income).mean() * 100.0)
    return percentile, int(predictions_series.shape[0])


def format_prediction_ranking_insight(percentile: float, sample_size: int) -> str:
    if np.isnan(percentile) or sample_size <= 0:
        return "Ranking insight is unavailable for the current selection."

    if percentile >= 80:
        return f"This district ranks in the top {100 - int(percentile // 1)}% of predicted income among {sample_size} districts."
    if percentile <= 20:
        return f"This district ranks in the bottom {int(percentile // 1)}% of predicted income among {sample_size} districts."
    if percentile < 50:
        return "This district performs below the national median predicted income level."
    return "This district performs above the national median predicted income level."


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
    st.markdown(
        "<p class='page-tagline'>Explore district-level income outlooks with simple simulations "
        "and clear explanations.</p>",
        unsafe_allow_html=True,
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
    model_options = list(models.keys())
    default_model_index = (
        model_options.index(DEFAULT_MODEL_LABEL) if DEFAULT_MODEL_LABEL in model_options else 0
    )

    _sidebar_section_kicker("Setup")
    st.sidebar.header("Explore settings")
    st.sidebar.caption("Choose a district and year baseline, then adjust local conditions.")
    selected_model_name = st.sidebar.selectbox(
        "Prediction model",
        model_options,
        index=default_model_index,
    )
    st.sidebar.caption(MODEL_EXPLANATIONS.get(selected_model_name, ""))
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

    st.sidebar.divider()

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

    _sidebar_section_kicker("Scenario actions", spaced=True)
    if st.sidebar.button("Reset to Baseline", use_container_width=True):
        for feature in feature_cols:
            st.session_state[f"input_{feature}"] = baseline_values[feature]
        st.rerun()

    st.sidebar.divider()

    _sidebar_section_kicker("Local indicators")
    st.sidebar.subheader("District conditions")
    st.sidebar.caption("Adjust values to see how predicted income responds.")
    input_values: dict[str, float] = {}
    for group_name, group_features in FEATURE_GROUPS.items():
        _sidebar_group_labelbar(group_name)
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

    st.sidebar.divider()

    _sidebar_section_kicker("Projection settings")
    st.sidebar.subheader("Scenario projection")
    st.sidebar.caption(
        "Project the current scenario forward by changing the existing indicators each year. "
        "This is a scenario projection, not a causal economic simulation."
    )
    simulation_years = st.sidebar.slider(
        "Simulation length (years)",
        min_value=1,
        max_value=5,
        value=3,
        help="Projects the current scenario forward for up to five additional years.",
    )
    annual_poverty_change = st.sidebar.number_input(
        "Annual poverty change (pp)",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help=(
            "Applied each year to both absolute and relative poverty rates. "
            "Negative values reduce poverty; positive values increase it."
        ),
    )
    annual_infrastructure_change = st.sidebar.number_input(
        "Annual infrastructure change (pp)",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help=(
            "Applied each year to piped water, sanitation, and electricity access. "
            "Positive values improve access; negative values reduce it."
        ),
    )
    annual_gini_change = st.sidebar.number_input(
        "Annual Gini change",
        min_value=-0.1,
        max_value=0.1,
        value=0.0,
        step=0.001,
        format="%.3f",
        help="Applied each year to the Gini coefficient. Negative values reduce inequality.",
    )

    annual_feature_changes = {feature: 0.0 for feature in feature_cols}
    for feature in _POVERTY_PROJECTION_FEATURES:
        if feature in annual_feature_changes:
            annual_feature_changes[feature] = float(annual_poverty_change)
    for feature in _INFRASTRUCTURE_PROJECTION_FEATURES:
        if feature in annual_feature_changes:
            annual_feature_changes[feature] = float(annual_infrastructure_change)
    if "gini" in annual_feature_changes:
        annual_feature_changes["gini"] = float(annual_gini_change)

    selected_model = models[selected_model_name]
    predicted_income = predict(selected_model, input_values, feature_cols)
    baseline_prediction = predict(selected_model, baseline_values, feature_cols)
    actual_income = float(selected_row[TARGET_COL]) if TARGET_COL in selected_row and pd.notna(selected_row[TARGET_COL]) else np.nan

    delta_from_actual = predicted_income - actual_income if not np.isnan(actual_income) else np.nan
    delta_from_baseline = predicted_income - baseline_prediction
    projection_df = build_projection_comparison_df(
        selected_model,
        baseline_values,
        input_values,
        annual_feature_changes,
        feature_cols,
        feature_bounds,
        simulation_years,
        int(selected_year) if selected_year is not None else None,
    )

    st.markdown(
        "<div class='section-note'><strong>Overview:</strong> "
        "Start with the national picture, then move into district-level scenario testing and model comparison."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-kicker'>National context</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-title'>National Socioeconomic Overview</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='national-overview-intro'>These official district-level socioeconomic overview maps from "
        "OpenDOSM provide national context before district-level simulation begins. They help users see broad "
        "patterns across Malaysia, while the dashboard combines that national perspective with interactive local exploration.</p>",
        unsafe_allow_html=True,
    )

    first_row_cols = st.columns(3, gap="medium")
    for col, map_spec in zip(first_row_cols, NATIONAL_MAP_SPECS[:3]):
        with col:
            render_national_map_card(
                map_spec["title"],
                ASSETS_DIR / map_spec["filename"],
                map_spec["caption"],
            )

    second_row_cols = st.columns([0.28, 1, 1, 0.28], gap="medium")
    for col, map_spec in zip(second_row_cols[1:3], NATIONAL_MAP_SPECS[3:]):
        with col:
            render_national_map_card(
                map_spec["title"],
                ASSETS_DIR / map_spec["filename"],
                map_spec["caption"],
            )

    with st.expander("Data & Methodology", expanded=False):
        st.markdown(
            """
            <div class='section-note'>
            <strong>Data sources:</strong> District-level indicators come from official DOSM data
            compiled by OpenDOSM for civic-tech use.
            <br/><br/>
            <strong>Why only 2019 and 2022?</strong> These are the most recent nationwide years
            where the district datasets are complete and comparable.
            <br/><br/>
            <strong>Why about 300 rows?</strong> The dataset includes districts across two years,
            so the row count reflects districts multiplied by 2019 and 2022 entries.
            <br/><br/>
            <strong>How to interpret results:</strong> This dashboard is a scenario simulation tool.
            It helps explore "what if" adjustments to local conditions, not causal forecasts.
            <br/><br/>
            <strong>Model evaluation:</strong> We compare models using cross-validation, a standard
            way to check how well a model performs on held-out data.
            <br/><br/>
            <strong>What features are used?</strong> Poverty, inequality, and infrastructure indicators
            are used as predictive inputs to estimate income outcomes.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='block-gap-chart'></div>", unsafe_allow_html=True)
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
            "<div class='section-subtitle section-subtitle--tight'>"
            "We estimate district-level median household income based on poverty, inequality, and infrastructure indicators."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-title'>How to read the simulations</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle section-subtitle--tight'>"
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

        st.markdown("<div class='block-gap-chart'></div>", unsafe_allow_html=True)
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

        percentile, sample_size = compute_prediction_ranking(
            df,
            selected_model,
            feature_cols,
            predicted_income,
            selected_year,
        )
        ranking_text = format_prediction_ranking_insight(percentile, sample_size)
        st.markdown(
            "<div class='kpi-card'>"
            "<div class='kpi-label'>District ranking insight</div>"
            f"<div class='kpi-value' style='font-size:1rem; line-height:1.5;'>{html.escape(ranking_text)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

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

        st.divider()
        st.subheader("Multi-year scenario projection")
        st.caption(
            "Projected values come from yearly changes to the existing indicators only. "
            "The trained model is reused as-is for each projected year, and predicted income "
            "is not fed back into future inputs."
        )

        projection_fig, projection_ax = plt.subplots(figsize=(9, 4.8))
        projection_fig.patch.set_facecolor("#0f172a")
        projection_ax.plot(
            projection_df["Projection step"],
            projection_df["Baseline projection (RM)"],
            color="#6ea8fe",
            marker="o",
            linewidth=2,
            label="Baseline projection",
        )
        projection_ax.plot(
            projection_df["Projection step"],
            projection_df["Scenario projection (RM)"],
            color="#7fb7b1",
            marker="o",
            linewidth=2,
            label="Scenario projection",
        )
        projection_ax.set_title("Baseline vs scenario projection")
        projection_ax.set_xlabel("Projection year")
        projection_ax.set_ylabel(f"{TARGET_LABEL} (RM)")
        projection_ax.set_xticks(projection_df["Projection step"])
        projection_ax.set_xticklabels(projection_df["Year"])
        projection_ax.legend(
            loc="best",
            facecolor="#0f172a",
            edgecolor="#1f334d",
            labelcolor="#e5edf5",
        )
        style_axes(projection_ax)
        st.pyplot(projection_fig)
        plt.close(projection_fig)

        st.dataframe(
            projection_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Year": st.column_config.TextColumn("Year", width="small"),
                "Baseline projection (RM)": st.column_config.NumberColumn(
                    "Baseline projection (RM)",
                    format="%.2f",
                ),
                "Scenario projection (RM)": st.column_config.NumberColumn(
                    "Scenario projection (RM)",
                    format="%.2f",
                ),
                "Δ vs baseline (RM)": st.column_config.NumberColumn(
                    "Δ vs baseline (RM)",
                    format="%.2f",
                ),
            },
        )

    with tabs[2]:
        st.subheader("Model comparison")
        st.markdown(
            "<p class='model-compare-lede'><strong>Evaluation metrics</strong> are fixed training-phase measurements "
            "(cross-validation on historical data). <strong>Scenario predictions</strong> update when you change "
            "district conditions in the sidebar—they show each model&rsquo;s estimate for your current inputs, "
            "not a new accuracy score.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-title'>What each model means</div>",
            unsafe_allow_html=True,
        )
        st.caption("These descriptions stay fixed so the dashboard remains educational and easy to interpret.")
        explainer_cols = st.columns(2)
        for idx, model_name in enumerate(models.keys()):
            with explainer_cols[idx % 2]:
                render_model_explanation_card(
                    model_name,
                    MODEL_EXPLANATIONS.get(model_name, "No explanation available."),
                )

        st.markdown(
            "<div class='section-title'>Scenario output across models</div>",
            unsafe_allow_html=True,
        )
        st.caption("Predictions refresh when you adjust sidebar values. Each row uses the same scenario inputs.")

        scenario_rows: list[dict[str, object]] = []
        for model_name, model_obj in models.items():
            pred_scenario = predict(model_obj, input_values, feature_cols)
            pred_district_baseline = predict(model_obj, baseline_values, feature_cols)
            scenario_rows.append(
                {
                    "Model": model_name,
                    "Sidebar selection": "Selected" if model_name == selected_model_name else "",
                    "Scenario (RM)": round(pred_scenario, 2),
                    "District baseline (RM)": round(pred_district_baseline, 2),
                    "Δ vs district baseline (RM)": round(pred_scenario - pred_district_baseline, 2),
                }
            )
        scenario_pred_df = pd.DataFrame(scenario_rows)
        st.dataframe(
            scenario_pred_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Sidebar selection": st.column_config.TextColumn("Your model", width="small"),
                "Scenario (RM)": st.column_config.NumberColumn("Scenario (RM)", format="%.2f"),
                "District baseline (RM)": st.column_config.NumberColumn("District baseline (RM)", format="%.2f"),
                "Δ vs district baseline (RM)": st.column_config.NumberColumn(
                    "Δ vs district baseline (RM)", format="%.2f", help="Change from official district values to your scenario."
                ),
            },
        )

        st.divider()
        st.markdown(
            "<p class='model-compare-divider-label'>Fixed training metrics (do not change with sliders)</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-title'>Prediction accuracy on historical data</div>",
            unsafe_allow_html=True,
        )
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
        bar_colors: list[str] = []
        for m in metrics_display["Model"]:
            if m == selected_model_name:
                bar_colors.append("#7fb7b1")
            elif m == best_model_name:
                bar_colors.append("#6ea8fe")
            else:
                bar_colors.append("#5a6d82")
        rmse_ax.bar(
            metrics_display["Model"],
            metrics_display["Prediction error"],
            color=bar_colors,
        )
        rmse_ax.set_title("Prediction error (lower is better)")
        rmse_ax.set_ylabel("Prediction error")
        rmse_ax.set_xlabel("Model")
        style_axes(rmse_ax)
        st.pyplot(rmse_fig)
        plt.close(rmse_fig)

    with tabs[3]:
        st.subheader("District Insight Summary")
        st.markdown(
            "<div class='viz-live-note'>"
            "Charts and insights update automatically as district conditions change in the sidebar."
            "</div>",
            unsafe_allow_html=True,
        )
        income_series = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()
        national_medians = df[feature_cols].median(numeric_only=True)

        insight_feature_order = [
            "poverty_absolute",
            "poverty_relative",
            "gini",
            "electricity",
            "piped_water",
            "sanitation",
        ]
        insight_li_html: list[str] = []
        for fk in insight_feature_order:
            insight_li_html.append(
                format_baseline_insight_list_item(
                    fk,
                    float(baseline_values.get(fk, np.nan)),
                    float(national_medians.get(fk, np.nan)),
                )
            )
        insight_li_html.append(format_scenario_shift_list_item(delta_from_baseline))
        st.markdown(
            '<ul class="interpretation-list">' + "".join(insight_li_html) + "</ul>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-title'>Indicator relationships</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle section-subtitle--tight'>"
            "Each chart highlights the selected district and your scenario so you can compare typical patterns."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='block-gap-chart'></div>", unsafe_allow_html=True)

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
                    insight_html = build_indicator_insight(
                        feature,
                        baseline_values[feature],
                        input_values[feature],
                        median_value,
                        delta_from_baseline,
                    )
                    st.markdown(insight_html, unsafe_allow_html=True)

        st.markdown("<div class='block-gap-chart'></div>", unsafe_allow_html=True)
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
        dist_insights_html = format_distribution_insights_html(actual_percentile, scenario_percentile)
        if dist_insights_html:
            st.markdown(dist_insights_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
