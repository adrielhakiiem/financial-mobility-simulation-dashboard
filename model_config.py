from dataclasses import dataclass
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


DATA_PATH = Path("data/processed/final_dataset.csv")
MODELS_DIR = Path("models")
FEATURES_PATH = MODELS_DIR / "features.pkl"
METRICS_PATH = MODELS_DIR / "metrics.csv"
TARGET_COL = "income_median"
DEFAULT_MODEL_LABEL = "Random Forest"

FEATURE_COLS = [
    "poverty_absolute",
    "poverty_relative",
    "gini",
    "piped_water",
    "sanitation",
    "electricity",
]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    filename: str
    description: str


MODEL_SPECS = (
    ModelSpec(
        key="linear",
        label="Linear Regression",
        filename="linear_model.pkl",
        description=(
            "Draws one overall straight-line relationship between the district indicators and income. "
            "It is the easiest model to explain, but it can miss more complex patterns."
        ),
    ),
    ModelSpec(
        key="ridge",
        label="Ridge Regression",
        filename="ridge_model.pkl",
        description=(
            "Works like linear regression but gently shrinks unstable estimates when the indicators overlap. "
            "That usually makes it a bit more stable on small datasets."
        ),
    ),
    ModelSpec(
        key="rf",
        label="Random Forest",
        filename="rf_model.pkl",
        description=(
            "Combines many decision trees and averages their answers. "
            "It can capture non-linear patterns better than linear models, but the final prediction is less direct to explain."
        ),
    ),
    ModelSpec(
        key="gb",
        label="Gradient Boosting",
        filename="gb_model.pkl",
        description=(
            "Builds many small trees step by step, with each new step focusing on earlier mistakes. "
            "It often finds nuanced patterns in structured data, but it needs care because small datasets can make it overly sensitive."
        ),
    ),
)


def build_models() -> dict[str, object]:
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(),
        "rf": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "gb": GradientBoostingRegressor(random_state=42),
    }


def model_labels_by_key() -> dict[str, str]:
    return {spec.key: spec.label for spec in MODEL_SPECS}


def model_files_by_label() -> dict[str, str]:
    return {spec.label: spec.filename for spec in MODEL_SPECS}


def model_descriptions_by_label() -> dict[str, str]:
    return {spec.label: spec.description for spec in MODEL_SPECS}
