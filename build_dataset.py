from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "final_dataset.csv"
TARGET_YEARS = [2019, 2022]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to support reliable joins."""
    df.columns = df.columns.str.strip().str.lower()
    return df


def _add_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date column and extract year, mirroring notebook logic."""
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df


def load_data(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load all raw datasets and apply common preprocessing steps."""
    dataframes = {
        "income": pd.read_csv(raw_dir / "income_district.csv"),
        "poverty": pd.read_csv(raw_dir / "poverty_district.csv"),
        "gini": pd.read_csv(raw_dir / "gini_district.csv"),
        "amenities": pd.read_csv(raw_dir / "amenities.csv"),
    }

    for key, df in dataframes.items():
        dataframes[key] = _add_year_column(_clean_columns(df))

    return dataframes


def filter_data(dataframes: dict[str, pd.DataFrame], years: list[int] = TARGET_YEARS) -> dict[str, pd.DataFrame]:
    """Filter each dataframe to the target years."""
    filtered = {
        key: df[df["year"].isin(years)].copy()
        for key, df in dataframes.items()
    }

    print(f"Income filtered shape: {filtered['income'].shape}")
    print(f"Poverty filtered shape: {filtered['poverty'].shape}")
    print(f"Gini filtered shape: {filtered['gini'].shape}")
    print(f"Amenities filtered shape: {filtered['amenities'].shape}")

    return filtered


def merge_data(filtered_dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge filtered dataframes using left joins with income as base."""
    df_income_filtered = filtered_dataframes["income"]
    df_poverty_filtered = filtered_dataframes["poverty"]
    df_gini_filtered = filtered_dataframes["gini"]
    df_amenities_filtered = filtered_dataframes["amenities"]

    df_merged = df_income_filtered.copy()

    df_merged = df_merged.merge(
        df_poverty_filtered[["state", "district", "year", "poverty_absolute", "poverty_relative"]],
        on=["state", "district", "year"],
        how="left",
    )

    df_merged = df_merged.merge(
        df_gini_filtered[["state", "district", "year", "gini"]],
        on=["state", "district", "year"],
        how="left",
    )

    df_merged = df_merged.merge(
        df_amenities_filtered[["state", "district", "year", "piped_water", "sanitation", "electricity"]],
        on=["state", "district", "year"],
        how="left",
    )

    return df_merged


def save_data(df: pd.DataFrame, output_file: Path = OUTPUT_FILE) -> None:
    """Save merged dataframe to the processed data directory."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


def main() -> None:
    dataframes = load_data()
    filtered_dataframes = filter_data(dataframes)
    df_merged = merge_data(filtered_dataframes)
    save_data(df_merged)

    print(f"Merged DataFrame shape: {df_merged.shape}")
    print(f"\nColumn names:\n{df_merged.columns.tolist()}")


if __name__ == "__main__":
    main()
