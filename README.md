# Financial Mobility Simulation Dashboard

An end-to-end machine learning application for district-level income prediction and socioeconomic scenario simulation using official Malaysian data.

## Overview

Financial Mobility Simulation Dashboard is an interactive web application that predicts district-level median household income in Malaysia using machine learning and official socioeconomic indicators from the Department of Statistics Malaysia (DOSM).

Rather than functioning as a standalone prediction tool, the dashboard is built as an educational scenario-exploration platform — allowing users to investigate how changes in poverty rate, infrastructure development, and income inequality (Gini coefficient) may influence projected household income under different socioeconomic conditions.

The project combines data engineering, machine learning, and interactive visualization into a full-stack data application, covering the complete pipeline from data collection to cloud deployment.

## Features

- District-level income prediction
- Interactive scenario simulation (1–5 year projections)
- Multiple ML model comparison (Linear Regression, Ridge, Random Forest, Gradient Boosting)
- Model performance evaluation (RMSE, MAE, R², 5-fold cross-validation)
- National socioeconomic overview
- Dynamic, interactive visualizations
- User authentication via Supabase
- Scenario saving, reloading, and comparison
- Guest mode support (no account required)
- Methodology and model explanation pages

## Dashboard Modules

**Landing Page** — Introduces the project objectives and dashboard capabilities.

**National Overview** — Explore historical socioeconomic indicators across Malaysian districts.

**Projection Dashboard** — Generate district-level income projections by adjusting poverty rate, infrastructure development, and Gini coefficient, projecting 1–5 years from the latest available dataset.

**Scenario Presets** — Built-in presets (Baseline Stability, Infrastructure Investment, Poverty Reduction Initiative, Economic Downturn) for non-technical users to explore illustrative scenarios.

**Visual Analytics** — Predicted vs. observed income, distribution plots, scatter plots, feature comparisons, and dynamic interpretation summaries.

**Model Information** — Compares Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting. Gradient Boosting was selected as the final model, achieving an R² of 0.77.

**My Scenarios** — Authenticated users can save, reload, compare, and delete simulation scenarios.

## Machine Learning Pipeline

```
Official DOSM Datasets
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Cross Validation
        │
        ▼
Model Evaluation
        │
        ▼
Scenario Simulation
        │
        ▼
Interactive Dashboard
```

## Dataset

Publicly available socioeconomic data from Malaysia's Department of Statistics (DOSM), covering the 2019 and 2022 reporting years. Indicators include:

- Median household income
- Poverty incidence
- Gini coefficient
- Electricity access
- Piped water access
- Sanitation access

The final analytical dataset contains **311 district-level observations**.

## Technology Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Web Application | Streamlit |
| Database & Authentication | Supabase |
| Version Control | Git, GitHub |

## Project Structure

```
financial-mobility-simulation-dashboard/
│
├── app.py
├── build_dataset.py
├── train_models.py
├── model_config.py
├── generate_metrics.py
│
├── data/
├── models/
├── notebooks/
├── assets/
├── pages/
├── utils/
└── .streamlit/
```

## Installation

```bash
git clone https://github.com/adrielhakiiem/financial-mobility-simulation-dashboard.git
cd financial-mobility-simulation-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Example Workflow

1. Select a district.
2. Choose a projection period (1–5 years).
3. Apply a scenario preset or customize socioeconomic indicators.
4. Generate predictions.
5. Explore interactive visualizations.
6. Save scenarios for comparison.

## Project Highlights

- End-to-end data science application (data collection through deployment)
- Built on official Malaysian government socioeconomic data
- Interactive, scenario-based simulation engine
- Educational machine learning platform
- Cloud-deployed with user authentication
- Reproducible ML workflow with cross-validated model comparison

## Future Improvements

- Incorporate newly released DOSM datasets
- Expand coverage with additional socioeconomic indicators
- Support international datasets
- Improve mobile responsiveness
- Integrate explainable AI techniques (e.g., SHAP)
- Deploy as a scalable cloud service with API support

## Live Demo

*(add your deployed Streamlit URL)*

## Repository

https://github.com/adrielhakiiem/financial-mobility-simulation-dashboard
