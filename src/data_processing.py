"""
data_processing.py
-------------------
Preprocessing and feature engineering pipeline for the Predictive Modeling
and Risk Scoring for Bank Customer Churn project.

Responsibilities:
- Load and validate the raw dataset
- Drop non-informative identifier fields
- Engineer derived features (balance-to-salary ratio, product density,
  engagement-product interaction, age-tenure interaction)
- One-hot encode categorical variables (Geography, Gender)
- Provide a single row -> model-ready row transform for the Streamlit app

This module has NO dependency on the trained model artifacts, so it can be
reused identically by the training script, the notebook, and the app.
"""

import pandas as pd
import numpy as np

RAW_NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

ENGINEERED_FEATURES = [
    "BalanceToSalaryRatio", "ProductDensity", "EngagementProductInteraction",
    "AgeTenureInteraction",
]

CATEGORICAL_FEATURES = ["Geography", "Gender"]

# Final feature order used by every trained model -- keep stable so saved
# models and the Streamlit app always agree on column order.
MODEL_FEATURES = (
    RAW_NUMERIC_FEATURES
    + ENGINEERED_FEATURES
    + ["Geography_Germany", "Geography_Spain", "Gender_Male"]
)


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and confirm binary/category field consistency."""
    df = df.copy()

    # Handle missing values: numeric -> median, categorical -> mode.
    for col in RAW_NUMERIC_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    binary_cols = ["HasCrCard", "IsActiveMember", "Exited"]
    for col in binary_cols:
        if col in df.columns:
            bad = ~df[col].isin([0, 1])
            if bad.any():
                raise ValueError(f"Column {col} contains values other than 0/1")

    return df


def drop_noninformative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier fields that carry no predictive signal."""
    df = df.copy()
    for col in ["CustomerId", "Surname", "Year"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the derived features requested in the project brief."""
    df = df.copy()

    # Balance-to-Salary ratio: how much of a customer's income their
    # balance represents. +1 avoids division by zero for zero-salary rows.
    df["BalanceToSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

    # Product density indicator: products held per year of tenure. New
    # customers holding many products score high here.
    df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    # Engagement-product interaction: active members with many products
    # behave very differently from inactive members with many products.
    df["EngagementProductInteraction"] = df["IsActiveMember"] * df["NumOfProducts"]

    # Age-tenure interaction: relationship length relative to customer age.
    df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode Geography and Gender.
    drop_first=True -> France and Female become the reference levels,
    matching Geography_Germany / Geography_Spain / Gender_Male as used
    throughout the model and the app.
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

    # Ensure all expected dummy columns exist even if a category is absent
    # from a given batch (e.g. a single-row prediction in the app).
    for col in ["Geography_Germany", "Geography_Spain", "Gender_Male"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].astype(int)

    return df


def build_model_table(path_or_df, drop_target: bool = False) -> pd.DataFrame:
    """
    Full pipeline: load -> validate -> drop IDs -> engineer -> encode.
    Accepts either a CSV path or an already-loaded DataFrame (used by the
    Streamlit app to transform a single what-if row).
    Returns a DataFrame containing MODEL_FEATURES (+ Exited, unless dropped).
    """
    if isinstance(path_or_df, str):
        df = load_raw_data(path_or_df)
    else:
        df = path_or_df.copy()

    df = validate_data(df)
    df = drop_noninformative_features(df)
    df = engineer_features(df)
    df = encode_categoricals(df)

    keep = list(MODEL_FEATURES)
    if "Exited" in df.columns and not drop_target:
        keep = keep + ["Exited"]

    return df[keep]


def single_customer_to_row(customer: dict) -> pd.DataFrame:
    """
    Convert a dict of raw customer inputs (as collected from Streamlit form
    widgets) into a one-row, model-ready DataFrame with the exact same
    feature engineering and encoding as training.
    """
    raw = pd.DataFrame([customer])
    return build_model_table(raw, drop_target=True)


if __name__ == "__main__":
    table = build_model_table("data/European_Bank.csv")
    print("Model-ready table shape:", table.shape)
    print(table.head())
