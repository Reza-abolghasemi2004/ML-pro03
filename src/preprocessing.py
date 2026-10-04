"""Data cleaning and the sklearn preprocessing pipeline.

The ColumnTransformer is built once and placed inside every model pipeline,
so scaling/encoding is always fitted on training folds only (no leakage).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def clean_dataset(df):
    """Fix the quirks of the telco-style churn schema.

    Returns a copy; the input frame is never modified.
    """
    df = df.copy()

    # TotalCharges arrives as text and is blank for brand-new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    new_customers = df["TotalCharges"].isna() & (df["tenure"].fillna(0) == 0)
    df.loc[new_customers, "TotalCharges"] = 0.0

    # categorical columns can carry stray whitespace or mixed types
    for col in config.RAW_CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()

    if config.TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[config.TARGET_COLUMN])

    return df


def build_preprocessor(numeric_features, categorical_features):
    numeric_branch = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_branch = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("num", numeric_branch, numeric_features),
        ("cat", categorical_branch, categorical_features),
    ])
