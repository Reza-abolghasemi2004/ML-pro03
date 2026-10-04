"""Loading and sanity-checking the churn dataset."""

import logging
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


class DatasetError(ValueError):
    """Raised when a file cannot be used as the churn dataset."""


def load_dataset(path):
    """Read a CSV file and check it matches the expected churn schema."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DatasetError(f"Could not parse '{path.name}' as a CSV file: {exc}") from exc

    if df.empty:
        raise DatasetError("The CSV file is empty.")

    validate_columns(df)
    logger.info("Loaded %s (%d rows, %d columns).", path.name, df.shape[0], df.shape[1])
    return df


def validate_columns(df):
    """The pipeline relies on fixed column names, so fail early if some are absent."""
    missing = [col for col in config.REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetError("Missing required column(s): " + ", ".join(missing))


def dataset_summary(df):
    """Basic health numbers shown on the Dataset page."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
