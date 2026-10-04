"""Derived features built only from raw customer fields.

Everything here is computed from information available at prediction time,
so nothing leaks from the target.
"""

import numpy as np
import pandas as pd

PROTECTION_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
SERVICE_COLS = PROTECTION_COLS + ["PhoneService", "MultipleLines", "StreamingTV", "StreamingMovies"]


def add_features(df):
    df = df.copy()

    # how many of the available services the customer actually pays for
    df["ServicesCount"] = df[SERVICE_COLS].eq("Yes").sum(axis=1).astype(int)

    # customers with no protection-style add-ons tend to churn more
    df["HasSupport"] = df[PROTECTION_COLS].eq("Yes").any(axis=1).astype(int)

    # average spend per month of relationship; fall back to the current
    # monthly charge for customers with zero tenure
    tenure = df["tenure"].clip(lower=0)
    df["AvgMonthlyCharge"] = np.where(
        tenure > 0,
        df["TotalCharges"] / tenure.replace(0, 1),
        df["MonthlyCharges"],
    )

    # coarse tenure buckets capture the "new customer" risk better than the
    # raw month count for linear models
    df["TenureGroup"] = pd.cut(tenure, bins=[-1, 12, 24, 48, 72], labels=[0, 1, 2, 3]).astype(float)

    return df
