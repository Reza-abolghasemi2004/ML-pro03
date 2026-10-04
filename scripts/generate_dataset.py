"""Generate a synthetic customer-churn dataset in the classic telco schema.

We couldn't ship the original Kaggle dataset with the project, so this
script fabricates data with the same columns and the same kind of churn
signal (month-to-month contracts, fiber optic, electronic checks and short
tenure all push churn up). Run it from the project root:

    python scripts/generate_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT = BASE_DIR / "data" / "dataset.csv"

N = 7043
SEED = 42


def main():
    rng = np.random.default_rng(SEED)

    gender = rng.choice(["Male", "Female"], size=N)
    senior = rng.binomial(1, 0.16, N)
    partner = np.where(rng.random(N) < 0.48, "Yes", "No")
    dependents = np.where(rng.random(N) < 0.30, "Yes", "No")

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=N, p=[0.55, 0.21, 0.24]
    )

    # tenure depends on the contract type: month-to-month customers include a
    # lot of very new accounts, two-year customers skew long-tenured
    tenure = np.zeros(N, dtype=int)
    m2m = contract == "Month-to-month"
    one = contract == "One year"
    two = contract == "Two year"
    tenure[m2m] = rng.integers(0, 50, size=m2m.sum())
    tenure[one] = rng.integers(1, 61, size=one.sum())
    tenure[two] = rng.integers(12, 73, size=two.sum())
    # some month-to-month customers are still loyal after years
    loyal = m2m & (rng.random(N) < 0.18)
    tenure[loyal] = rng.integers(30, 73, size=loyal.sum())

    phone = rng.binomial(1, 0.90, N).astype(bool)
    multiple = np.where(phone, np.where(rng.random(N) < 0.42, "Yes", "No"), "No phone service")

    internet = rng.choice(["DSL", "Fiber optic", "No"], size=N, p=[0.34, 0.45, 0.21])
    has_internet = internet != "No"
    fiber = internet == "Fiber optic"
    dsl = internet == "DSL"

    def addon(p_fiber, p_dsl):
        prob = np.where(fiber, p_fiber, np.where(dsl, p_dsl, 0.0))
        return np.where(has_internet, np.where(rng.random(N) < prob, "Yes", "No"), "No internet service")

    online_security = addon(0.15, 0.35)
    online_backup = addon(0.30, 0.40)
    device_protection = addon(0.25, 0.35)
    tech_support = addon(0.15, 0.30)
    streaming_tv = addon(0.55, 0.35)
    streaming_movies = addon(0.55, 0.35)

    paperless = np.where(rng.random(N) < 0.60, "Yes", "No")
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=N, p=[0.22, 0.22, 0.22, 0.34],
    )

    # monthly charges depend mostly on the internet service and add-ons
    base = np.where(
        internet == "No", rng.uniform(19, 30, N),
        np.where(dsl, rng.uniform(40, 70, N), rng.uniform(68, 108, N)),
    )
    extras = sum(
        col == "Yes" for col in
        (online_security, online_backup, device_protection, tech_support,
         streaming_tv, streaming_movies, multiple)
    )
    monthly = np.clip(base + extras * rng.uniform(3.5, 6.0, N), 18.5, 119.0).round(2)

    total = (tenure * monthly * rng.uniform(0.96, 1.04, N)).round(2)
    total_series = total.astype(object)
    total_series[tenure == 0] = ""  # blank for brand-new customers, like the real data

    # churn probability: a logistic function of the risky-looking fields
    z = (
        -2.80
        + 1.35 * (contract == "Month-to-month")
        - 0.25 * (contract == "One year")
        - 1.05 * (contract == "Two year")
        + 0.75 * fiber
        + 0.10 * dsl
        - 0.85 * (internet == "No")
        + 0.45 * (payment == "Electronic check")
        + 0.55 * ((online_security == "No") & has_internet)
        + 0.35 * ((tech_support == "No") & has_internet)
        + 0.25 * ((device_protection == "No") & has_internet)
        - 0.022 * tenure
        + 0.018 * (monthly - 55)
        + 0.30 * (partner == "No")
        + 0.15 * senior
    )
    churn_probability = 1.0 / (1.0 + np.exp(-z))
    churn = np.where(rng.random(N) < churn_probability, "Yes", "No")

    customer_ids = [
        f"{rng.integers(1000, 9999)}-{rng.integers(10000, 99999)}" for _ in range(N)
    ]

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": np.where(phone, "Yes", "No"),
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total_series,
        "Churn": churn,
    })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"Wrote {OUTPUT} — {len(df)} rows, churn rate {churn_rate:.1%}")


if __name__ == "__main__":
    main()
