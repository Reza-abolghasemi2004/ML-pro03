"""Headless end-to-end check: load -> train all models -> predict one customer.

Handy when you change something in src/ and want to make sure the whole
pipeline still works without opening the GUI:

    python scripts/smoke_test.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_dataset, dataset_summary
from src.prediction import PredictionService
from src.training import run_full_training


def main():
    df = load_dataset("data/dataset.csv")
    summary = dataset_summary(df)
    print(f"dataset: {summary['rows']} rows, {summary['columns']} columns, "
          f"{summary['missing_cells']} missing cells")

    cb = {
        "log": lambda m: print("   ", m),
        "model": lambda m: print(">> model:", m),
        "model_done": lambda name, res: print(f">> done {name}: test ROC-AUC {res['test']['roc_auc']:.3f}"),
    }

    start = time.time()
    result = run_full_training(df, cb=cb, dataset_path="data/dataset.csv")

    print("\n=== results ===")
    print("best model:", result.best_model)
    for name, res in result.results.items():
        t = res["test"]
        print(f"{name:25s} acc={t['accuracy']:.3f} prec={t['precision']:.3f} rec={t['recall']:.3f} "
              f"f1={t['f1']:.3f} auc={t['roc_auc']:.3f} params={res['best_params']}")
    print(f"total time: {time.time() - start:.1f}s")

    svc = PredictionService()
    svc.load()
    risky = {"gender": "Male", "SeniorCitizen": "0", "Partner": "No", "Dependents": "No",
             "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
             "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
             "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
             "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
             "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0}
    print("\nrisky customer prediction:", svc.predict(risky))

    row = df.iloc[5].to_dict()
    row.pop("customerID", None)
    print(f"sample row prediction (actual churn: {row['Churn']}):", svc.predict(row))


if __name__ == "__main__":
    main()
