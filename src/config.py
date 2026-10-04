"""Central place for paths, column names and training defaults."""

from pathlib import Path

# --- paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

DATASET_PATH = DATA_DIR / "dataset.csv"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
LOG_FILE = LOGS_DIR / "app.log"

# --- dataset schema ---------------------------------------------------------
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"      # "Yes" means the customer churned
NEGATIVE_LABEL = "No"

RAW_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]

RAW_CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod",
]

# columns created by src/feature_engineering.py
ENGINEERED_NUMERIC = ["AvgMonthlyCharge", "ServicesCount", "HasSupport", "TenureGroup"]

REQUIRED_COLUMNS = RAW_CATEGORICAL + RAW_NUMERIC + [TARGET_COLUMN]

# --- training defaults -------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TUNING_SCORING = "roc_auc"

MODEL_NAMES = ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"]

MODEL_COLORS = {
    "Logistic Regression": "#4f46e5",
    "K-Nearest Neighbors": "#0ea5e9",
    "Support Vector Machine": "#f59e0b",
}
