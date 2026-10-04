"""Prediction page: fill in one customer's details and score them."""

import logging

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src import config
from src.prediction import ModelNotFoundError, PredictionService

from .widgets import risk_color

logger = logging.getLogger(__name__)

# option lists follow the telco churn schema
OPTIONS = {
    "gender": ["Male", "Female"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

FIELD_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


class PredictionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataframe = None
        self.service = PredictionService()
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # ---------------- left: the form ----------------
        left = QVBoxLayout()

        title = QLabel("Prediction")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Enter one customer's profile and let the saved model score it.")
        subtitle.setObjectName("PageSubtitle")
        left.addWidget(title)
        left.addWidget(subtitle)

        form_box = QGroupBox("Customer Information")
        form = QFormLayout(form_box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.combos = {}
        for field in FIELD_ORDER:
            combo = QComboBox()
            combo.addItems(OPTIONS[field])
            self.combos[field] = combo
            form.addRow(field, combo)

        self.tenure_spin = QSpinBox()
        self.tenure_spin.setRange(0, 72)
        self.tenure_spin.setValue(12)
        form.addRow("tenure (months)", self.tenure_spin)

        self.monthly_spin = QDoubleSpinBox()
        self.monthly_spin.setRange(15.0, 130.0)
        self.monthly_spin.setDecimals(2)
        self.monthly_spin.setValue(65.0)
        form.addRow("MonthlyCharges", self.monthly_spin)

        self.total_spin = QDoubleSpinBox()
        self.total_spin.setRange(0.0, 9000.0)
        self.total_spin.setDecimals(2)
        self.total_spin.setValue(780.0)
        form.addRow("TotalCharges", self.total_spin)

        left.addWidget(form_box)

        buttons = QHBoxLayout()
        example_btn = QPushButton("Fill Example")
        example_btn.clicked.connect(self.fill_example)
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setObjectName("PrimaryButton")
        self.predict_btn.clicked.connect(self.predict)
        buttons.addWidget(example_btn)
        buttons.addWidget(self.predict_btn)
        left.addLayout(buttons)
        left.addStretch(1)
        layout.addLayout(left, 3)

        # ---------------- right: the result ----------------
        right = QVBoxLayout()
        result_box = QGroupBox("Result")
        result_layout = QVBoxLayout(result_box)
        result_layout.setSpacing(12)

        self.model_info = QLabel("Saved model: not loaded yet")
        self.model_info.setStyleSheet("color:#6b7280; font-size: 11px;")
        self.model_info.setWordWrap(True)
        result_layout.addWidget(self.model_info)

        self.prediction_label = QLabel("—")
        self.prediction_label.setStyleSheet("font-size: 26px; font-weight: 800; color:#9ca3af;")
        result_layout.addWidget(self.prediction_label)

        prob_row = QHBoxLayout()
        prob_row.addWidget(QLabel("Churn probability:"))
        self.probability_label = QLabel("—")
        self.probability_label.setStyleSheet("font-size: 22px; font-weight: 700; color:#111827;")
        prob_row.addWidget(self.probability_label)
        prob_row.addStretch(1)
        result_layout.addLayout(prob_row)

        risk_row = QHBoxLayout()
        risk_row.addWidget(QLabel("Risk level:"))
        self.risk_label = QLabel("—")
        self.risk_label.setStyleSheet("font-size: 15px; font-weight: 700; color:#9ca3af;")
        risk_row.addWidget(self.risk_label)
        risk_row.addStretch(1)
        result_layout.addLayout(risk_row)

        self.explanation = QLabel(
            "The prediction shows how confident the model is that this customer will cancel "
            "the subscription. HIGH risk usually means a retention action (discount, support "
            "call, contract upgrade) is worth trying; LOW risk means the customer looks stable."
        )
        self.explanation.setWordWrap(True)
        self.explanation.setStyleSheet("color:#4b5563;")
        result_layout.addWidget(self.explanation)

        right.addWidget(result_box)
        right.addStretch(1)
        layout.addLayout(right, 2)

        self._apply_available_model_state()

    # ------------------------------------------------------------------ api
    def set_dataframe(self, df):
        self.dataframe = df

    def refresh_model_state(self):
        """Called after training finishes so the page notices the new model."""
        self.service.bundle = None
        self._apply_available_model_state()

    # -------------------------------------------------------------- helpers
    def _apply_available_model_state(self):
        if self.service.is_available():
            self.model_info.setText(f"Saved model found at {self.service.path.name} "
                                    f"(models folder). It will be loaded on the first prediction.")
            self.predict_btn.setEnabled(True)
        else:
            self.model_info.setText("No saved model yet — run training first or place a "
                                    "churn_model.joblib file in the models folder.")
            self.predict_btn.setEnabled(False)

    def fill_example(self):
        """Copy a random row from the loaded dataset into the form."""
        if self.dataframe is None:
            QMessageBox.information(self, "No Dataset", "Load a dataset first to pull an example row.")
            return
        row = self.dataframe.sample(1, random_state=np.random.randint(0, 10_000)).iloc[0]
        for field, combo in self.combos.items():
            value = str(row[field]).strip()
            index = combo.findText(value)
            combo.setCurrentIndex(max(index, 0))
        self.tenure_spin.setValue(int(row.get("tenure", 0)))
        self.monthly_spin.setValue(float(row.get("MonthlyCharges", 65.0)))
        total = row.get("TotalCharges", 0.0)
        try:
            self.total_spin.setValue(float(total))
        except (TypeError, ValueError):
            self.total_spin.setValue(0.0)

    def _collect_record(self):
        record = {field: combo.currentText() for field, combo in self.combos.items()}
        # SeniorCitizen is stored as 0/1 in the raw schema
        record["SeniorCitizen"] = "1" if record["SeniorCitizen"] == "Yes" else "0"
        record["tenure"] = int(self.tenure_spin.value())
        record["MonthlyCharges"] = float(self.monthly_spin.value())
        record["TotalCharges"] = float(self.total_spin.value())
        return record

    def predict(self):
        record = self._collect_record()

        # a few consistency rules the model was not trained to handle
        if record["PhoneService"] == "No" and record["MultipleLines"] == "Yes":
            QMessageBox.warning(self, "Invalid Input", "MultipleLines can only be 'Yes' when PhoneService is active.")
            return
        if record["InternetService"] == "No":
            for col in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                        "StreamingTV", "StreamingMovies"):
                if record[col] == "Yes":
                    QMessageBox.warning(
                        self, "Invalid Input",
                        f"{col} can only be 'Yes' when InternetService is DSL or Fiber optic.")
                    return

        try:
            if self.service.bundle is None:
                self.service.load()
                self.model_info.setText(
                    f"Loaded model: {self.service.bundle['model_name']} "
                    f"(trained {self.service.bundle['created_at'][:10]}, "
                    f"test ROC-AUC {self.service.bundle['test_metrics']['roc_auc']:.3f})."
                )
            outcome = self.service.predict(record)
        except ModelNotFoundError as exc:
            QMessageBox.warning(self, "Model Not Found", str(exc))
            self._apply_available_model_state()
            return
        except Exception as exc:  # noqa: BLE001 - any scoring error goes to the UI
            logger.exception("Prediction failed")
            QMessageBox.critical(self, "Prediction Error", f"Prediction failed:\n\n{exc}")
            return

        probability = outcome["churn_probability"]
        will_churn = outcome["predicted_label"] == "Churn"
        self.prediction_label.setText("LIKELY TO CHURN" if will_churn else "LIKELY TO STAY")
        self.prediction_label.setStyleSheet(
            f"font-size: 26px; font-weight: 800; color:{'#dc2626' if will_churn else '#16a34a'};"
        )
        self.probability_label.setText(f"{probability * 100:.0f}%")
        self.risk_label.setText(outcome["risk"])
        self.risk_label.setStyleSheet(f"font-size: 15px; font-weight: 700; color:{risk_color(outcome['risk'])};")

        if outcome["risk"] == "HIGH":
            detail = ("The model sees a strong churn pattern here — short-tenure, month-to-month "
                      "contracts with few add-ons look like this. A retention offer is worth trying.")
        elif outcome["risk"] == "MEDIUM":
            detail = ("The signal is mixed. The model is not certain; monitoring or a light-touch "
                      "offer would be reasonable.")
        else:
            detail = "This profile looks stable — nothing here resembles the typical churner pattern."
        self.explanation.setText(
            f"With {probability * 100:.0f}% estimated churn probability, this customer is classified "
            f"as {outcome['risk']} risk. {detail}"
        )
        logger.info("Prediction: %s (p=%.3f, risk=%s)", outcome["predicted_label"],
                    probability, outcome["risk"])
