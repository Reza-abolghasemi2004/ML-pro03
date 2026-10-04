"""Overview page: project snapshot and pipeline status checklist."""

from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src import config

from .widgets import StatCard, StatusItem


class OverviewPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("Overview")
        title.setObjectName("PageTitle")
        subtitle = QLabel("A quick snapshot of the dataset, the models and the pipeline state.")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # --- stat cards ------------------------------------------------------
        self.cards = {}
        grid = QGridLayout()
        grid.setSpacing(12)

        card_defs = [
            ("dataset", "Dataset Size", "—", "#111827"),
            ("features", "Number of Features", "—", "#111827"),
            ("target", "Target Variable", config.TARGET_COLUMN, "#4f46e5"),
            ("classes", "Number of Classes", "—", "#4f46e5"),
            ("best_model", "Best Model", "—", "#16a34a"),
            ("accuracy", "Best Accuracy", "—", "#0ea5e9"),
            ("f1", "Best F1", "—", "#0ea5e9"),
            ("roc_auc", "Best ROC-AUC", "—", "#0ea5e9"),
            ("status", "Training Status", "Not started", "#d97706"),
        ]
        for i, (key, name, value, color) in enumerate(card_defs):
            card = StatCard(name, value, color)
            self.cards[key] = card
            grid.addWidget(card, i // 3, i % 3)
        layout.addLayout(grid)

        # --- pipeline checklist ----------------------------------------------
        checklist_box = QGroupBox("Pipeline Status")
        checklist_layout = QVBoxLayout(checklist_box)
        self.status_items = {
            "dataset": StatusItem("Dataset loaded"),
            "preprocessing": StatusItem("Preprocessing ready"),
            "training": StatusItem("Training completed"),
            "best": StatusItem("Best model selected"),
        }
        for item in self.status_items.values():
            checklist_layout.addWidget(item)

        about_box = QGroupBox("About This Project")
        about_layout = QVBoxLayout(about_box)
        about = QLabel(
            "This dashboard trains and compares classification models that predict whether a "
            "subscription customer is likely to churn. Load a dataset on the Dataset page, start "
            "training on the Training page, inspect cross-validation on the Validation page, compare "
            "final test results on the Comparison page, then score individual customers on the "
            "Prediction page."
        )
        about.setWordWrap(True)
        about.setStyleSheet("color: #4b5563;")
        about_layout.addWidget(about)

        row = QHBoxLayout()
        row.addWidget(checklist_box, 1)
        row.addWidget(about_box, 2)
        layout.addLayout(row)
        layout.addStretch(1)

        self._training_running = False

    # ------------------------------------------------------------------ api
    def set_training_running(self, running):
        self._training_running = running
        self.cards["status"].set_value("Running..." if running else self.cards["status"].value_label.text())

    def refresh(self, df=None, result=None):
        """Recompute every card from the current application state."""
        if df is not None:
            self.cards["dataset"].set_value(f"{df.shape[0]:,} rows")
            self.cards["dataset"].set_sub(f"{df.shape[1]} columns")
            n_features = df.shape[1] - 1 if config.TARGET_COLUMN in df.columns else df.shape[1]
            self.cards["features"].set_value(n_features)
            if config.TARGET_COLUMN in df.columns:
                self.cards["classes"].set_value(df[config.TARGET_COLUMN].nunique())
            self.status_items["dataset"].set_state("ok")
            self.status_items["preprocessing"].set_state("ok")

        if self._training_running:
            self.cards["status"].set_value("Running...")
            self.status_items["training"].set_state("pending")
        elif result is not None:
            best = result.results[result.best_model]["test"]
            self.cards["status"].set_value("Completed")
            self.cards["best_model"].set_value(result.best_model)
            self.cards["accuracy"].set_value(f"{best['accuracy']:.3f}")
            self.cards["f1"].set_value(f"{best['f1']:.3f}")
            self.cards["roc_auc"].set_value(f"{best['roc_auc']:.3f}")
            self.cards["best_model"].set_sub(
                f"saved to models/churn_model.joblib · {result.total_time:.0f}s total"
            )
            self.status_items["training"].set_state("ok")
            self.status_items["best"].set_state("ok")
        else:
            if not self._training_running:
                self.cards["status"].set_value("Not started")
