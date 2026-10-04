"""Validation page: cross-validation summary and tuning results."""

import numpy as np
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src import config

from .widgets import FigureWidget

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


class ValidationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title = QLabel("Validation")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "5-fold stratified cross-validation results (training split only) and the "
            "hyperparameters found by grid search."
        )
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.placeholder = QLabel("No cross-validation results yet — run training first.")
        self.placeholder.setStyleSheet("color:#9ca3af; padding: 30px;")
        layout.addWidget(self.placeholder)

        # CV table
        cv_box = QGroupBox("Cross-Validation Results (mean ± std)")
        cv_layout = QVBoxLayout(cv_box)
        self.cv_table = QTableWidget(0, 6)
        self.cv_table.setHorizontalHeaderLabels(
            ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        )
        self.cv_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cv_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.cv_table.setAlternatingRowColors(True)
        cv_layout.addWidget(self.cv_table)
        cv_box.setVisible(False)
        layout.addWidget(cv_box)

        # tuning table
        tuning_box = QGroupBox("Hyperparameter Tuning (GridSearchCV)")
        tuning_layout = QVBoxLayout(tuning_box)
        self.tuning_table = QTableWidget(0, 3)
        self.tuning_table.setHorizontalHeaderLabels(["Model", "Best Parameters", "Best CV Score (ROC-AUC)"])
        self.tuning_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tuning_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tuning_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tuning_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tuning_layout.addWidget(self.tuning_table)
        tuning_box.setVisible(False)
        layout.addWidget(tuning_box)

        # chart
        chart_box = QGroupBox("CV Metrics Comparison")
        chart_layout = QVBoxLayout(chart_box)
        self.chart = FigureWidget(height=3.2)
        chart_layout.addWidget(self.chart)
        chart_box.setVisible(False)
        layout.addWidget(chart_box, 1)

        self._cv_box = cv_box
        self._tuning_box = tuning_box
        self._chart_box = chart_box
        layout.addStretch(1)

    # ------------------------------------------------------------------ api
    def show_results(self, result):
        self.placeholder.setVisible(False)
        self._cv_box.setVisible(True)
        self._tuning_box.setVisible(True)
        self._chart_box.setVisible(True)

        names = result.model_names
        self._fill_cv_table(names, result.results)
        self._fill_tuning_table(names, result.results)
        self._draw_chart(names, result.results)

    def _fill_cv_table(self, names, results):
        self.cv_table.setRowCount(len(names))
        for row, name in enumerate(names):
            cv = results[name]["cv"]
            self.cv_table.setItem(row, 0, QTableWidgetItem(name))
            for col, metric in enumerate(METRICS, start=1):
                text = f"{cv[metric]['mean']:.3f} ± {cv[metric]['std']:.3f}"
                self.cv_table.setItem(row, col, QTableWidgetItem(text))

    def _fill_tuning_table(self, names, results):
        self.tuning_table.setRowCount(len(names))
        for row, name in enumerate(names):
            params = ", ".join(f"{k} = {v}" for k, v in results[name]["best_params"].items())
            self.tuning_table.setItem(row, 0, QTableWidgetItem(name))
            self.tuning_table.setItem(row, 1, QTableWidgetItem(params))
            self.tuning_table.setItem(row, 2, QTableWidgetItem(f"{results[name]['best_cv_score']:.4f}"))

    def _draw_chart(self, names, results):
        self.chart.clear()
        ax = self.chart.add_ax(111)
        x = np.arange(len(METRICS))
        width = 0.8 / max(len(names), 1)
        for i, name in enumerate(names):
            cv = results[name]["cv"]
            means = [cv[metric]["mean"] for metric in METRICS]
            offset = (i - (len(names) - 1) / 2) * width
            ax.bar(x + offset, means, width, label=name, color=config.MODEL_COLORS.get(name))
        ax.set_xticks(x, [m.replace("_", "-") for m in METRICS])
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("Mean cross-validation score per metric")
        ax.legend(fontsize=8)
        self.chart.redraw()
