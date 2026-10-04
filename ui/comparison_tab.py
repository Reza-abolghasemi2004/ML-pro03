"""Model comparison page: final test-set metrics, ROC curves, confusion matrix."""

import numpy as np
import seaborn as sns
from PyQt6.QtGui import QBrush, QColor
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

from .widgets import FigureWidget, StatCard

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


class ComparisonPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title = QLabel("Model Comparison")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Final numbers on the held-out test set — the winner gets saved to disk.")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.placeholder = QLabel("Nothing to compare yet — run training first.")
        self.placeholder.setStyleSheet("color:#9ca3af; padding: 30px;")
        layout.addWidget(self.placeholder)

        # best model highlight
        self.best_card = StatCard("Best Model", "—", accent="#16a34a")
        self.best_card.setVisible(False)
        layout.addWidget(self.best_card)

        # comparison table
        table_box = QGroupBox("Test-Set Results")
        table_layout = QVBoxLayout(table_box)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Training Time (s)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        table_layout.addWidget(self.table)
        table_box.setVisible(False)
        layout.addWidget(table_box)

        # charts row
        charts_row = QHBoxLayout()
        self.roc_chart = FigureWidget(height=3.0)
        self.bar_chart = FigureWidget(height=3.0)
        self.cm_chart = FigureWidget(height=3.0)
        for chart, box_title in (
            (self.roc_chart, "ROC Curves (test set)"),
            (self.bar_chart, "Test Metrics Comparison"),
            (self.cm_chart, "Confusion Matrix — Best Model"),
        ):
            box = QGroupBox(box_title)
            box_layout = QVBoxLayout(box)
            box_layout.addWidget(chart)
            charts_row.addWidget(box)
            box.setVisible(False)
        layout.addLayout(charts_row, 1)

        self._table_box = table_box
        self._chart_boxes = [b for b in (self.roc_chart, self.bar_chart, self.cm_chart)]

    # ------------------------------------------------------------------ api
    def show_results(self, result):
        self.placeholder.setVisible(False)
        self.best_card.setVisible(True)
        self._table_box.setVisible(True)
        for box in self._chart_boxes:
            box.parent().setVisible(True)

        names = result.model_names
        results = result.results
        best = result.best_model
        best_metrics = results[best]["test"]

        self.best_card.set_value(best)
        self.best_card.set_sub(
            f"test ROC-AUC {best_metrics['roc_auc']:.3f} · saved to models/churn_model.joblib · "
            f"total training time {result.total_time:.0f}s"
        )

        self._fill_table(names, results, best)
        self._draw_roc(names, results)
        self._draw_bars(names, results)
        self._draw_confusion(best, results)

    def _fill_table(self, names, results, best):
        self.table.setRowCount(len(names))
        highlight = QBrush(QColor("#ecfdf5"))
        for row, name in enumerate(names):
            test = results[name]["test"]
            total_time = results[name]["train_time"] + results[name]["cv_time"] + results[name]["tune_time"]
            values = [
                name,
                f"{test['accuracy']:.3f}",
                f"{test['precision']:.3f}",
                f"{test['recall']:.3f}",
                f"{test['f1']:.3f}",
                f"{test['roc_auc']:.3f}",
                f"{total_time:.1f}",
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if name == best:
                    item.setBackground(highlight)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(row, col, item)

    def _draw_roc(self, names, results):
        self.roc_chart.clear()
        ax = self.roc_chart.add_ax(111)
        for name in names:
            fpr, tpr = results[name]["roc_curve"]
            auc = results[name]["test"]["roc_auc"]
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})",
                    color=config.MODEL_COLORS.get(name))
        ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curves on the test set")
        ax.legend(fontsize=8, loc="lower right")
        self.roc_chart.redraw()

    def _draw_bars(self, names, results):
        self.bar_chart.clear()
        ax = self.bar_chart.add_ax(111)
        x = np.arange(len(METRICS))
        width = 0.8 / max(len(names), 1)
        for i, name in enumerate(names):
            test = results[name]["test"]
            values = [test[m] for m in METRICS]
            offset = (i - (len(names) - 1) / 2) * width
            ax.bar(x + offset, values, width, label=name, color=config.MODEL_COLORS.get(name))
        ax.set_xticks(x, [m.replace("_", "-") for m in METRICS])
        ax.set_ylim(0.5, 1.0)
        ax.set_title("Test-set scores per metric")
        ax.legend(fontsize=8)
        self.bar_chart.redraw()

    def _draw_confusion(self, best, results):
        self.cm_chart.clear()
        ax = self.cm_chart.add_ax(111)
        cm = np.array(results[best]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"],
                    cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(best)
        self.cm_chart.redraw()
