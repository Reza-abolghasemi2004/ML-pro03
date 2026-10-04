"""Training page: start the pipeline, watch progress, folds and live charts."""

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

from src import config

from .widgets import FigureWidget, StatCard
from .workers import TrainingWorker

logger = logging.getLogger(__name__)

METRIC_CARDS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


class TrainingPage(QWidget):
    finished_ok = pyqtSignal(object)   # TrainingResult
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataframe = None
        self.dataset_path = ""
        self.worker = None
        self.fold_series = {}          # model -> [(fold, roc_auc)]
        self.cv_scores = {}            # model -> mean CV roc_auc
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel("Training")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Baseline fit, 5-fold stratified cross-validation and grid search run "
            "in a background thread — the UI stays responsive."
        )
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # controls row
        controls = QHBoxLayout()
        self.model_checks = {}
        for name in config.MODEL_NAMES:
            box = QCheckBox(name)
            box.setChecked(True)
            self.model_checks[name] = box
            controls.addWidget(box)
        controls.addStretch(1)
        self.start_button = QPushButton("Start Training")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self.start_training)
        controls.addWidget(self.start_button)
        layout.addLayout(controls)

        # progress row
        progress_box = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_box)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        info_row = QHBoxLayout()
        self.model_label = QLabel("Current Model: —")
        self.stage_label = QLabel("Current Stage: idle")
        self.fold_label = QLabel("Current Fold: —")
        for label in (self.model_label, self.stage_label, self.fold_label):
            label.setStyleSheet("color: #374151; font-weight: 600;")
            info_row.addWidget(label)
        info_row.addStretch(1)
        progress_layout.addLayout(info_row)
        layout.addWidget(progress_box)

        # live metric cards (values from the latest fold / model)
        cards_row = QHBoxLayout()
        self.metric_cards = {}
        for metric in METRIC_CARDS:
            card = StatCard(metric.replace("_", "-"), "—", accent="#0ea5e9")
            self.metric_cards[metric] = card
            cards_row.addWidget(card)
        layout.addLayout(cards_row)

        # bottom: log window + live charts
        splitter = QSplitter(Qt.Orientation.Horizontal)

        log_box = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("LogView")
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)
        splitter.addWidget(log_box)

        charts_box = QGroupBox("Live Charts")
        charts_layout = QVBoxLayout(charts_box)
        self.fold_chart = FigureWidget(height=2.4)
        self.summary_chart = FigureWidget(height=2.4)
        charts_layout.addWidget(self.fold_chart)
        charts_layout.addWidget(self.summary_chart)
        splitter.addWidget(charts_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------------ api
    def set_dataframe(self, df, path=""):
        self.dataframe = df
        self.dataset_path = path

    def append_log(self, line):
        self.log_view.appendPlainText(line)

    # ------------------------------------------------------------- training
    def start_training(self):
        if self.dataframe is None:
            QMessageBox.warning(self, "No Dataset", "Load a dataset on the Dataset page first.")
            return

        selected = [name for name, box in self.model_checks.items() if box.isChecked()]
        if not selected:
            QMessageBox.warning(self, "No Models", "Select at least one model to train.")
            return

        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "Training is already running.")
            return

        self._reset_run()
        self.worker = TrainingWorker(self.dataframe, selected, self.dataset_path, parent=self)
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self._on_status)
        self.worker.log.connect(self.append_log)
        self.worker.current_model.connect(self._on_model)
        self.worker.fold_done.connect(self._on_fold)
        self.worker.model_done.connect(self._on_model_done)
        self.worker.succeeded.connect(self._on_succeeded)
        self.worker.failed.connect(self._on_failed)

        self.start_button.setEnabled(False)
        self.worker.start()

    def _reset_run(self):
        self.fold_series = {}
        self.cv_scores = {}
        self.progress_bar.setValue(0)
        self.log_view.clear()
        self.fold_chart.clear()
        self.summary_chart.clear()
        self.fold_chart.redraw()
        self.summary_chart.redraw()
        for card in self.metric_cards.values():
            card.set_value("—")
        self.model_label.setText("Current Model: —")
        self.stage_label.setText("Current Stage: starting")
        self.fold_label.setText("Current Fold: —")

    # -------------------------------------------------------------- signals
    def _on_progress(self, value):
        self.progress_bar.setValue(value)

    def _on_status(self, text):
        self.stage_label.setText(f"Current Stage: {text}")

    def _on_model(self, name):
        self.model_label.setText(f"Current Model: {name}")

    def _on_fold(self, model, fold_i, n_folds, metrics):
        self.fold_label.setText(f"Current Fold: {fold_i} / {n_folds}")
        for metric in METRIC_CARDS:
            self.metric_cards[metric].set_value(f"{metrics[metric]:.3f}")
        self.fold_series.setdefault(model, []).append((fold_i, metrics["roc_auc"]))
        self._draw_fold_chart()

    def _on_model_done(self, model, result):
        self.cv_scores[model] = result["cv"]["roc_auc"]["mean"]
        for metric in METRIC_CARDS:
            self.metric_cards[metric].set_value(f"{result['cv'][metric]['mean']:.3f}")
            self.metric_cards[metric].set_sub("CV mean")
        self._draw_summary_chart()

    def _on_succeeded(self, result):
        self.start_button.setEnabled(True)
        self.stage_label.setText("Current Stage: completed")
        self.append_log(f"Done. Best model: {result.best_model} "
                        f"(test ROC-AUC {result.results[result.best_model]['test']['roc_auc']:.3f}).")
        self.finished_ok.emit(result)

    def _on_failed(self, message):
        self.start_button.setEnabled(True)
        self.stage_label.setText("Current Stage: failed")
        logger.error("Training failed: %s", message)
        QMessageBox.critical(self, "Training Error", f"Training failed:\n\n{message}")
        self.failed.emit(message)

    # --------------------------------------------------------------- charts
    def _draw_fold_chart(self):
        """Validation ROC-AUC per fold, one line per model."""
        self.fold_chart.clear()
        ax = self.fold_chart.add_ax(111)
        for model, points in self.fold_series.items():
            folds, scores = zip(*points)
            ax.plot(folds, scores, marker="o", label=model, color=config.MODEL_COLORS.get(model))
        ax.set_title("Validation ROC-AUC per fold (live)")
        ax.set_xlabel("Fold")
        ax.set_ylabel("ROC-AUC")
        ax.legend(fontsize=8)
        self.fold_chart.redraw()

    def _draw_summary_chart(self):
        """Mean CV ROC-AUC for every model finished so far."""
        self.summary_chart.clear()
        ax = self.summary_chart.add_ax(111)
        models = list(self.cv_scores)
        values = [self.cv_scores[m] for m in models]
        colors = [config.MODEL_COLORS.get(m, "#4f46e5") for m in models]
        ax.bar(models, values, color=colors)
        ax.set_ylim(0.5, 1.0)
        ax.set_title("Mean CV ROC-AUC so far")
        ax.set_ylabel("ROC-AUC")
        for tick in ax.get_xticklabels():
            tick.set_fontsize(8)
        self.summary_chart.redraw()
