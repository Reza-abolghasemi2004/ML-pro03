"""Main window: sidebar navigation + one stacked page per dashboard section."""

import logging

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src import config

from .comparison_tab import ComparisonPage
from .dashboard import OverviewPage
from .dataset_tab import DatasetPage
from .log_bus import attach_bus_handler
from .logs_tab import LogsPage
from .prediction_tab import PredictionPage
from .training_tab import TrainingPage
from .validation_tab import ValidationPage

logger = logging.getLogger(__name__)

PAGES = [
    ("Overview", "\u25C9  Overview"),
    ("Dataset", "\u25A4  Dataset"),
    ("Training", "\u25B6  Training"),
    ("Validation", "\u2713  Validation"),
    ("Comparison", "\u21C4  Comparison"),
    ("Prediction", "\u2726  Prediction"),
    ("Logs", "\u2261  Logs"),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        attach_bus_handler()

        self.setWindowTitle("Customer Churn Prediction — ML Dashboard")
        self.resize(1320, 840)

        self.training_result = None
        self.dataset_path = ""

        self._build_ui()
        self._connect_pages()

        # auto-load the bundled dataset shortly after the window appears
        if config.DATASET_PATH.exists():
            QTimer.singleShot(150, lambda: self.dataset_page.load_from_path(config.DATASET_PATH))

    # ------------------------------------------------------------------ ui
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ----- sidebar -----
        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(215)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(14, 18, 14, 14)
        side_layout.setSpacing(6)

        app_title = QLabel("Churn ML")
        app_title.setObjectName("AppTitle")
        app_subtitle = QLabel("Customer Churn Prediction")
        app_subtitle.setObjectName("AppSubtitle")
        side_layout.addWidget(app_title)
        side_layout.addWidget(app_subtitle)
        side_layout.addSpacing(18)

        self.nav_buttons = []
        for key, label in PAGES:
            button = QPushButton(label)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked, k=key: self._switch_page(k))
            side_layout.addWidget(button)
            self.nav_buttons.append((key, button))
        self.nav_buttons[0][1].setChecked(True)

        side_layout.addStretch(1)
        footer = QLabel("ML Pipeline · PyQt6 · scikit-learn\nv1.0")
        footer.setObjectName("SidebarFooter")
        side_layout.addWidget(footer)

        # ----- pages -----
        self.overview_page = OverviewPage()
        self.dataset_page = DatasetPage()
        self.training_page = TrainingPage()
        self.validation_page = ValidationPage()
        self.comparison_page = ComparisonPage()
        self.prediction_page = PredictionPage()
        self.logs_page = LogsPage()

        self.stack = QStackedWidget()
        for page in (self.overview_page, self.dataset_page, self.training_page,
                     self.validation_page, self.comparison_page, self.prediction_page,
                     self.logs_page):
            self.stack.addWidget(page)

        root.addWidget(sidebar)
        root.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.dataset_status = QLabel("Dataset: not loaded")
        self.model_status = QLabel("Model: none saved")
        self.statusBar().addWidget(self.dataset_status)
        self.statusBar().addPermanentWidget(self.model_status)
        self._update_model_status()

    def _page_index(self, key):
        return [k for k, _ in PAGES].index(key)

    def _switch_page(self, key):
        self.stack.setCurrentIndex(self._page_index(key))
        for page_key, button in self.nav_buttons:
            button.setChecked(page_key == key)

    # ------------------------------------------------------------- wiring
    def _connect_pages(self):
        self.dataset_page.dataset_loaded.connect(self._on_dataset_loaded)
        self.training_page.finished_ok.connect(self._on_training_done)
        self.training_page.failed.connect(self._on_training_failed)

    def _on_dataset_loaded(self, df, path):
        self.dataset_path = path
        self.training_page.set_dataframe(df, path)
        self.prediction_page.set_dataframe(df)
        self.overview_page.refresh(df=df, result=self.training_result)
        self.dataset_status.setText(f"Dataset: {df.shape[0]:,} rows · {df.shape[1]} columns")

    def _on_training_done(self, result):
        self.training_result = result
        self.overview_page.set_training_running(False)
        self.overview_page.refresh(df=self.dataset_page.df, result=result)
        self.validation_page.show_results(result)
        self.comparison_page.show_results(result)
        self.prediction_page.refresh_model_state()
        self._update_model_status()
        self.statusBar().showMessage(
            f"Training finished — best model: {result.best_model}", 10_000
        )
        logger.info("Training completed. Best model: %s", result.best_model)

    def _on_training_failed(self, message):
        self.overview_page.set_training_running(False)
        self.overview_page.refresh(df=self.dataset_page.df, result=self.training_result)
        self.statusBar().showMessage("Training failed — see the Logs page.", 10_000)

    def _update_model_status(self):
        if config.MODEL_PATH.exists():
            self.model_status.setText(f"Model: {config.MODEL_PATH.name} saved")
        else:
            self.model_status.setText("Model: none saved")

    # ------------------------------------------------------------- events
    def closeEvent(self, event):
        # don't close while training is still running
        worker = self.training_page.worker
        if worker is not None and worker.isRunning():
            worker.wait(5000)
        event.accept()
