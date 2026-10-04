"""Dataset page: load a CSV, inspect its health and browse charts."""

import logging

import seaborn as sns
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src import config
from src.data_loader import DatasetError, dataset_summary, load_dataset

from .widgets import CHURN_PALETTE, DataFrameModel, FigureWidget, StatCard

logger = logging.getLogger(__name__)


class DatasetPage(QWidget):
    dataset_loaded = pyqtSignal(object, str)  # (DataFrame, path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title = QLabel("Dataset")
        title.setObjectName("PageTitle")
        subtitle = QLabel("Load the customer CSV, check its quality and explore the raw distributions.")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # file picker row
        picker = QHBoxLayout()
        self.path_edit = QLineEdit(str(config.DATASET_PATH))
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        load_btn = QPushButton("Load Dataset")
        load_btn.setObjectName("PrimaryButton")
        load_btn.clicked.connect(lambda: self.load_from_path(self.path_edit.text().strip()))
        picker.addWidget(self.path_edit, 1)
        picker.addWidget(browse_btn)
        picker.addWidget(load_btn)
        layout.addLayout(picker)

        # health cards
        cards_row = QHBoxLayout()
        self.rows_card = StatCard("Number of Rows", accent="#111827")
        self.cols_card = StatCard("Number of Columns", accent="#111827")
        self.missing_card = StatCard("Missing Values", accent="#d97706")
        self.duplicates_card = StatCard("Duplicate Rows", accent="#d97706")
        for card in (self.rows_card, self.cols_card, self.missing_card, self.duplicates_card):
            cards_row.addWidget(card)
        layout.addLayout(cards_row)

        # tables: preview + column info
        tables_row = QHBoxLayout()

        preview_box = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_table = QTableView()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_model = DataFrameModel()
        self.preview_table.setModel(self.preview_model)
        preview_layout.addWidget(self.preview_table)
        tables_row.addWidget(preview_box, 3)

        info_box = QGroupBox("Columns · Types · Missing")
        info_layout = QVBoxLayout(info_box)
        self.info_table = QTableWidget(0, 3)
        self.info_table.setHorizontalHeaderLabels(["Column", "Type", "Missing"])
        self.info_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.info_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        info_layout.addWidget(self.info_table)
        tables_row.addWidget(info_box, 2)
        layout.addLayout(tables_row)

        # charts
        charts_row = QHBoxLayout()
        self.churn_chart = FigureWidget(height=2.8)
        self.numeric_chart = FigureWidget(height=2.8)
        self.corr_chart = FigureWidget(height=2.8)
        for chart, box_title in (
            (self.churn_chart, "Churn Distribution"),
            (self.numeric_chart, "Monthly Charges by Churn"),
            (self.corr_chart, "Correlation Matrix"),
        ):
            box = QGroupBox(box_title)
            box_layout = QVBoxLayout(box)
            box_layout.addWidget(chart)
            charts_row.addWidget(box)
        layout.addLayout(charts_row)

    # ------------------------------------------------------------------ api
    def load_from_path(self, path):
        try:
            df = load_dataset(path)
        except DatasetError as exc:
            logger.error("Dataset load failed: %s", exc)
            QMessageBox.critical(self, "Dataset Error", str(exc))
            return

        self.df = df
        self.path_edit.setText(str(path))
        self._update_summary(df)
        self._update_preview(df)
        self._update_column_info(df)
        self._draw_charts(df)
        logger.info("Dataset ready for training: %s", path)
        self.dataset_loaded.emit(df, str(path))

    # -------------------------------------------------------------- helpers
    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a CSV dataset", str(config.DATA_DIR), "CSV files (*.csv)")
        if path:
            self.path_edit.setText(path)
            self.load_from_path(path)

    def _update_summary(self, df):
        summary = dataset_summary(df)
        self.rows_card.set_value(f"{summary['rows']:,}")
        self.cols_card.set_value(summary["columns"])
        self.missing_card.set_value(summary["missing_cells"])
        self.missing_card.set_sub("blank cells across all columns")
        self.duplicates_card.set_value(summary["duplicate_rows"])

    def _update_preview(self, df):
        self.preview_model.set_dataframe(df.head(200))

    def _update_column_info(self, df):
        missing = df.isna().sum()
        self.info_table.setRowCount(df.shape[1])
        for row, col in enumerate(df.columns):
            self.info_table.setItem(row, 0, QTableWidgetItem(col))
            self.info_table.setItem(row, 1, QTableWidgetItem(str(df[col].dtype)))
            self.info_table.setItem(row, 2, QTableWidgetItem(str(int(missing[col]))))

    def _draw_charts(self, df):
        try:
            self._draw_churn_chart(df)
            self._draw_numeric_chart(df)
            self._draw_corr_chart(df)
        except Exception as exc:  # noqa: BLE001 - charts must never crash the app
            logger.warning("Could not draw dataset charts: %s", exc)

    def _draw_churn_chart(self, df):
        self.churn_chart.clear()
        ax = self.churn_chart.add_ax(111)
        order = [config.NEGATIVE_LABEL, config.POSITIVE_LABEL]
        sns.countplot(data=df, x=config.TARGET_COLUMN, hue=config.TARGET_COLUMN,
                      hue_order=order, order=order, legend=False,
                      palette=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]], ax=ax)
        ax.set_title("Churn vs retained customers")
        ax.set_xlabel("")
        self.churn_chart.redraw()

    def _draw_numeric_chart(self, df):
        self.numeric_chart.clear()
        ax = self.numeric_chart.add_ax(111)
        sns.histplot(data=df, x="MonthlyCharges", hue=config.TARGET_COLUMN,
                     hue_order=[config.NEGATIVE_LABEL, config.POSITIVE_LABEL],
                     palette=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]],
                     kde=True, ax=ax)
        ax.set_title("Monthly charges split by churn")
        self.numeric_chart.redraw()

    def _draw_corr_chart(self, df):
        self.corr_chart.clear()
        ax = self.corr_chart.add_ax(111)
        numeric = df.select_dtypes(include="number").copy()
        numeric[config.TARGET_COLUMN] = (df[config.TARGET_COLUMN] == config.POSITIVE_LABEL).astype(int)
        numeric = numeric.loc[:, numeric.columns != "SeniorCitizen"] if numeric.shape[1] > 8 else numeric
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, annot=False,
                    square=False, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Numeric correlation matrix")
        ax.tick_params(axis="x", labelsize=8)
        self.corr_chart.redraw()
