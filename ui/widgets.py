"""Reusable UI building blocks: cards, status rows, charts, table model."""

import matplotlib

# the dashboard never uses pyplot: figures are created directly and drawn
# through the Qt canvas below, which also fixes the backend for us
import seaborn as sns
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QAbstractTableModel, Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout

sns.set_theme(style="whitegrid")

# matplotlib ships a dark-ish default text color that clashes with our cards
matplotlib.rcParams["axes.edgecolor"] = "#d1d5db"
matplotlib.rcParams["figure.facecolor"] = "#ffffff"
matplotlib.rcParams["axes.facecolor"] = "#ffffff"

CHURN_PALETTE = {"Yes": "#ef4444", "No": "#22c55e"}
NEGATIVE_COLOR = "#22c55e"
POSITIVE_COLOR = "#ef4444"


class StatCard(QFrame):
    """Small white card with a title, a big value and an optional subtitle."""

    def __init__(self, title, value="—", accent="#4f46e5", parent=None):
        super().__init__(parent)
        self.setObjectName("StatCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(2)

        self.title_label = QLabel(title.upper())
        self.title_label.setObjectName("CardTitle")

        self.value_label = QLabel(value)
        self.value_label.setObjectName("CardValue")
        self.value_label.setStyleSheet(f"color: {accent};")
        self.value_label.setWordWrap(True)

        self.sub_label = QLabel("")
        self.sub_label.setObjectName("CardSub")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.sub_label)

    def set_value(self, value):
        self.value_label.setText(str(value))

    def set_sub(self, text):
        self.sub_label.setText(text)


class StatusItem(QFrame):
    """One line of the system checklist: coloured dot + label."""

    COLORS = {"ok": "#16a34a", "pending": "#d97706", "fail": "#dc2626", "off": "#9ca3af"}

    def __init__(self, text, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        self.dot = QLabel("●")
        self.dot.setStyleSheet(f"color: {self.COLORS['off']}; font-size: 13px;")
        self.label = QLabel(text)
        layout.addWidget(self.dot)
        layout.addWidget(self.label, 1)

    def set_state(self, state):
        """state: 'ok' | 'pending' | 'fail' | 'off'"""
        self.dot.setStyleSheet(f"color: {self.COLORS.get(state, self.COLORS['off'])}; font-size: 13px;")


class FigureWidget(FigureCanvasQTAgg):
    """Matplotlib canvas with a few conveniences for the dashboard."""

    def __init__(self, height=3.0, parent=None):
        self.fig = Figure(figsize=(6.0, height), dpi=100)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._height = height

    def clear(self):
        self.fig.clear()

    def add_ax(self, *args, **kwargs):
        return self.fig.add_subplot(*args, **kwargs)

    def redraw(self):
        self.fig.tight_layout()
        self.draw_idle()


class DataFrameModel(QAbstractTableModel):
    """Read-only table model to show a DataFrame preview."""

    def __init__(self, dataframe=None, max_rows=200, parent=None):
        super().__init__(parent)
        self._df = dataframe
        self._max_rows = max_rows

    def set_dataframe(self, dataframe):
        self.beginResetModel()
        self._df = dataframe
        self.endResetModel()

    def rowCount(self, parent=None):
        if self._df is None:
            return 0
        return min(len(self._df), self._max_rows)

    def columnCount(self, parent=None):
        if self._df is None:
            return 0
        return self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or self._df is None:
            return None
        value = self._df.iat[index.row(), index.column()]
        # show NaN as an empty cell instead of 'nan'
        try:
            if value != value:
                return ""
        except TypeError:
            pass
        return str(value)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or self._df is None:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)


def risk_color(risk):
    return {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}.get(risk, "#6b7280")
