"""Logs page: everything the application logs, filterable by level."""

from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src import config

from .log_bus import LogBus

LEVEL_COLORS = {
    "INFO": "#93c5fd",
    "WARNING": "#fcd34d",
    "ERROR": "#fca5a5",
    "CRITICAL": "#f87171",
    "DEBUG": "#9ca3af",
}


class LogsPage(QWidget):
    MAX_LINES = 3000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        LogBus.instance().record.connect(self._on_record)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel("Logs")
        title.setObjectName("PageTitle")
        subtitle = QLabel(f"Application activity. Also written to {config.LOG_FILE.name} inside the logs folder.")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls = QHBoxLayout()
        self.filters = {}
        for level in ("INFO", "WARNING", "ERROR"):
            box = QCheckBox(level)
            box.setChecked(True)
            self.filters[level] = box
            controls.addWidget(box)
        controls.addStretch(1)
        save_btn = QPushButton("Save As...")
        save_btn.clicked.connect(self._save)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear)
        controls.addWidget(save_btn)
        controls.addWidget(clear_btn)
        layout.addLayout(controls)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("LogView")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(self.MAX_LINES)
        layout.addWidget(self.log_view, 1)

    # -------------------------------------------------------------- helpers
    def _on_record(self, level, message):
        # ERROR filter also covers CRITICAL
        if level in ("INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG"):
            bucket = "ERROR" if level == "CRITICAL" else level
            if bucket in self.filters and not self.filters[bucket].isChecked():
                return
        color = LEVEL_COLORS.get(level, "#e2e8f0")
        self.log_view.appendHtml(
            f'<span style="color:{color}; white-space:pre-wrap;">{_escape(message)}</span>'
        )

    def _clear(self):
        self.log_view.clear()

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save logs", "app_logs.txt", "Text files (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.log_view.toPlainText())


def _escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
