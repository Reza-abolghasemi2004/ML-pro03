"""Entry point: `python app.py` opens the churn prediction dashboard."""

import logging
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from src.utils import ensure_dirs, setup_logging


def main():
    ensure_dirs()
    setup_logging()
    logger = logging.getLogger("app")

    try:
        from ui.main_window import MainWindow
        from ui.styles import STYLESHEET
    except ImportError as exc:
        # most common cause: PyQt6 or matplotlib missing from the environment
        print(f"Missing dependency: {exc}\nRun: pip install -r requirements.txt")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("Customer Churn Prediction")
    app.setStyleSheet(STYLESHEET)

    try:
        window = MainWindow()
    except Exception as exc:  # noqa: BLE001 - show a dialog instead of a traceback
        logger.exception("Could not start the dashboard")
        QMessageBox.critical(None, "Startup Error", f"The dashboard could not start:\n\n{exc}")
        sys.exit(1)

    window.show()
    logger.info("Dashboard started.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
