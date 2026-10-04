"""Small shared helpers: directories and logging setup."""

import logging

from . import config

_LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def ensure_dirs():
    """Create the folders the app writes into, if they are missing."""
    for folder in (config.DATA_DIR, config.MODELS_DIR, config.LOGS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def setup_logging(level=logging.INFO):
    """Configure root logging once: console + rotating-free file output.

    The Qt side adds its own handler (ui.log_bus) so log records also show
    up inside the dashboard.
    """
    root = logging.getLogger()
    if getattr(root, "_churn_configured", False):
        return root

    ensure_dirs()
    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # matplotlib is very chatty at INFO level and nobody needs that
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    root._churn_configured = True
    return root
