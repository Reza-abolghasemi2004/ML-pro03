"""Bridges Python logging into the Qt world.

A singleton QObject carries log records as signals so any page (Logs page,
training log window) can subscribe without owning the logger.
"""

import logging

from PyQt6.QtCore import QObject, pyqtSignal


class LogBus(QObject):
    record = pyqtSignal(str, str)  # (level name, formatted message)

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class BusLogHandler(logging.Handler):
    """Forwards every log record to the LogBus singleton."""

    def __init__(self):
        super().__init__()
        self.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record):
        try:
            LogBus.instance().record.emit(record.levelname, self.format(record))
        except Exception:
            # never let a logging failure take down the app
            pass


def attach_bus_handler():
    root = logging.getLogger()
    if not any(isinstance(handler, BusLogHandler) for handler in root.handlers):
        root.addHandler(BusLogHandler())
