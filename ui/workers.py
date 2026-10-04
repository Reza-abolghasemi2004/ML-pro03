"""Background workers so heavy ML work never blocks the UI thread.

TrainingWorker runs the whole src.training pipeline inside a QThread and
streams progress to the UI through Qt signals.
"""

from PyQt6.QtCore import QThread, pyqtSignal

from src.training import run_full_training


class TrainingWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)              # current stage text
    log = pyqtSignal(str)                 # one log line
    current_model = pyqtSignal(str)       # model being worked on
    fold_done = pyqtSignal(str, int, int, dict)   # model, fold i, n folds, metrics
    model_done = pyqtSignal(str, dict)    # model, full result dict
    succeeded = pyqtSignal(object)        # TrainingResult
    failed = pyqtSignal(str)              # error message

    def __init__(self, dataframe, selected_models=None, dataset_path="", parent=None):
        super().__init__(parent)
        self._df = dataframe
        self._selected_models = selected_models
        self._dataset_path = dataset_path

    def run(self):
        callbacks = {
            "progress": self.progress.emit,
            "status": self.status.emit,
            "log": self.log.emit,
            "model": self.current_model.emit,
            "fold": self.fold_done.emit,
            "model_done": self.model_done.emit,
        }
        try:
            result = run_full_training(
                self._df,
                selected_models=self._selected_models,
                cb=callbacks,
                dataset_path=self._dataset_path,
            )
        except Exception as exc:  # noqa: BLE001 - report any failure to the UI
            self.log.emit(f"ERROR  Training failed: {exc}")
            self.failed.emit(str(exc))
            return

        self.succeeded.emit(result)
