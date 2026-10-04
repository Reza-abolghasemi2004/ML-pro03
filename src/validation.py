"""Manual stratified cross-validation loop.

sklearn's cross_validate would be shorter, but doing the folds by hand lets
the training worker report "fold 3/5 done" live in the UI.
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from . import config
from .evaluation import evaluate_predictions

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def cross_validate_pipeline(pipeline, X, y, n_splits=config.CV_FOLDS, on_fold=None):
    """Returns {metric: {'mean', 'std', 'folds': [...]}} for one pipeline."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    per_fold = {metric: [] for metric in METRICS}

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_metrics = evaluate_predictions(fold_pipe, X.iloc[val_idx], y.iloc[val_idx])

        for metric in METRICS:
            per_fold[metric].append(fold_metrics[metric])

        if on_fold is not None:
            on_fold(fold_index, n_splits, fold_metrics)

    return {
        metric: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "folds": values,
        }
        for metric, values in per_fold.items()
    }
