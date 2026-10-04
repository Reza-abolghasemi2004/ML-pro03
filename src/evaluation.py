"""Metrics computation and full test-set evaluation."""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, prob_positive):
    """All five metrics in one dict. `prob_positive` is P(churn)."""
    y_true_binary = (np.asarray(y_true) == config.POSITIVE_LABEL).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=config.POSITIVE_LABEL, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=config.POSITIVE_LABEL, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=config.POSITIVE_LABEL, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_binary, prob_positive)),
    }


def evaluate_predictions(pipeline, X, y):
    """Score an already-fitted pipeline on a given split."""
    y_pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)
    positive_index = list(pipeline.classes_).index(config.POSITIVE_LABEL)
    return compute_metrics(y, y_pred, proba[:, positive_index])


def full_test_evaluation(pipeline, X_test, y_test):
    """Everything the UI needs after the final test-set pass:
    metrics, ROC curve points and the confusion matrix."""
    metrics = evaluate_predictions(pipeline, X_test, y_test)

    proba = pipeline.predict_proba(X_test)
    positive_index = list(pipeline.classes_).index(config.POSITIVE_LABEL)
    y_true_binary = (np.asarray(y_test) == config.POSITIVE_LABEL).astype(int)
    fpr, tpr, _ = roc_curve(y_true_binary, proba[:, positive_index])

    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[config.NEGATIVE_LABEL, config.POSITIVE_LABEL])

    return {
        "metrics": metrics,
        "roc_curve": (fpr.tolist(), tpr.tolist()),
        "confusion_matrix": cm.tolist(),
    }
