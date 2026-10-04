"""The full training pipeline, orchestrated in one place.

Order of operations:

    clean -> feature engineering -> train/test split -> per model:
    baseline fit -> stratified CV -> grid search tuning -> test evaluation
    -> comparison -> best model saved with joblib

This module knows nothing about Qt. The UI passes plain callback functions
(progress, log, fold updates, ...) so the same code also runs headless.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .evaluation import evaluate_predictions, full_test_evaluation
from .feature_engineering import add_features
from .models import build_models, get_param_grids, strip_prefix
from .preprocessing import build_preprocessor, clean_dataset
from .tuning import tune_pipeline
from .validation import cross_validate_pipeline

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    target: str
    n_samples: int
    n_train: int
    n_test: int
    model_names: list
    results: dict                 # model name -> cv/tuning/test details
    best_model: str
    best_pipeline: object
    numeric_features: list
    categorical_features: list
    model_path: str
    total_time: float
    dataset_path: str = ""


def run_full_training(raw_df, selected_models=None, cb=None, dataset_path=""):
    """Run everything and return a TrainingResult.

    `cb` is an optional dict of callbacks: progress(int), status(str),
    log(str), model(str), fold(name, i, n, metrics), model_done(name, res).
    """
    cb = cb or {}

    def emit(name, *args):
        fn = cb.get(name)
        if fn is not None:
            fn(*args)

    def log(message, level="info"):
        getattr(logger, level)(message)
        emit("log", message)

    started = time.time()

    # 1. clean + feature engineering -----------------------------------------
    emit("status", "Cleaning data")
    emit("progress", 2)
    log("Cleaning dataset...")
    df = clean_dataset(raw_df)
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        log(f"Removing {duplicate_rows} duplicate rows.")
        df = df.drop_duplicates().reset_index(drop=True)

    df = add_features(df)
    log(f"Data ready: {df.shape[0]} rows, {df.shape[1] - 1} features (4 engineered).")
    emit("progress", 8)

    # 2. split once, before anything gets fitted ------------------------------
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    log(f"Train/test split: {len(X_train)} train rows, {len(X_test)} test rows.")
    emit("status", "Train/test split done")
    emit("progress", 10)

    numeric_features = config.RAW_NUMERIC + config.ENGINEERED_NUMERIC
    categorical_features = config.RAW_CATEGORICAL
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 3. per-model loop --------------------------------------------------------
    all_models = build_models()
    grids = get_param_grids()
    names = selected_models or config.MODEL_NAMES
    names = [n for n in config.MODEL_NAMES if n in names]  # keep display order

    results = {}
    per_model_span = 78.0 / len(names)  # progress 12% -> 90%

    for index, name in enumerate(names, start=1):
        base = 12 + (index - 1) * per_model_span
        emit("model", name)
        emit("status", f"Baseline fit: {name}")

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", all_models[name]),
        ])

        # baseline: default hyperparameters, quick fit on the training split
        step_start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - step_start
        baseline = evaluate_predictions(pipeline, X_train, y_train)
        log(
            f"[{index}/{len(names)}] {name}: baseline fitted in {train_time:.1f}s "
            f"(train accuracy {baseline['accuracy']:.3f}, ROC-AUC {baseline['roc_auc']:.3f})."
        )
        emit("progress", int(base + 6))

        # stratified cross-validation on the training split only
        emit("status", f"Cross-validation: {name}")
        log(f"{name}: {config.CV_FOLDS}-fold stratified cross-validation started.")

        def _on_fold(fold_i, n_folds, fold_metrics, _name=name, _base=base):
            emit("fold", _name, fold_i, n_folds, fold_metrics)
            emit("progress", int(_base + 6 + (fold_i / n_folds) * 10))
            log(f"{_name}: fold {fold_i}/{n_folds} done "
                f"(accuracy {fold_metrics['accuracy']:.3f}, ROC-AUC {fold_metrics['roc_auc']:.3f}).")

        cv_start = time.time()
        cv_summary = cross_validate_pipeline(pipeline, X_train, y_train, on_fold=_on_fold)
        cv_time = time.time() - cv_start
        log(f"{name}: CV finished in {cv_time:.1f}s — mean ROC-AUC "
            f"{cv_summary['roc_auc']['mean']:.3f} ± {cv_summary['roc_auc']['std']:.3f}.")

        # hyperparameter tuning; GridSearchCV refits the winner on the full
        # training set, so best_estimator_ is our final model for this family
        emit("status", f"Hyperparameter tuning: {name}")
        grid = grids[name]
        n_combinations = 1
        for values in grid.values():
            n_combinations *= len(values)
        log(f"{name}: grid search over {n_combinations} hyperparameter combinations.")

        tune_start = time.time()
        search = tune_pipeline(pipeline, grid, X_train, y_train)
        tune_time = time.time() - tune_start
        best_params = strip_prefix(search.best_params_)
        log(f"{name}: best parameters {best_params} "
            f"(CV ROC-AUC {search.best_score_:.3f}, tuning took {tune_time:.1f}s).")

        # final evaluation on the untouched test split
        emit("status", f"Test evaluation: {name}")
        best_pipeline = search.best_estimator_
        test_eval = full_test_evaluation(best_pipeline, X_test, y_test)
        m = test_eval["metrics"]
        log(f"{name}: test results — accuracy {m['accuracy']:.3f}, precision {m['precision']:.3f}, "
            f"recall {m['recall']:.3f}, F1 {m['f1']:.3f}, ROC-AUC {m['roc_auc']:.3f}.")

        results[name] = {
            "pipeline": best_pipeline,
            "cv": cv_summary,
            "cv_time": cv_time,
            "train_time": train_time,
            "tune_time": tune_time,
            "best_params": best_params,
            "best_cv_score": float(search.best_score_),
            "test": m,
            "roc_curve": test_eval["roc_curve"],
            "confusion_matrix": test_eval["confusion_matrix"],
        }
        emit("model_done", name, results[name])
        emit("progress", int(base + per_model_span))

    # 4. comparison + best model ----------------------------------------------
    emit("status", "Comparing models")
    emit("progress", 92)
    best_model = max(results, key=lambda n: results[n]["test"]["roc_auc"])
    log(f"Model comparison finished. Best model: {best_model} "
        f"(test ROC-AUC {results[best_model]['test']['roc_auc']:.3f}).")

    # 5. persist the winning pipeline -----------------------------------------
    emit("status", "Saving model")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    test_metrics = results[best_model]["test"]
    bundle = {
        "pipeline": results[best_model]["pipeline"],
        "model_name": best_model,
        "best_params": results[best_model]["best_params"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_column": config.TARGET_COLUMN,
        "test_metrics": test_metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, config.MODEL_PATH)
    log(f"Best model saved to {config.MODEL_PATH.relative_to(config.BASE_DIR)}.")

    emit("status", "Completed")
    emit("progress", 100)
    total_time = time.time() - started
    log(f"Training pipeline finished in {total_time:.0f}s.")

    return TrainingResult(
        target=config.TARGET_COLUMN,
        n_samples=int(len(df)),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        model_names=list(results.keys()),
        results=results,
        best_model=best_model,
        best_pipeline=results[best_model]["pipeline"],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_path=str(config.MODEL_PATH),
        total_time=total_time,
        dataset_path=str(dataset_path),
    )
