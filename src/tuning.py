"""Hyperparameter tuning with GridSearchCV."""

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config


def tune_pipeline(pipeline, param_grid, X, y, n_splits=config.CV_FOLDS, scoring=config.TUNING_SCORING):
    """Run grid search on a full pipeline (preprocessing included).

    refit=True means the returned best_estimator_ is already retrained on the
    whole training set, ready for the final test evaluation.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=True,
        n_jobs=1,  # keep it serial: friendlier inside a Qt worker thread
    )
    search.fit(X, y)
    return search
