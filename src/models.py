"""Model factory: the three classifiers used in this project.

Grid keys carry the `model__` prefix because the estimators are always used
inside a Pipeline whose last step is named "model".
"""

import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# sklearn 1.9 deprecated probability=True on SVC in favour of
# CalibratedClassifierCV; we keep the simpler API and silence just this note
warnings.filterwarnings(
    "ignore",
    message="The `probability` parameter was deprecated",
    category=FutureWarning,
)


def build_models():
    return {
        # max_iter raised because the scaled feature space sometimes needs
        # more iterations to converge
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        # probability=True so ROC-AUC can be computed; it makes SVM slower
        # but we need calibrated-ish scores for the comparison
        "Support Vector Machine": SVC(probability=True),
    }


def get_param_grids():
    return {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
        },
        "K-Nearest Neighbors": {
            "model__n_neighbors": [5, 9, 15, 21],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        "Support Vector Machine": {
            "model__C": [0.5, 1.0, 5.0],
            "model__gamma": ["scale"],
            "model__kernel": ["rbf", "linear"],
        },
    }


def strip_prefix(params):
    """{'model__C': 1.0} -> {'C': 1.0} for display purposes."""
    return {key.split("__", 1)[1]: value for key, value in params.items()}
