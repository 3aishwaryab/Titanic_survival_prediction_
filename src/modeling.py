"""Model training, selection and persistence utilities."""
from typing import Tuple, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer

from .config import RANDOM_SEED, MODEL_FILENAME, PREPROCESSOR_FILENAME, TEST_SIZE, CV_FOLDS
from .data_processing import load_data, split_features_target
from .feature_engineering import build_preprocessor


def build_model_candidates(random_state: int = RANDOM_SEED) -> Dict[str, Tuple[Pipeline, Dict[str, Any]]]:
    """Define model pipelines with hyperparameter grids for GridSearchCV."""
    preprocessor, _ = build_preprocessor()

    candidates = {}

    # Logistic Regression
    pipe_lr = Pipeline(steps=[("preprocessor", preprocessor), ("clf", LogisticRegression(random_state=random_state, max_iter=1000))])
    param_grid_lr = {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]}
    candidates["logistic"] = (pipe_lr, param_grid_lr)

    # Random Forest
    pipe_rf = Pipeline(steps=[("preprocessor", preprocessor), ("clf", RandomForestClassifier(random_state=random_state))])
    param_grid_rf = {"clf__n_estimators": [50, 100], "clf__max_depth": [None, 5, 10]}
    candidates["random_forest"] = (pipe_rf, param_grid_rf)

    # SVM
    pipe_svc = Pipeline(steps=[("preprocessor", preprocessor), ("clf", SVC(probability=True, random_state=random_state))])
    param_grid_svc = {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"]}
    candidates["svc"] = (pipe_svc, param_grid_svc)

    return candidates


def train_and_select(X: pd.DataFrame, y: pd.Series, cv: int = CV_FOLDS, random_state: int = RANDOM_SEED) -> Dict[str, Any]:
    """Train candidate models with GridSearchCV and return results plus best model.

    For each candidate, perform GridSearchCV (scoring by F1), then compute cross-validated
    metrics (accuracy, precision, recall, f1, roc_auc) to summarize candidate performance.

    Returns a dict with keys: best_model, best_name, results (cv metrics per model)
    """
    from sklearn.model_selection import cross_validate
    import json
    import datetime

    candidates = build_model_candidates(random_state=random_state)

    best_score = -np.inf
    best_model = None
    best_name = None
    results = {}

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    for name, (pipeline, grid) in candidates.items():
        gs = GridSearchCV(pipeline, grid, cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(X, y)
        best = gs.best_estimator_

        # Compute a collection of CV metrics using the best estimator
        cv_res = cross_validate(best, X, y, cv=cv, scoring=list(scoring.values()), return_train_score=False)
        summary = {m: float(cv_res[f"test_{m}"].mean()) for m in scoring.keys()}

        results[name] = {
            "best_params": gs.best_params_,
            **{f"cv_{k}": v for k, v in summary.items()},
        }

        # We still use f1 as selection criterion
        if summary["f1"] > best_score:
            best_score = summary["f1"]
            best_model = best
            best_name = name

    # Persist best model as both canonical and timestamped artifact
    from .config import MODELS_DIR
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    versioned_path = MODELS_DIR / f"best_model_{best_name}_{ts}.pkl"

    joblib.dump(best_model, str(MODEL_FILENAME))
    joblib.dump(best_model, str(versioned_path))

    # Save results summary to a JSON file for record keeping
    metrics_path = MODELS_DIR / f"metrics_summary_{ts}.json"
    with open(metrics_path, "w") as fh:
        json.dump({"best_name": best_name, "results": results}, fh, indent=2)

    return {"best_name": best_name, "best_model": best_model, "results": results, "metrics_path": str(metrics_path)}


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Compute evaluation metrics and return a dictionary with arrays where appropriate."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }
    return metrics


def load_model(path: str = MODEL_FILENAME):
    """Load a persisted sklearn Pipeline model.

    Args:
        path: Path to joblib file.

    Raises:
        ImportError: with actionable instructions when dependencies required to unpickle are missing.
    """
    try:
        return joblib.load(path)
    except ModuleNotFoundError as e:
        # Common case: missing optional dependency such as `dill` used when object was pickled.
        raise ImportError(
            f"Failed to load model because a required Python module is missing: {e.name}.\n"
            "If you see `dill` in the error, run `pip install dill` and ensure dependency versions match those used when the model was saved (e.g., scikit-learn 1.3.2)."
        ) from e
    except Exception as e:
        # Generic re-raise with context for easier debugging
        raise RuntimeError(f"Failed to load model from {path}: {e}") from e


def predict_single(model, record: pd.DataFrame) -> dict:
    """Given a fitted pipeline and a single-record DataFrame, return prediction and probability.

    The input record must contain the original dataset columns (e.g., Name, Sex, Age, Fare, SibSp, Parch, Pclass).
    """
    pred = model.predict(record)[0]
    proba = model.predict_proba(record)[0, 1]
    return {"prediction": int(pred), "probability": float(proba)}


if __name__ == "__main__":
    # Quick training entrypoint
    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    results = train_and_select(X_train, y_train)
    best_model = results["best_model"]
    evals = evaluate_model(best_model, X_test, y_test)
    print("Best model:", results["best_name"])
    print("Evaluation:", evals)
    # save model done in train_and_select
