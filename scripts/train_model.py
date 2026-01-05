"""Script to train models and persist the best model.

Usage: python scripts/train_model.py
"""
import sys, os
# Ensure project root is on sys.path so `src` is importable when running from scripts/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processing import load_data, split_features_target
from src.modeling import train_and_select, evaluate_model, load_model
from src.config import RANDOM_SEED, TEST_SIZE
from sklearn.model_selection import train_test_split


def main():
    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    results = train_and_select(X_train, y_train)
    best_model = results["best_model"]
    metrics = evaluate_model(best_model, X_test, y_test)

    # Save evaluation artifacts
    from src.evaluation import plot_confusion_matrix, plot_roc
    import datetime
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    metrics_path = f"models/metrics_{ts}.json"

    import json
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # plots
    cm_fig = plot_confusion_matrix(metrics["confusion_matrix"]) if "confusion_matrix" in metrics else None
    roc_fig = plot_roc(y_test, metrics["y_proba"]) if "y_proba" in metrics else None

    if cm_fig:
        cm_fig.savefig(f"models/confusion_matrix_{ts}.png")
    if roc_fig:
        roc_fig.savefig(f"models/roc_curve_{ts}.png")

    print("Training completed. Best model:", results["best_name"])
    print("Metrics:", metrics)
    print("Artifacts saved:", metrics_path)

if __name__ == "__main__":
    main()