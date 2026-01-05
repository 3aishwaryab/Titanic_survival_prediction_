"""Evaluation helpers: plotting and metric report generation."""
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(cm, labels=(0, 1)):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_yticklabels([str(l) for l in labels])
    plt.tight_layout()
    return fig


def plot_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    return fig


def metrics_summary(metrics: Dict[str, Any]) -> Dict[str, float]:
    # Extract scalar metrics
    return {k: float(v) for k, v in metrics.items() if k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
