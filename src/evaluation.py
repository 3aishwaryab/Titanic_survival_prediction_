"""Evaluation helpers: plotting and metric report generation."""
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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


def get_feature_importance(model, top_n: int = 15) -> Optional[pd.DataFrame]:
    """Extract feature importance for tree-based models.
    
    Args:
        model: Fitted sklearn Pipeline with preprocessor and classifier
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature names and importance scores, or None if model doesn't support feature importance
    """
    from sklearn.pipeline import Pipeline
    
    if not isinstance(model, Pipeline):
        return None
    
    # Check if classifier has feature_importances_ attribute (tree-based models)
    classifier = model.named_steps.get('clf')
    if classifier is None or not hasattr(classifier, 'feature_importances_'):
        return None
    
    importances = classifier.feature_importances_
    
    # Get feature names from preprocessor
    preprocessor = model.named_steps.get('preprocessor')
    if preprocessor is None:
        return None
    
    try:
        # Get feature names after preprocessing
        feature_names = preprocessor.get_feature_names_out()
        # Clean up feature names (remove transformer prefixes like 'num__', 'cat__')
        clean_names = [name.split('__')[-1] if '__' in name else name for name in feature_names]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': clean_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    except Exception:
        # Fallback: use generic feature names
        return pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(importances))],
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)


def plot_feature_importance(model, top_n: int = 15, figsize=(10, 6)):
    """Plot feature importance for tree-based models.
    
    Args:
        model: Fitted sklearn Pipeline
        top_n: Number of top features to plot
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object, or None if feature importance not available
    """
    importance_df = get_feature_importance(model, top_n=top_n)
    if importance_df is None or importance_df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=importance_df, x='importance', y='feature', ax=ax, palette='viridis')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Top {len(importance_df)} Feature Importances')
    plt.tight_layout()
    return fig
