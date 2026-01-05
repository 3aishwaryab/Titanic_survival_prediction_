"""Data loading utilities and simple cleaning helpers.

This module provides functions to load the processed dataset and minimal checks.
"""
from typing import Tuple
import pandas as pd
from .config import RAW_DATA_FILE


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load processed Titanic data CSV.

    Args:
        path: Optional path override. If None, uses `RAW_DATA_FILE` from config.

    Returns:
        DataFrame containing the dataset.
    """
    p = path or str(RAW_DATA_FILE)
    df = pd.read_csv(p)
    return df


def split_features_target(df: pd.DataFrame, target: str = "Survived") -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target.

    Args:
        df: Input dataframe with target column.
        target: Name of target column.

    Returns:
        X, y
    """
    X = df.drop(columns=[target])
    y = df[target].copy()
    return X, y


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform minimal safe cleaning suitable for training and inference.

    - Fill missing numeric values (Age, Fare) with median
    - Ensure types for `Pclass`, `SibSp`, `Parch`

    Returns a cleaned copy of the DataFrame.
    """
    df = df.copy()
    # Numeric imputations
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Ensure integer-like columns
    for c in ("Pclass", "SibSp", "Parch"):
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    return df


def train_val_test_split(
    df: pd.DataFrame,
    target: str = "Survived",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
):
    """Create reproducible train/val/test splits.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split

    df = clean_data(df)
    X = df.drop(columns=[target])
    y = df[target].copy()

    if stratify:
        strat = y
    else:
        strat = None

    # First split: temp / test
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )
    except ValueError:
        # Fallback to non-stratified split when dataset is too small or class counts are insufficient
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )

    # compute val fraction relative to remaining data
    val_frac = val_size / (1 - test_size)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_frac, random_state=random_state, stratify=y_temp if stratify else None
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_frac, random_state=random_state, stratify=None
        )

    return X_train, X_val, X_test, y_train, y_val, y_test
