"""Feature engineering pipeline for Titanic data.

Contains domain-aware feature creation and a scikit-learn ColumnTransformer for encoding and scaling.
"""
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class TitleExtractor(BaseEstimator, TransformerMixin):
    """Extract title from passenger `Name` and group rare titles.

    If `Name` column is missing, infer a coarse title from `Sex` and `Age`.
    """

    def __init__(self, rare_threshold: int = 10):
        self.rare_threshold = rare_threshold

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "Name" in df.columns:
            df = df.assign(Title=df["Name"].apply(lambda n: n.split(",")[1].split(".")[0].strip()))
        else:
            # Fallback heuristic for datasets without Name
            def infer_title(row):
                if row.get("Sex") == "female":
                    if row.get("Age") is not None and row.get("Age") < 18:
                        return "Miss"
                    return "Mrs"
                else:
                    if row.get("Age") is not None and row.get("Age") < 14:
                        return "Master"
                    return "Mr"

            df = df.assign(Title=df.apply(infer_title, axis=1))

        title_counts = df["Title"].value_counts()
        rare_titles = title_counts[title_counts < self.rare_threshold].index
        df["Title"] = df["Title"].replace(list(rare_titles), "Rare")
        return df[["Title"]]


class FamilyFeatures(BaseEstimator, TransformerMixin):
    """Create FamilySize and IsAlone features.

    FamilySize = SibSp + Parch + 1
    IsAlone = FamilySize == 1
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = df.assign(FamilySize=df["SibSp"] + df["Parch"] + 1)
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        return df[["FamilySize", "IsAlone"]]


class BinningTransformer(BaseEstimator, TransformerMixin):
    """Create Fare bins and Age bins."""

    def __init__(self, age_bins=None, fare_bins=None):
        self.age_bins = age_bins or [0, 12, 20, 40, 60, 80]
        self.fare_bins = fare_bins or [0, 7.91, 14.454, 31, 1000]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["AgeBin"] = pd.cut(df["Age"], bins=self.age_bins, labels=False, include_lowest=True)
        df["FareBin"] = pd.cut(df["Fare"], bins=self.fare_bins, labels=False, include_lowest=True)
        return df[["AgeBin", "FareBin"]]


class FeatureCreator(BaseEstimator, TransformerMixin):
    """Composite transformer that adds Title, FamilySize/IsAlone, AgeBin and FareBin to the DataFrame."""

    def __init__(self, rare_threshold: int = 10, age_bins=None, fare_bins=None):
        # Store init params as attributes for sklearn cloning compatibility
        self.rare_threshold = rare_threshold
        self.age_bins = age_bins
        self.fare_bins = fare_bins
        self.title_ext = TitleExtractor(rare_threshold=rare_threshold)
        self.family = FamilyFeatures()
        self.bins = BinningTransformer(age_bins=age_bins, fare_bins=fare_bins)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # Title
        t = self.title_ext.transform(df)
        df = df.join(t)
        # Family
        f = self.family.transform(df)
        df = df.join(f)
        # Bins
        b = self.bins.transform(df)
        df = df.join(b)
        # Interaction features
        # Sex × Pclass is a helpful categorical interaction (e.g., 'male_3') for the model
        df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
        return df


def build_preprocessor() -> Tuple[Pipeline, list]:
    """Construct a Pipeline that first creates features, then applies encoding/scaling.

    Returns:
        pipeline: Pipeline that yields a numpy array suitable for sklearn estimators
        feature_names: Names of features (informational)
    """
    # Define the feature creator to add new columns
    feature_creator = FeatureCreator()

    # Feature groups for preprocessor
    numeric_features = ["Age", "Fare", "FamilySize"]
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_features = ["Sex", "Pclass", "Title", "AgeBin", "FareBin", "IsAlone", "Sex_Pclass"]
    # Use the newer `sparse_output` argument to avoid deprecation warnings in recent scikit-learn
    # Keep behavior consistent: return dense arrays for downstream estimators
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(steps=[("feature_creator", feature_creator), ("preprocessor", preprocessor)])

    feature_names = numeric_features + categorical_features
    return pipeline, feature_names


# Keep full_feature_pipeline for compatibility
def full_feature_pipeline() -> Pipeline:
    """Return a pipeline that creates features only (no scaling/encoding)."""
    feature_pipeline = Pipeline(
        steps=[
            ("feature_creator", FeatureCreator()),
        ]
    )
    return feature_pipeline
