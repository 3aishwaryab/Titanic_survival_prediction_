"""Configuration constants and utility functions.

Keep seeds, file paths and any global constants here for reproducibility.
"""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RAW_DATA_FILE = DATA_DIR / "cleaned_titanic.csv"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MODEL_FILENAME = MODELS_DIR / "best_model.pkl"  # joblib dump
PREPROCESSOR_FILENAME = MODELS_DIR / "preprocessor.pkl"
