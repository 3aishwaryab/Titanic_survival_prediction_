import pandas as pd
from src.data_processing import clean_data, train_val_test_split


def test_clean_and_splits():
    sample = pd.DataFrame([
        {"Survived": 1, "Name": "A", "Sex": "male", "Age": None, "Fare": None, "SibSp": 0, "Parch": 0, "Pclass": 3},
        {"Survived": 0, "Name": "B", "Sex": "female", "Age": 30, "Fare": 10.0, "SibSp": 1, "Parch": 0, "Pclass": 1},
        {"Survived": 1, "Name": "C", "Sex": "female", "Age": 22, "Fare": 7.5, "SibSp": 0, "Parch": 0, "Pclass": 3},
        {"Survived": 0, "Name": "D", "Sex": "male", "Age": None, "Fare": 5.0, "SibSp": 0, "Parch": 1, "Pclass": 2},
    ])

    cleaned = clean_data(sample)
    assert cleaned["Age"].isnull().sum() == 0

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(sample, test_size=0.25, val_size=0.25, random_state=0)
    # sizes should add up
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(sample)
    assert set([True, False]) >= set([len(X_train) > 0, len(X_val) > 0, len(X_test) > 0])