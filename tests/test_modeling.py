import pandas as pd
from src.modeling import build_model_candidates, predict_single


def test_pipeline_predict_single():
    # minimal synthetic dataset
    df = pd.DataFrame([
        {"Name": "A, Mr. A", "Sex": "male", "Age": 22.0, "Fare": 7.25, "SibSp": 0, "Parch": 0, "Pclass": 3, "Survived": 0},
        {"Name": "B, Mrs. B", "Sex": "female", "Age": 38.0, "Fare": 71.2833, "SibSp": 1, "Parch": 0, "Pclass": 1, "Survived": 1},
        {"Name": "C, Miss. C", "Sex": "female", "Age": 26.0, "Fare": 7.925, "SibSp": 0, "Parch": 0, "Pclass": 3, "Survived": 1},
        {"Name": "D, Mr. D", "Sex": "male", "Age": 35.0, "Fare": 8.05, "SibSp": 0, "Parch": 0, "Pclass": 3, "Survived": 0},
    ])

    X = df.drop(columns=["Survived"]) 
    y = df["Survived"]

    # Use logistic pipeline for a quick fit
    candidates = build_model_candidates()
    pipeline = candidates["logistic"][0]
    pipeline.fit(X, y)

    # single record
    sample = X.iloc[[0]]
    res = predict_single(pipeline, sample)
    assert set(res.keys()) == {"prediction", "probability"}
    assert 0.0 <= res["probability"] <= 1.0