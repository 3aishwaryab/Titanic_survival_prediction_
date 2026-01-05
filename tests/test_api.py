import joblib
import pandas as pd
from fastapi.testclient import TestClient
from src.modeling import build_model_candidates
import os

MODEL_PATH = "models/best_model.pkl"


def test_api_predict():
    # Train and persist a simple pipeline
    df = pd.DataFrame([
        {"Name": "A, Mr. A", "Sex": "male", "Age": 22.0, "Fare": 7.25, "SibSp": 0, "Parch": 0, "Pclass": 3, "Survived": 0},
        {"Name": "B, Mrs. B", "Sex": "female", "Age": 38.0, "Fare": 71.2833, "SibSp": 1, "Parch": 0, "Pclass": 1, "Survived": 1},
        {"Name": "C, Miss. C", "Sex": "female", "Age": 26.0, "Fare": 7.925, "SibSp": 0, "Parch": 0, "Pclass": 3, "Survived": 1},
        {"Name": "D, Mr. D", "Sex": "male", "Age": 35.0, "Fare": 8.05, "SibSp": 0, "Parch": 0, "Pclass": 3, "Survived": 0},
    ])

    X = df.drop(columns=["Survived"]) 
    y = df["Survived"]

    pipeline = build_model_candidates()["logistic"][0]
    pipeline.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # Import app and ensure MODEL is available to the app
    import api.main as api_main
    api_main.MODEL = pipeline
    client = TestClient(api_main.app)

    payload = {"Name": "Test, Mr. T","Sex": "male","Age": 30,"SibSp": 0,"Parch": 0,"Fare": 7.25,"Pclass": 3}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert "prediction" in j and "probability" in j
