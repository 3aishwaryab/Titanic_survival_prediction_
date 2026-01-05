# Titanic Survival Prediction — Production-ready refactor

## Overview
This repository refactors the academic Titanic project into a modular, production-ready codebase suitable for strict academic evaluation and portfolio demonstration.

## Architecture ✅
- `src/` — core reusable modules
  - `data_processing.py` — data loading and splitting
  - `feature_engineering.py` — domain-aware features and preprocessing pipeline
  - `modeling.py` — candidate models, training, selection, persistence
  - `evaluation.py` — metric calculation and plots
  - `config.py` — global constants, file paths and reproducibility
- `app/` — Streamlit UI for interactive exploration and single passenger prediction
- `api/` — FastAPI inference service for realtime predictions
- `models/` — saved model artifacts (created after training)
- `scripts/` — utility scripts (e.g., `scripts/train_model.py`)
- `data/processed/cleaned_titanic.csv` — cleaned dataset used for training

## Key features implemented ✅
- Clean, modular PEP8-compliant code with docstrings and type hints
- Domain-aware feature engineering: `FamilySize`, `IsAlone`, title extraction + rare title grouping, `AgeBin`, `FareBin`, interaction-ready features
- Categorical encoding (OneHot) and numeric scaling (StandardScaler)
- Model candidates: Logistic Regression, **Decision Tree**, Random Forest, SVM
- Cross-validation and GridSearchCV for hyperparameter tuning
- Evaluation: Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix
- **Feature importance visualization** for tree-based models
- **SHAP explanations** for model interpretability ("Why did passenger X survive?")
- Streamlit app with sidebar filters and single-passenger prediction (live confidence %)
- FastAPI inference service accepting JSON payloads and returning probability
- Dockerfile to run the Streamlit app
- **Ready for Streamlit Cloud deployment**

## How to run locally 🧪
1. Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Train the model (saves to `models/best_model.pkl`):

```bash
python scripts/train_model.py
```

3. Run Streamlit UI (http://localhost:8501):

```bash
streamlit run app/streamlit_app.py
```

4. Run the FastAPI inference server (http://localhost:8001):

```bash
uvicorn api.main:app --reload --port 8001
```

Example curl:

```bash
curl -X POST "http://127.0.0.1:8001/predict" -H "Content-Type: application/json" -d '{"Name":"Doe, Mr. John","Sex":"male","Age":29,"SibSp":0,"Parch":0,"Fare":7.25,"Pclass":3}'
```

## Docker
Build and serve the Streamlit app:

```bash
docker build -t titanic-app .
docker run -p 8501:8501 titanic-app
```

## Evaluation notes
- Use `scripts/train_model.py` to re-run training with GridSearchCV and CV folds (see `src/modeling.py` for candidate grids).
- Model persistence uses `joblib` to save the full pipeline (preprocessing + estimator) for consistent predictions.
- Training artifacts (metrics JSON and plot PNGs) are saved under `models/metrics_*.json`, `models/confusion_matrix_*.png` and `models/roc_curve_*.png` for reproducibility and grading.
- If you see errors while loading a saved model about a missing module (e.g., `ModuleNotFoundError: No module named 'dill'`) or warnings about scikit-learn versions, run `pip install -r requirements.txt` to install `dill` and use the pinned `scikit-learn==1.3.2` for best compatibility.

## Deployment 🚀

### Streamlit Cloud (Recommended - Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file path: `app/streamlit_app.py`
5. Deploy!

**Your app will be live at:** `https://<your-username>-titanic-project.streamlit.app`

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

### Local Deployment

```bash
streamlit run app/streamlit_app.py
```

## New Features ✨

### Model Interpretability
- **Feature Importance**: See which features drive predictions (for tree-based models)
- **SHAP Explanations**: Understand "Why did passenger X survive?" with interactive explanations
- Both features are integrated into the Streamlit app

### Model Candidates
- **Decision Tree** added to model selection pipeline
- All models evaluated with 5-fold cross-validation and GridSearchCV

## Next steps & improvements
- Add CI with linting and unit tests for preprocessing, model training, and API endpoints
- Add more advanced feature encoding (target encoding / embeddings)
- Add structured logging and observability for production inference
- Add learning curve analysis for bias-variance diagnosis
