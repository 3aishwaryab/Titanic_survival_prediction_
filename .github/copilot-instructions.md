# GitHub Copilot / AI agent instructions — Titanic Survival Project

Purpose: give concise, actionable context so an AI coding agent is immediately productive working on this repo.

Quick architecture overview
- Core modules live in `src/`:
  - `src/data_processing.py` — load `data/processed/cleaned_titanic.csv` and split features/target
  - `src/feature_engineering.py` — all domain transformers (TitleExtractor, FamilyFeatures, BinningTransformer) and `build_preprocessor()` which returns a Pipeline
  - `src/modeling.py` — candidate pipelines (logistic, random_forest, svc), `train_and_select()`, `evaluate_model()`, persistence via `joblib` to `src/config.py` MODEL_FILENAME
  - `src/evaluation.py` — plotting helpers (ROC, confusion matrix)
- UX/services:
  - `app/streamlit_app.py` — interactive UI that expects a persisted pipeline at `models/best_model.pkl` and uses `predict_proba` for probabilities
  - `api/main.py` — FastAPI inference endpoint that uses Pydantic `Passenger` model and `predict_single()` utility
- Config & reproducibility: `src/config.py` contains `RANDOM_SEED`, `TEST_SIZE`, `CV_FOLDS`, and model paths. Always use these constants when adding experiments.

Developer workflows & important commands
- Install dependencies: `pip install -r requirements.txt`
- Train & persist best model: `python scripts/train_model.py` (saves to `models/best_model.pkl` as defined by `src/config.py`)
- Run Streamlit UI locally: `streamlit run app/streamlit_app.py`
- Run inference API: `uvicorn api.main:app --reload --port 8001`
- Docker for Streamlit: `docker build -t titanic-app .` then `docker run -p 8501:8501 titanic-app`
- Tests: `pytest -q` (current tests cover feature engineering; add tests for models and API endpoints)

Project-specific conventions & patterns (do not break these)
- Feature pipeline design: implement stateless sklearn-compatible transformers (inherit `BaseEstimator`, `TransformerMixin`) and expose `fit`/`transform`. Example: add a new feature by creating a transformer in `src/feature_engineering.py` and compose it in `FeatureCreator`.
- Preprocessing/pipeline contract: `build_preprocessor()` returns `(pipeline, feature_names)` where `pipeline` is a `Pipeline` that first creates features then a `ColumnTransformer` that scales numeric (`StandardScaler`) and one-hot encodes categorical (`OneHotEncoder(sparse=False, handle_unknown="ignore")`). Keep this contract to ensure models and API work.
- Model persistence: persist the full sklearn `Pipeline` (preprocessor + classifier) via `joblib.dump()` to `src/config.MODEL_FILENAME`. Streamlit and API expect the full pipeline so feature creation + encoding occurs at inference time.
- Single-record prediction: use `src/modeling.predict_single(model, df)` where `df` is a single-row DataFrame with original columns (Name, Sex, Age, SibSp, Parch, Fare, Pclass).
- Seeds & reproducible splits: Always use `src/config.RANDOM_SEED` when splitting or creating models.

Integration points & examples
- To add a new model candidate: extend `build_model_candidates()` in `src/modeling.py` with a pipeline `(preprocessor, estimator)` and a `param_grid` entry. GridSearchCV is used with `scoring='f1'` by default.
- To add a new feature transformer: implement transformer + add to `FeatureCreator`. Update `tests/test_feature_engineering.py` with a small DataFrame asserting expected output columns.
- To add API input validation or new fields: extend `api/main.py` `Passenger` Pydantic model and confirm `predict_single()` handles missing values or defaults.

Testing & QA to add or follow
- Add tests for: (1) `src/modeling.train_and_select()` (mock small dataset with known labels), (2) `src/modeling.predict_single()`, (3) `api/main.py` endpoints (use `TestClient`) to assert 200 and correct JSON structure.
- Add CI job (GitHub Actions) that runs `pytest`, `flake8`/`ruff` and optionally `mypy`.

Notes for reviewers / agents working on tasks
- Keep changes small and well-tested; update README's `How to run` and `Evaluation notes` sections if you change training/serving behavior.
- Prefer changing `src/config.py` constants instead of hardcoding paths in scripts or app code.
- When adjusting encoders or feature names, update `app/streamlit_app.py` and `api/main.py` reading logic so single-record inputs match pipeline expectations.
- For model upgrades: always persist a version or timestamped filename in `models/` and optionally update `models/best_model.pkl` symlink (or overwrite) after validation.

Contact points in repo (quick grep targets)
- `TitleExtractor`, `FamilyFeatures`, `BinningTransformer` → `src/feature_engineering.py`
- `build_model_candidates`, `train_and_select`, `predict_single` → `src/modeling.py`
- `scripts/train_model.py` ○ training CLI
- `app/streamlit_app.py` ○ UI
- `api/main.py` ○ inference API
- Tests: `tests/test_feature_engineering.py` (example tests)

If anything in this file is unclear or missing, ask for a short example change you'd like automated (e.g., "Add new feature X with a transformer and tests" or "Add RandomForest hyperparameter grid to include min_samples_leaf").
