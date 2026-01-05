"""Streamlit application for Titanic Survival prediction.

Sidebar filters and live prediction using the persisted model in `models/best_model.pkl`.
"""
from typing import Dict
import os
import sys
# Ensure project root is on sys.path so `src` is importable when running via `streamlit run`.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import joblib
from src.data_processing import load_data
from src.modeling import load_model, predict_single

MODEL_PATH = "models/best_model.pkl"

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

st.title("Titanic Survival Prediction")
st.markdown("Predict survival probability for passengers using a trained model.")

@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except ImportError as e:
        # Informative guidance for missing unpickle-time dependencies
        st.error(f"Model load error: {e}.\nHint: run `pip install -r requirements.txt` to install missing packages (e.g., dill).")
        return None
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None


@st.cache_data
def predict_dataframe(_model, df: pd.DataFrame):
    """Run model.predict_proba on a DataFrame, ensuring required columns exist.

    This helper fills missing columns with sensible defaults so uploaded CSVs don't fail
    and caches results for performance. The `_model` parameter is intentionally
    prefixed with an underscore so Streamlit will not attempt to hash the scikit-learn
    Pipeline (which is not hashable by default and would raise an error).
    """
    req_cols = ["Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Pclass"]
    w = df.copy()
    # Fill minimal defaults
    if "Name" not in w.columns:
        w["Name"] = "Unknown, Mr. Unknown"
    if "Sex" not in w.columns:
        w["Sex"] = "male"
    if "Age" not in w.columns:
        w["Age"] = w.get("Age", 0)
    if "Fare" not in w.columns:
        w["Fare"] = w.get("Fare", 0)
    for c in ["SibSp", "Parch", "Pclass"]:
        if c not in w.columns:
            w[c] = 0
    # Ensure types are sensible
    w["Age"] = pd.to_numeric(w["Age"], errors="coerce").fillna(0)
    w["Fare"] = pd.to_numeric(w["Fare"], errors="coerce").fillna(0)
    try:
        proba = _model.predict_proba(w)[:, 1]
    except Exception as e:
        # Re-raise with context for easier debugging in UI
        raise RuntimeError(f"Prediction failed: {e}")
    return proba


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    age_min = int(df["Age"].min())
    age_max = int(df["Age"].max())
    age = st.sidebar.slider("Age", min_value=age_min, max_value=age_max, value=(age_min, age_max), key="filter_age")
    sex = st.sidebar.multiselect("Sex", options=df["Sex"].unique().tolist(), default=df["Sex"].unique().tolist(), key="filter_sex")
    pclass = st.sidebar.multiselect("Pclass", options=sorted(df["Pclass"].unique().tolist()), default=sorted(df["Pclass"].unique().tolist()), key="filter_pclass")
    fare_min = float(df["Fare"].min())
    fare_max = float(df["Fare"].max())
    fare = st.sidebar.slider("Fare", min_value=float(fare_min), max_value=float(fare_max), value=(fare_min, fare_max), key="filter_fare")

    mask = (
        (df["Age"] >= age[0]) & (df["Age"] <= age[1]) &
        (df["Sex"].isin(sex)) &
        (df["Pclass"].isin(pclass)) &
        (df["Fare"] >= fare[0]) & (df["Fare"] <= fare[1])
    )
    return df[mask].copy()


def main():
    df = get_data()
    model = get_model()

    # Apply filters once and reuse across layout
    filtered = sidebar_filters(df)

    st.title("Titanic Survival Prediction")
    st.markdown("Predict survival probability for passengers using a trained model. Use the filters to narrow the set or upload a CSV for batch predictions.")

    # Layout: filters & upload on the left, predictions and visualizations on the right
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Filters & Upload")
        st.write(f"Showing {len(filtered)} passengers matching filters")

        uploaded = st.file_uploader("Upload CSV of passengers for batch prediction", type=["csv"])
        if uploaded is not None:
            udf = pd.read_csv(uploaded)
            st.markdown("**Uploaded data preview**")
            st.dataframe(udf.head(10))
            if model is not None:
                try:
                    preds = predict_dataframe(model, udf)
                    udf = udf.assign(pred_proba=preds)
                    st.dataframe(udf.sort_values("pred_proba", ascending=False).head(20))
                    csv_bytes = udf.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
            else:
                st.warning("Model not loaded")

        st.markdown("---")
        st.subheader("Single passenger prediction")
        with st.form(key="single_predict_form"):
            name = st.text_input("Name", value="Doe, Mr. John")
            sex = st.selectbox("Sex", options=["male", "female"]) 
            pclass = st.selectbox("Pclass", options=[1, 2, 3])
            age = st.slider("Age", min_value=0.0, max_value=120.0, value=30.0, key="single_age")
            sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0, key="single_sibsp")
            parch = st.number_input("Parch", min_value=0, max_value=10, value=0, key="single_parch")
            fare = st.number_input("Fare", min_value=0.0, value=32.20, key="single_fare")
            submit = st.form_submit_button("Predict", key="single_submit")

        if submit:
            sample = pd.DataFrame([{"PassengerId": 0, "Name": name, "Sex": sex, "Age": age, "SibSp": int(sibsp), "Parch": int(parch), "Fare": float(fare), "Pclass": int(pclass)}])
            if model is not None:
                try:
                    res = predict_single(model, sample)
                    st.metric(label="Survival probability", value=f"{res['probability']*100:.1f}%")
                    st.write("**Prediction:**", "Yes" if res["prediction"] == 1 else "No")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            else:
                st.warning("Model not available")

    with col2:
        st.subheader("Top predictions")
        if model is not None and not filtered.empty:
            try:
                preds = predict_dataframe(model, filtered)
                filtered = filtered.assign(pred_proba=preds)
                # Column selector
                default_cols = ["PassengerId", "Name", "Sex", "Age", "Pclass", "Fare", "pred_proba"]
                cols_to_show = st.multiselect("Columns to display", options=default_cols, default=default_cols, key="cols_to_show")
                st.dataframe(filtered.sort_values("pred_proba", ascending=False)[cols_to_show].head(50))
                csv_bytes = filtered.to_csv(index=False).encode("utf-8")
                st.download_button("Download filtered predictions as CSV", data=csv_bytes, file_name="filtered_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Failed to compute predictions for filtered set: {e}")

            # Show latest training artifacts if present
            try:
                import glob, json
                metrics_files = sorted(glob.glob("models/metrics_*.json"))
                if metrics_files:
                    latest = metrics_files[-1]
                    st.markdown(f"**Latest training metrics:** `{latest}`")
                    with open(latest) as fh:
                        metrics = json.load(fh)
                    if isinstance(metrics, dict):
                        # Present a compact summary if available
                        summary = metrics.get("metrics") or metrics.get("summary") or metrics
                        if isinstance(summary, dict):
                            st.write(summary)
                from glob import glob as _glob
                cmfs = sorted(_glob("models/confusion_matrix_*.png"))
                rcfs = sorted(_glob("models/roc_curve_*.png"))
                if cmfs:
                    st.image(cmfs[-1], caption="Confusion matrix", use_column_width=True)
                if rcfs:
                    st.image(rcfs[-1], caption="ROC curve", use_column_width=True)
            except Exception:
                pass
        else:
            st.info("No model or no data to display predictions.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built for reproducible evaluation. Use the `Train` script to re-fit models and update `models/best_model.pkl`.")


if __name__ == "__main__":
    main()