"""Streamlit application for Titanic Survival prediction.

Production-ready app with model interpretability, error analysis, and clear UI.
"""
from typing import Dict, Optional
import os
import sys
# Ensure project root is on sys.path so `src` is importable when running via `streamlit run`.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import load_data
from src.modeling import load_model, predict_single
from src.evaluation import get_feature_importance, plot_feature_importance, plot_confusion_matrix

MODEL_PATH = "models/best_model.pkl"

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Titanic-themed background
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                    url('https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(20, 30, 50, 0.95), rgba(10, 20, 40, 0.95));
    }
    
    h1 {
        color: #1a4d8c;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-weight: 700;
    }
    
    h2, h3 {
        color: #2c5aa0;
    }
    
    [data-testid="stMetricValue"] {
        color: #1a4d8c;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1a4d8c, #2c5aa0);
        color: white;
        border: none;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def get_data():
    """Load and cache the Titanic dataset."""
    return load_data()

@st.cache_resource
def get_model():
    """Load and cache the trained model."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please run `python scripts/train_model.py` first.")
        return None
    except ImportError as e:
        st.error(f"Model load error: {e}.\n\n**Solution:** Run `pip install -r requirements.txt` to install missing packages (e.g., dill).")
        return None
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

def validate_passenger_input(name: str, sex: str, age: float, pclass: int, 
                             sibsp: int, parch: int, fare: float) -> tuple:
    """Validate passenger input for logical consistency.
    
    Returns:
        (is_valid, error_message)
    """
    if age < 0 or age > 120:
        return False, "Age must be between 0 and 120 years."
    
    if pclass not in [1, 2, 3]:
        return False, "Passenger class must be 1, 2, or 3."
    
    if sibsp < 0 or sibsp > 10:
        return False, "Number of siblings/spouses must be between 0 and 10."
    
    if parch < 0 or parch > 10:
        return False, "Number of parents/children must be between 0 and 10."
    
    if fare < 0:
        return False, "Fare cannot be negative."
    
    # Historical context: Children typically had lower fares
    if age < 12 and fare > 100:
        return False, "Warning: Very high fare for a child. Please verify."
    
    # Family size consistency
    family_size = 1 + sibsp + parch
    if family_size > 15:
        return False, "Total family size (including passenger) exceeds 15. Please verify."
    
    return True, None

@st.cache_data
def predict_dataframe(_model, df: pd.DataFrame):
    """Run model.predict_proba on a DataFrame, ensuring required columns exist."""
    req_cols = ["Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Pclass"]
    w = df.copy()
    w = w.reset_index(drop=True)
    
    # Fill missing columns with sensible defaults
    if "Name" not in w.columns:
        def generate_name(row):
            sex = row.get("Sex", "male")
            return "Unknown, Mrs. Unknown" if sex == "female" else "Unknown, Mr. Unknown"
        w["Name"] = w.apply(generate_name, axis=1)
    
    for col, default in [("Sex", "male"), ("Age", 0), ("Fare", 0), 
                          ("SibSp", 0), ("Parch", 0), ("Pclass", 3)]:
        if col not in w.columns:
            w[col] = default
    
    # Ensure types are correct
    w["Age"] = pd.to_numeric(w["Age"], errors="coerce").fillna(0)
    w["Fare"] = pd.to_numeric(w["Fare"], errors="coerce").fillna(0)
    w["SibSp"] = pd.to_numeric(w["SibSp"], errors="coerce").fillna(0).astype(int)
    w["Parch"] = pd.to_numeric(w["Parch"], errors="coerce").fillna(0).astype(int)
    w["Pclass"] = pd.to_numeric(w["Pclass"], errors="coerce").fillna(3).astype(int)
    
    try:
        proba = _model.predict_proba(w)[:, 1]
        return proba
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

def analyze_errors(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Analyze model misclassifications and return error patterns."""
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Get misclassifications
        errors = X_test[y_pred != y_test].copy()
        if len(errors) == 0:
            return {"error_count": 0, "patterns": {}}
        
        errors['true_label'] = y_test[y_pred != y_test].values
        errors['predicted_label'] = y_pred[y_pred != y_test]
        errors['confidence'] = y_proba[y_pred != y_test]
        
        # Analyze patterns
        patterns = {}
        
        # False negatives (predicted died, actually survived)
        fn_mask = (errors['predicted_label'] == 0) & (errors['true_label'] == 1)
        if fn_mask.sum() > 0:
            fn_data = errors[fn_mask]
            patterns['false_negatives'] = {
                'count': int(fn_mask.sum()),
                'avg_confidence': float(fn_data['confidence'].mean()),
                'common_sex': fn_data['Sex'].mode().iloc[0] if 'Sex' in fn_data.columns else 'N/A',
                'common_pclass': int(fn_data['Pclass'].mode().iloc[0]) if 'Pclass' in fn_data.columns else 'N/A',
            }
        
        # False positives (predicted survived, actually died)
        fp_mask = (errors['predicted_label'] == 1) & (errors['true_label'] == 0)
        if fp_mask.sum() > 0:
            fp_data = errors[fp_mask]
            patterns['false_positives'] = {
                'count': int(fp_mask.sum()),
                'avg_confidence': float(fp_data['confidence'].mean()),
                'common_sex': fp_data['Sex'].mode().iloc[0] if 'Sex' in fp_data.columns else 'N/A',
                'common_pclass': int(fp_data['Pclass'].mode().iloc[0]) if 'Pclass' in fp_data.columns else 'N/A',
            }
        
        return {
            "error_count": int(len(errors)),
            "total_test": int(len(X_test)),
            "error_rate": float(len(errors) / len(X_test)),
            "patterns": patterns
        }
    except Exception as e:
        return {"error": str(e)}

def get_shap_explanation(model, sample: pd.DataFrame) -> Optional[Dict]:
    """Generate SHAP explanation for a single prediction."""
    try:
        import shap
        
        feature_creator = model.named_steps.get('feature_creator')
        preprocessor = model.named_steps.get('preprocessor')
        classifier = model.named_steps.get('clf')
        
        if not all([feature_creator, preprocessor, classifier]):
            return None
        
        # Transform sample
        sample_with_features = feature_creator.transform(sample)
        sample_processed = preprocessor.transform(sample_with_features)
        
        # Check if tree-based model
        is_tree_model = (
            hasattr(classifier, 'tree_') or
            hasattr(classifier, 'estimators_') or
            hasattr(classifier, 'feature_importances_')
        )
        
        if not is_tree_model:
            return None
        
        # Generate SHAP values
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(sample_processed)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class
        
        feature_names = preprocessor.get_feature_names_out()
        shap_array = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        # Create readable explanation
        contributions = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_array,
            'Impact': ['Increases survival probability' if v > 0 else 'Decreases survival probability' 
                      for v in shap_array]
        }).sort_values('SHAP Value', key=lambda x: abs(x), ascending=False)
        
        return {
            'contributions': contributions,
            'base_value': explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
            'shap_values': shap_array
        }
    except Exception:
        return None

# ============================================================================
# SIDEBAR FILTERS (for data exploration)
# ============================================================================

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Create sidebar filters for data exploration."""
    st.sidebar.header("🔍 Data Filters")
    st.sidebar.caption("Filter the dataset to explore passenger patterns")
    
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    age = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max), key="filter_age")
    
    sex_options = df["Sex"].unique().tolist()
    sex = st.sidebar.multiselect("Sex", sex_options, default=sex_options, key="filter_sex")
    
    pclass_options = sorted(df["Pclass"].unique().tolist())
    pclass = st.sidebar.multiselect("Passenger Class", pclass_options, default=pclass_options, key="filter_pclass")
    
    fare_min, fare_max = float(df["Fare"].min()), float(df["Fare"].max())
    fare = st.sidebar.slider("Fare Range", fare_min, fare_max, (fare_min, fare_max), key="filter_fare")
    
    mask = (
        (df["Age"] >= age[0]) & (df["Age"] <= age[1]) &
        (df["Sex"].isin(sex)) &
        (df["Pclass"].isin(pclass)) &
        (df["Fare"] >= fare[0]) & (df["Fare"] <= fare[1])
    )
    
    filtered_count = mask.sum()
    st.sidebar.metric("Filtered Passengers", filtered_count)
    
    return df[mask].copy()

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load data and model
    df = get_data()
    model = get_model()
    
    if model is None:
        st.stop()
    
    # Sidebar filters
    filtered = sidebar_filters(df)
    
    # Main title (only once)
    st.title("🚢 Titanic Survival Prediction")
    st.markdown("**Predict survival probability using a trained machine learning model with interpretability.**")
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Single Prediction", 
        "📊 Batch Prediction", 
        "📈 Model Evaluation", 
        "🔬 Model Interpretability"
    ])
    
    # ========================================================================
    # TAB 1: SINGLE PASSENGER PREDICTION
    # ========================================================================
    with tab1:
        st.header("Single Passenger Prediction")
        st.caption("Enter passenger details to get a survival prediction with explanation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.form("single_prediction_form"):
                st.subheader("Passenger Information")
                
                name = st.text_input("Name", value="Doe, Mr. John", 
                                    help="Enter passenger name (used for title extraction)")
                sex = st.radio("Sex", options=["male", "female"], horizontal=True,
                              help="Passenger gender")
                pclass = st.selectbox("Passenger Class", options=[1, 2, 3],
                                     help="1 = First class, 2 = Second class, 3 = Third class")
                age = st.slider("Age", min_value=0.0, max_value=120.0, value=30.0, step=0.5,
                               help="Passenger age in years")
                sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0,
                                       help="Number of siblings or spouses aboard")
                parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0,
                                      help="Number of parents or children aboard")
                fare = st.number_input("Fare", min_value=0.0, value=32.20, step=0.01,
                                      help="Ticket fare in pounds")
                
                submit = st.form_submit_button("🔮 Predict Survival", use_container_width=True)
        
        with col2:
            if submit:
                # Validate input
                is_valid, error_msg = validate_passenger_input(name, sex, age, pclass, sibsp, parch, fare)
                
                if not is_valid:
                    st.error(f"❌ Invalid input: {error_msg}")
                else:
                    sample = pd.DataFrame([{
                        "PassengerId": 0, "Name": name, "Sex": sex, "Age": age,
                        "SibSp": int(sibsp), "Parch": int(parch), "Fare": float(fare),
                        "Pclass": int(pclass)
                    }])
                    
                    try:
                        # Get prediction
                        res = predict_single(model, sample)
                        prob = res['probability']
                        pred = res['prediction']
                        
                        # Display prediction
                        st.subheader("Prediction Result")
                        st.metric("Survival Probability", f"{prob*100:.1f}%")
                        st.metric("Prediction", "✅ Survived" if pred == 1 else "❌ Did Not Survive")
                        
                        # SHAP Explanation
                        st.markdown("---")
                        st.subheader("🔍 Why This Prediction?")
                        st.caption("Understanding which factors influenced this prediction")
                        
                        shap_result = get_shap_explanation(model, sample)
                        
                        if shap_result:
                            contributions = shap_result['contributions']
                            
                            # Summary
                            top_positive = contributions[contributions['SHAP Value'] > 0].head(3)
                            top_negative = contributions[contributions['SHAP Value'] < 0].head(3)
                            
                            if len(top_positive) > 0:
                                st.success("**Factors increasing survival probability:**")
                                for _, row in top_positive.iterrows():
                                    st.write(f"  • {row['Feature']}: +{row['SHAP Value']:.3f}")
                            
                            if len(top_negative) > 0:
                                st.error("**Factors decreasing survival probability:**")
                                for _, row in top_negative.iterrows():
                                    st.write(f"  • {row['Feature']}: {row['SHAP Value']:.3f}")
                            
                            # Detailed table
                            with st.expander("📋 View all feature contributions"):
                                st.dataframe(contributions.head(15), use_container_width=True)
                            
                            # Visualization
                            fig, ax = plt.subplots(figsize=(10, 6))
                            top_contrib = contributions.head(10).sort_values('SHAP Value')
                            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in top_contrib['SHAP Value']]
                            ax.barh(top_contrib['Feature'], top_contrib['SHAP Value'], color=colors, alpha=0.7)
                            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                            ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
                            ax.set_title('Top 10 Feature Contributions', fontsize=12, fontweight='bold')
                            ax.grid(axis='x', alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                        else:
                            st.info("💡 SHAP explanations are available for tree-based models (Decision Tree, Random Forest).")
                    
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
    
    # ========================================================================
    # TAB 2: BATCH PREDICTION
    # ========================================================================
    with tab2:
        st.header("Batch Prediction via CSV Upload")
        st.caption("Upload a CSV file with passenger data to get predictions for multiple passengers")
        
        uploaded = st.file_uploader("Choose CSV file", type=["csv"], 
                                   help="CSV should contain columns: Name, Sex, Age, SibSp, Parch, Fare, Pclass")
        
        if uploaded is not None:
            try:
                udf = pd.read_csv(uploaded)
                st.success(f"✅ Loaded {len(udf)} passengers")
                
                st.subheader("Data Preview")
                st.dataframe(udf.head(10), use_container_width=True)
                
                if st.button("🔮 Predict All", use_container_width=True):
                    with st.spinner("Generating predictions..."):
                        preds = predict_dataframe(model, udf)
                        udf_result = udf.copy()
                        udf_result['Survival_Probability'] = preds
                        udf_result['Predicted_Survival'] = (preds > 0.5).astype(int)
                        udf_result = udf_result.sort_values('Survival_Probability', ascending=False)
                        
                        st.subheader("Predictions")
                        st.dataframe(udf_result, use_container_width=True)
                        
                        # Download button
                        csv_bytes = udf_result.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Predictions CSV",
                            data=csv_bytes,
                            file_name="titanic_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
    
    # ========================================================================
    # TAB 3: MODEL EVALUATION
    # ========================================================================
    with tab3:
        st.header("Model Performance Evaluation")
        st.caption("Global model performance metrics and error analysis")
        
        # Load metrics
        try:
            import glob
            import json
            
            metrics_files = sorted(glob.glob("models/metrics_*.json"))
            if not metrics_files:
                st.warning("No evaluation metrics found. Please run `python scripts/train_model.py` first.")
            else:
                latest = metrics_files[-1]
                with open(latest) as fh:
                    metrics = json.load(fh)
                
                # Display metrics
                st.subheader("📊 Performance Metrics")
                
                if isinstance(metrics, dict):
                    summary = metrics.get("metrics") or metrics.get("summary") or metrics
                    
                    if isinstance(summary, dict):
                        # Key metrics in columns
                        metric_cols = st.columns(5)
                        metric_cols[0].metric("Accuracy", f"{summary.get('accuracy', 0):.3f}",
                                             help="Overall prediction accuracy")
                        metric_cols[1].metric("Precision", f"{summary.get('precision', 0):.3f}",
                                              help="Of predicted survivors, how many actually survived")
                        metric_cols[2].metric("Recall", f"{summary.get('recall', 0):.3f}",
                                             help="Of actual survivors, how many were correctly identified")
                        metric_cols[3].metric("F1 Score", f"{summary.get('f1', 0):.3f}",
                                            help="Harmonic mean of precision and recall")
                        metric_cols[4].metric("ROC AUC", f"{summary.get('roc_auc', 0):.3f}",
                                             help="Area under ROC curve (higher is better)")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Confusion Matrix")
                    st.caption("Global model performance: True vs Predicted")
                    cm_files = sorted(glob.glob("models/confusion_matrix_*.png"))
                    if cm_files:
                        st.image(cm_files[-1], use_container_width=True)
                    else:
                        st.info("Confusion matrix not available. Run training script to generate.")
                
                with col2:
                    st.subheader("ROC Curve")
                    st.caption("Global model performance: True Positive Rate vs False Positive Rate")
                    roc_files = sorted(glob.glob("models/roc_curve_*.png"))
                    if roc_files:
                        st.image(roc_files[-1], use_container_width=True)
                    else:
                        st.info("ROC curve not available. Run training script to generate.")
                
                # Error Analysis
                st.markdown("---")
                st.subheader("🔍 Error Analysis")
                st.caption("Understanding where the model makes mistakes")
                
                try:
                    # Load test data for error analysis
                    from src.data_processing import split_features_target
                    from sklearn.model_selection import train_test_split
                    from src.config import TEST_SIZE, RANDOM_SEED
                    
                    X, y = split_features_target(df)
                    _, X_test, _, y_test = train_test_split(
                        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
                    )
                    
                    error_analysis = analyze_errors(model, X_test, y_test)
                    
                    if "error" not in error_analysis:
                        st.metric("Total Errors", error_analysis.get('error_count', 0),
                                 delta=f"{error_analysis.get('error_rate', 0)*100:.1f}% error rate")
                        
                        patterns = error_analysis.get('patterns', {})
                        
                        if 'false_negatives' in patterns:
                            fn = patterns['false_negatives']
                            st.warning(f"**False Negatives ({fn['count']}):** Model predicted 'Did Not Survive' but passenger actually survived. "
                                     f"Common pattern: {fn['common_sex']} passengers in class {fn['common_pclass']}. "
                                     f"Average confidence: {fn['avg_confidence']:.2f}")
                        
                        if 'false_positives' in patterns:
                            fp = patterns['false_positives']
                            st.info(f"**False Positives ({fp['count']}):** Model predicted 'Survived' but passenger actually died. "
                                  f"Common pattern: {fp['common_sex']} passengers in class {fp['common_pclass']}. "
                                  f"Average confidence: {fp['avg_confidence']:.2f}")
                    else:
                        st.warning("Error analysis unavailable")
                except Exception as e:
                    st.warning(f"Error analysis unavailable: {e}")
        
        except Exception as e:
            st.error(f"Could not load evaluation metrics: {e}")
    
    # ========================================================================
    # TAB 4: MODEL INTERPRETABILITY
    # ========================================================================
    with tab4:
        st.header("Global Model Interpretability")
        st.caption("Understanding which features the model considers most important overall")
        
        importance_df = get_feature_importance(model, top_n=15)
        
        if importance_df is not None and not importance_df.empty:
            st.subheader("📊 Feature Importance")
            st.caption("Top features that drive predictions across all passengers")
            
            # Display table
            st.dataframe(
                importance_df.rename(columns={'feature': 'Feature', 'importance': 'Importance'}),
                use_container_width=True
            )
            
            # Display plot
            fig = plot_feature_importance(model, top_n=15)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("---")
            st.subheader("💡 Interpretation")
            top_feature = importance_df.iloc[0]['feature']
            st.info(f"**Most Important Feature:** `{top_feature}`\n\n"
                   f"This feature has the highest impact on survival predictions across all passengers. "
                   f"Features are ranked by their contribution to the model's decision-making process.")
        else:
            st.info("💡 Feature importance is available for tree-based models (Decision Tree, Random Forest). "
                   "If you're using Logistic Regression or SVM, consider retraining with a tree-based model.")
    
    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Built for reproducible evaluation. Run `python scripts/train_model.py` to retrain models.")


if __name__ == "__main__":
    main()
