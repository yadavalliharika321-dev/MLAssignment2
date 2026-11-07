from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ============================================================
# PROJECT CONFIGURATION
# ============================================================
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"

TARGET = "income"
POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"

NUMERIC_FEATURES = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_FEATURES = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RAW_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", TARGET]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# Visual Style Config
BG, PANEL, BORDER = "#0D1117", "#161B22", "#30363D"
TEXT, MUTED = "#E6EDF3", "#8B949E"
AMBER, TEAL = "#F0A050", "#3FB8AF"

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def normalize_labels(df):
    df = df.copy()
    for column in CATEGORICAL_FEATURES + [TARGET]:
        df[column] = df[column].astype(str).str.strip()
    df[TARGET] = df[TARGET].str.rstrip(".")
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].replace("?", "Unknown")
    return df

def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    y_score = positive_scores(model, X)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

MODEL_NOTES = {
    "Logistic Regression": "LogisticRegression(max_iter=2000), using StandardScaler for numeric features and OneHotEncoder(handle_unknown='ignore', sparse_output=False) for categorical features.",
    "Decision Tree": "DecisionTreeClassifier(random_state=42), using the same preprocessing pipeline as the other classifiers.",
    "KNN": "KNeighborsClassifier(n_neighbors=5), using the same preprocessing pipeline. The saved model stores the training data.",
    "Naive Bayes": "GaussianNB, using the same preprocessing pipeline as the other classifiers.",
    "SVM": "LinearSVC(C=1.0, max_iter=3000, random_state=42), using the same preprocessing pipeline. AUC uses decision_function scores because LinearSVC does not provide predict_proba.",
}

@st.cache_resource
def load_models():
    missing_files = [filename for filename in MODEL_FILES.values() if not (MODEL_DIR / filename).is_file()]
    if missing_files:
        raise FileNotFoundError("Missing trained model file(s) in the model/ folder: " + ", ".join(missing_files) + ". Run the model-saving cell in ML_Assignment_2.ipynb first.")
    return {name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()}

@st.cache_data
def load_bundled_test():
    if not TEST_CSV.is_file():
        raise FileNotFoundError("test_data.csv was not found beside app.py. Run the notebook's final export cell to generate it.")
    return pd.read_csv(TEST_CSV, skipinitialspace=True)

@st.cache_data
def bundled_test_bytes():
    return TEST_CSV.read_bytes()

def prepare(raw):
    required_columns = [TARGET, *FEATURES]
    missing_columns = [column for column in required_columns if column not in raw.columns]
    if missing_columns:
        return None, ("Missing required columns: " + ", ".join(missing_columns))
    
    df = normalize_labels(raw[required_columns])
    valid_labels = {NEGATIVE_LABEL, POSITIVE_LABEL}
    found_labels = set(df[TARGET].dropna())
    unknown_labels = sorted(found_labels - valid_labels)

    if unknown_labels:
        return None, f"'{TARGET}' must contain only '{NEGATIVE_LABEL}' or '{POSITIVE_LABEL}'. Found: {', '.join(unknown_labels)}"
    if df[TARGET].isna().any():
        return None, f"The '{TARGET}' column contains missing values."

    X = df[FEATURES].copy()
    try:
        X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError):
        return None, ("Numeric feature columns must contain numbers only: " + ", ".join(NUMERIC_FEATURES))

    if X[NUMERIC_FEATURES].isna().any().any():
        return None, "Numeric feature columns must not contain missing values."
    if X.isna().any().any():
        return None, "Feature columns contain missing values. For unknown categorical values, use '?' so they can be converted to 'Unknown'."

    y = (df[TARGET] == POSITIVE_LABEL).astype(int)
    if y.nunique() < 2:
        return None, f"The test data must contain both '{NEGATIVE_LABEL}' and '{POSITIVE_LABEL}' rows to calculate AUC."
    return (X, y), None

# ============================================================
# STREAMLIT USER INTERFACE & CSS INJECTIONS
# ============================================================
st.set_page_config(
    page_title="Adult Income Classifiers",
    page_icon="📊",
    layout="wide",
)

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.6rem; }}
.hero {{
    background: linear-gradient(120deg, #1C2430 0%, #2A1D10 100%);
    border: 1px solid {BORDER}; border-radius: 16px; padding: 22px 28px; margin-bottom: 18px;
}}
.hero h1 {{ font-size: 2.1rem; margin: 0 0 4px 0; color: {TEXT}; }}
.hero p {{ color: #C9D1D9; margin: 0 0 12px 0; font-size: 1.02rem; }}
.chip {{
    display: inline-block; padding: 3px 12px; margin: 2px 6px 2px 0; border-radius: 999px;
    background: rgba(240,160,80,0.12); border: 1px solid rgba(240,160,80,0.45);
    color: {AMBER}; font-size: 0.82rem; font-weight: 600;
}}
.chip.teal {{ background: rgba(63,184,175,0.12); border-color: rgba(63,184,175,0.45); color: {TEAL}; }}
.step {{ font-size: 1.12rem; font-weight: 700; color: {TEXT}; margin-bottom: 2px; }}
.step .num {{
    display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px;
    border-radius: 50%; background: {AMBER}; color: {BG}; font-weight: 800; margin-right: 10px;
}}
.step-sub {{ color: {MUTED}; font-size: 0.88rem; margin: 0 0 8px 38px; }}
.status {{
    background: {PANEL}; border: 1px solid {BORDER}; border-left: 4px solid {AMBER};
    border-radius: 10px; padding: 10px 16px; margin: 6px 0 14px 0; color: #C9D1D9;
}}
.status b {{ color: {TEXT}; }}
.badge {{ padding: 2px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 700; }}
.badge.req {{ background: {AMBER}; color: {BG}; }}
.badge.bonus {{ background: {TEAL}; color: {BG}; }}
.blurb {{ color: #C9D1D9; font-size: 0.92rem; margin-top: 8px; }}
[data-testid="stMetric"] {{ background: {PANEL}; border-radius: 12px; }}
[data-testid="stMetricValue"] {{ color: {AMBER}; }}
[data-testid="stFileUploaderDropzone"] {{
    border: 2px dashed rgba(240,160,80,0.55); background: rgba(240,160,80,0.05);
}}
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR LAYOUT -----------------
with st.sidebar:
    st.markdown("### 📊 Dashboard Controls")
    st.markdown(
        "1. **Load Data Source** in container ①\n"
        "2. **Toggle Model Selector** in section ②\n"
        "3. Review evaluation benchmarks matrix metrics"
    )
    st.divider()
    st.markdown("### 📦 Features at a glance")
    st.markdown(
        f"- Target Field: `{TARGET}`\n"
        f"- Features: {len(FEATURES)} Dimensions\n"
        f"- Categorical: {len(CATEGORICAL_FEATURES)}\n"
        f"- Numeric: {len(NUMERIC_FEATURES)}"
    )
    st.divider()
    st.caption("Trained models are loaded straight from joblib binaries. retrain methods are not executed.")

# ----------------- HERO CALLOUT BOX -----------------
st.markdown(f"""
<div class="hero">
  <h1>📊 Adult Income (>50K): Classifier Comparison</h1>
  <p>Will an individual's salary cross the 50K boundary based on demographic metrics? Each entry tracks one case from census data. Upload a custom configuration or stick with the validation holdout split.</p>
  <span class="chip">{len(FEATURES)} Features</span>
  <span class="chip teal">5 Architectures</span>
  <span class="chip teal">No live training loops</span>
</div>
""", unsafe_allow_html=True)

# Interactive control blocks splitting
c_data, c_model = st.columns([1.15, 1], gap="large")

with c_data:
    with st.container(border=True):
        st.markdown('<div class="step"><span class="num">1</span>Load your test data</div>'
                    '<div class="step-sub">CSV holding target parameters and dimensions.</div>', unsafe_allow_html=True)
        data_source = st.radio("Data Choice", ["Bundled test_data.csv", "Upload CSV"], label_visibility="collapsed", horizontal=True)
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
            if uploaded_file is None:
                st.info("Upload a CSV file to evaluate architecture pipelines.")
                st.stop()
            try:
                raw_data = pd.read_csv(uploaded_file, skipinitialspace=True)
            except Exception as exc:
