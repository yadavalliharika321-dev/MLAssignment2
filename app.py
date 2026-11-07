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
# STREAMLIT USER INTERFACE
# ============================================================
st.set_page_config(
    page_title="Adult Income Classifiers",
    page_icon="📊",
    layout="wide",
)

# ----------------- SIDEBAR LAYOUT -----------------
with st.sidebar:
    st.title("📊 Adult Income")
    st.caption("Classifier Comparison Dashboard")
    st.divider()
    
    st.caption("TEST DATA")
    data_source = st.radio("Choose Data Source", ["Bundled test_data.csv", "Upload CSV"], label_visibility="collapsed")
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(f"Upload CSV ({TARGET} & features)", type=["csv"])
        if uploaded_file is None:
            st.info("Upload a CSV file to see the model results.")
            st.stop()
        try:
            raw_data = pd.read_csv(uploaded_file, skipinitialspace=True)
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
            st.stop()
    else:
        try:
            raw_data = load_bundled_test()
        except Exception as exc:
            st.error(str(exc))
            st.stop()
            
    if TEST_CSV.is_file():
        st.download_button("Download sample test_data.csv", data=bundled_test_bytes(), file_name="test_data.csv", mime="text/csv", use_container_width=True)
        
    st.divider()
    
    st.caption("MODEL SELECTION")
    selected_model = st.radio("Choose an architecture:", ["All models", *MODEL_FILES.keys()])
    
    st.divider()
    st.caption("HYPERPARAMETERS & NOTES")
    if selected_model != "All models":
        st.info(MODEL_NOTES[selected_model])
    else:
        st.info("Showing consolidated benchmark overview for all 5 active trained models.")

# Data Pipeline Processing
prepared_data, validation_error = prepare(raw_data)
if validation_error:
    st.error(validation_error)
    st.stop()

X, y = prepared_data

try:
    models = load_models()
except Exception as exc:
    st.error(str(exc))
    st.stop()

# ----------------- MAIN PANEL LAYOUT -----------------
st.title("Adult Income (>50K): Classifier Comparison")
st.info("This app evaluates the five trained pipelines saved from ML_Assignment_2.ipynb. Models are loaded from the model/ folder; they are not retrained in the app.")

# High-Level Structural Summary KPI Metrics Row
col_rows, col_neg, col_pos, col_feat = st.columns(4)
col_rows.metric("Evaluated Rows", f"{len(y):,}")
col_neg.metric(f"Negative ({NEGATIVE_LABEL})", f"{(y == 0).sum():,}")
col_pos.metric(f"Positive ({POSITIVE_LABEL})", f"{(y == 1).sum():,}")
col_feat.metric("Total Features", f"{len(FEATURES)}")

st.divider()

# Segmented Workspaces Tabs
tab_eval, tab_format = st.tabs(["Evaluation Model Matrix", "Expected CSV Schema Format"])

with tab_eval:
    # ----------------- ALL-MODEL VIEW -----------------
    if selected_model == "All models":
        results = pd.DataFrame({name: evaluate_model(model, X, y) for name, model in models.items()}).T
        st.subheader(f"Metrics on {len(y):,} test rows")
        st.dataframe(results[METRIC_NAMES].round(4), use_container_width=True)
        st.caption("The six metrics use the same definitions as the notebook. AUC uses predicted probabilities where available and decision scores for LinearSVC.")

    # ----------------- INDIVIDUAL-MODEL VIEW -----------------
    else:
        model = models[selected_model]
        st.subheader(f"Performance: {selected_model}")
        metrics = evaluate_model(model, X, y)

        # Performance Metric Score Cards Row
        metric_columns = st.columns(len(METRIC_NAMES))
        for column, metric_name in zip(metric_columns, METRIC_NAMES):
            column.metric(metric_name, f"{metrics[metric_name]:.4f}")

        st.divider()
        
        y_pred = model.predict(X)
        
        # Calculate Matrix Elements using safe, explicit coordinates
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        row_totals = cm.sum(axis=1, keepdims=True)
        row_percentages = (cm / np.clip(row_totals, 1, None)) * 100
        
        # Dual-Column Visual Analytics Layout Blocks
        matrix_col, report_col = st.columns(2)
        
        with matrix_col:
            st.subheader("Confusion Matrix")
            
            # Explicit string assignments bypass looping bracket formatting rules perfectly
            c00 = f"{cm[0, 0]:,} ({row_percentages[0, 0]:.1f}%)"
