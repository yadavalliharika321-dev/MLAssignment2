from pathlib import Path

import joblib
import pandas as pd
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

NUMERIC_FEATURES = [
    "age", "fnlwgt", "education_num", "capital_gain",
    "capital_loss", "hours_per_week",
]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"]

MODEL_NOTES = {
    "Logistic Regression": (
        "LogisticRegression(max_iter=2000), with StandardScaler for numeric "
        "features and OneHotEncoder(handle_unknown='ignore', sparse_output=False) "
        "for categorical features."
    ),
    "Decision Tree": (
        "DecisionTreeClassifier(random_state=42), using the shared preprocessing pipeline."
    ),
    "KNN": (
        "KNeighborsClassifier(n_neighbors=5), using the shared preprocessing pipeline. "
        "The saved model stores the training data."
    ),
    "Naive Bayes": (
        "GaussianNB, using the shared preprocessing pipeline."
    ),
    "SVM": (
        "LinearSVC(C=1.0, max_iter=3000, random_state=42), using the shared "
        "preprocessing pipeline. AUC uses decision_function scores because "
        "LinearSVC does not provide predict_proba."
    ),
}


def normalize_labels(df):
    df = df.copy()
    for column in CATEGORICAL_FEATURES + [TARGET]:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].str.rstrip(".")
    existing_cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    df[existing_cats] = df[existing_cats].replace("?", "Unknown")
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
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_score),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


@st.cache_resource
def load_models():
    missing = [f for f in MODEL_FILES.values() if not (MODEL_DIR / f).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s) in model/: " + ", ".join(missing)
            + ". Run the model-saving cell in ML_Assignment_2.ipynb first."
        )
    return {name: joblib.load(MODEL_DIR / filename)
            for name, filename in MODEL_FILES.items()}


@st.cache_data
def load_bundled_test():
    if not TEST_CSV.is_file():
        raise FileNotFoundError(
            "test_data.csv was not found beside app.py. Generate it using "
            "the notebook's final export cell."
        )
    return pd.read_csv(TEST_CSV, skipinitialspace=True)


def prepare(raw):
    required = [TARGET, *FEATURES]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        return None, "Missing required columns: " + ", ".join(missing)

    df = normalize_labels(raw[required])
    valid = {NEGATIVE_LABEL, POSITIVE_LABEL}
    found = set(df[TARGET].dropna())
    unknown = sorted(found - valid)
    if unknown:
        return None, f"'{TARGET}' must contain only {NEGATIVE_LABEL} or {POSITIVE_LABEL}. Found: {', '.join(unknown)}"
    if df[TARGET].isna().any():
        return None, f"The '{TARGET}' column contains missing values."

    X = df[FEATURES].copy()
    try:
        X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError):
        return None, "Numeric feature columns must contain numbers only: " + ", ".join(NUMERIC_FEATURES)
    if X[NUMERIC_FEATURES].isna().any().any():
        return None, "Numeric feature columns must not contain missing values."
    if X.isna().any().any():
        return None, "Feature columns contain missing values. Use '?' for unknown categorical values."
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)
    if y.nunique() < 2:
        return None, f"Test data must contain both '{NEGATIVE_LABEL}' and '{POSITIVE_LABEL}' rows to calculate AUC."
    return (X, y, df), None


# ============================================================
# STREAMLIT DASHBOARD
# ============================================================
st.set_page_config(
    page_title="Adult Income ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    [data-testid="stSidebar"] {background: #101c35;}
    [data-testid="stSidebar"] * {color: #f4f7ff;}
    .hero {
        padding: 1.5rem 1.8rem; border-radius: 16px;
        background: linear-gradient(115deg, #142b50, #286a8e);
        color: white; margin-bottom: 1.2rem;
    }
    .hero h1 {color: white; margin-bottom: .25rem;}
    .hero p {color: #e1edf7; margin-bottom: 0;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e5eaf2;
        padding: 14px; border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📊 IncomeIQ")
    st.caption("Adult income classification")
    page = st.radio(
        "NAVIGATION",
        ["Home", "Dataset Explorer", "Model Comparison", "Model Details", "Project Information"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Five trained classifiers · Six evaluation metrics")

st.markdown(
    '<div class="hero"><h1>Adult Income Classification Dashboard</h1>'
    '<p>Explore the dataset, compare machine-learning models, and inspect prediction results.</p></div>',
    unsafe_allow_html=True,
)

with st.expander("Choose test dataset", expanded=True):
    data_source = st.radio(
        "Data source", ["Bundled test_data.csv", "Upload CSV"],
        horizontal=True, label_visibility="collapsed",
    )
    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV with income and all 14 feature columns", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV file to load the dashboard.")
            st.stop()
        try:
            raw_data = pd.read_csv(uploaded, skipinitialspace=True)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()
    else:
        try:
            raw_data = load_bundled_test()
        except Exception as exc:
            st.error(str(exc))
            st.stop()

prepared, error = prepare(raw_data)
if error:
    st.error(error)
    st.stop()
X, y, clean_data = prepared

try:
    models = load_models()
except Exception as exc:
    st.error(str(exc))
    st.stop()

# Evaluate once; reuse results throughout the dashboard.
predictions = {name: model.predict(X) for name, model in models.items()}
metrics = pd.DataFrame(
    {name: evaluate_model(model, X, y) for name, model in models.items()}
).T[METRIC_NAMES]
prediction_frame = clean_data.copy()
prediction_frame.insert(0, "row_id", range(1, len(prediction_frame) + 1))
for model_name, pred in predictions.items():
    prediction_frame[f"{model_name} prediction"] = [
        POSITIVE_LABEL if value == 1 else NEGATIVE_LABEL for value in pred
    ]

best_model_name = metrics["F1"].idxmax()
class_counts = clean_data[TARGET].value_counts().reindex(
    [NEGATIVE_LABEL, POSITIVE_LABEL], fill_value=0
)

if page == "Home":
    st.subheader("Project at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{len(clean_data):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Missing values", f"{int(raw_data.isna().sum().sum()):,}")
    c4.metric("Models evaluated", len(models))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Income class distribution")
        st.bar_chart(class_counts.rename_axis("Income class"))
    with right:
        st.markdown("#### Model F1 comparison")
        st.bar_chart(metrics[["F1"]].sort_values("F1", ascending=False))

    st.markdown("#### Evaluation snapshot")
    st.dataframe(metrics.style.format("{:.4f}").highlight_max(axis=0), use_container_width=True)
    st.success(f"Highest F1 score in this test run: {best_model_name} ({metrics.loc[best_model_name, 'F1']:.4f}).")
    st.caption("The app loads saved model pipelines; it does not retrain models.")

elif page == "Dataset Explorer":
    st.subheader("Dataset Explorer")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(clean_data):,}")
    b.metric("Predictor features", len(FEATURES))
    c.metric("Columns in source", len(raw_data.columns))

    st.markdown("#### Income distribution")
    st.bar_chart(class_counts.rename_axis("Income class"))
    st.markdown("#### Missing values by column")
    missing_table = raw_data.isna().sum().rename("Missing values").to_frame()
    missing_table["Missing %"] = (missing_table["Missing values"] / len(raw_data) * 100).round(2)
    st.dataframe(missing_table, use_container_width=True)

    st.markdown("#### Feature overview")
    feature_info = pd.DataFrame({
        "Feature": FEATURES,
        "Type": ["Numeric" if f in NUMERIC_FEATURES else "Categorical" for f in FEATURES],
        "Unique values": [clean_data[f].nunique() for f in FEATURES],
    })
    st.dataframe(feature_info, hide_index=True, use_container_width=True)
    with st.expander("Preview dataset"):
        st.dataframe(clean_data.head(100), use_container_width=True)

elif page == "Model Comparison":
    st.subheader("Model Comparison")
    st.caption("All metrics are calculated on the currently selected test dataset.")
    st.dataframe(metrics.style.format("{:.4f}").highlight_max(axis=0), use_container_width=True)
    st.markdown("#### Compare evaluation metrics")
    selected_metrics = st.multiselect(
        "Metrics to display", METRIC_NAMES,
        default=["Accuracy", "Precision", "Recall", "F1", "AUC"],
    )
    if selected_metrics:
        st.bar_chart(metrics[selected_metrics])
    st.download_button(
        "⬇ Download evaluation metrics (CSV)",
        data=metrics.to_csv(index_label="Model").encode("utf-8"),
        file_name="model_evaluation_metrics.csv",
        mime="text/csv",
    )

elif page == "Model Details":
    st.subheader("Model Details & Prediction Explorer")
    selected_model = st.selectbox("Select classifier", list(models.keys()))
    model = models[selected_model]
    y_pred = predictions[selected_model]
    model_metrics = metrics.loc[selected_model]

    st.markdown(f"**Algorithm:** {selected_model}")
    st.caption(MODEL_NOTES[selected_model])
    metric_cols = st.columns(6)
    for col, metric_name in zip(metric_cols, METRIC_NAMES):
        col.metric(metric_name, f"{model_metrics[metric_name]:.4f}")

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {NEGATIVE_LABEL}", f"Actual {POSITIVE_LABEL}"],
        columns=[f"Predicted {NEGATIVE_LABEL}", f"Predicted {POSITIVE_LABEL}"],
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix heatmap")
        st.dataframe(
            cm_df.style.background_gradient(cmap="Blues").format("{:,}"),
            use_container_width=True,
        )
        st.caption("Rows represent actual classes; columns represent predicted classes.")
    with right:
        st.markdown("#### Classification report")
        report = classification_report(
            y, y_pred, labels=[0, 1],
            target_names=[NEGATIVE_LABEL, POSITIVE_LABEL],
            digits=4, zero_division=0, output_dict=True,
        )
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    st.markdown("#### Actual vs predicted income")
    explorer = prediction_frame[[
        "row_id", TARGET, f"{selected_model} prediction"
    ]].rename(columns={TARGET: "Actual income", f"{selected_model} prediction": "Predicted income"})
    explorer["Correct prediction"] = explorer["Actual income"] == explorer["Predicted income"]
    filter_choice = st.selectbox("Show predictions", ["All rows", "Correct only", "Incorrect only"])
    if filter_choice == "Correct only":
        explorer = explorer[explorer["Correct prediction"]]
    elif filter_choice == "Incorrect only":
        explorer = explorer[~explorer["Correct prediction"]]
    st.dataframe(explorer.head(500), use_container_width=True)
    st.caption("Showing up to 500 rows in the table. Download the complete prediction file below.")
    st.download_button(
        "⬇ Download prediction results (CSV)",
        data=prediction_frame.to_csv(index=False).encode("utf-8"),
        file_name="income_prediction_results.csv",
        mime="text/csv",
    )

elif page == "Project Information":
    st.subheader("Project Information")
    st.markdown("""
    **Objective**

    Compare supervised machine-learning classifiers for predicting whether an
    individual's annual income is above or at/below the $50K threshold.

    **Algorithms included**

    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors (KNN)
    - Gaussian Naive Bayes
    - Linear Support Vector Machine (SVM)

    **Methodology**

    1. Load the test dataset and validate the target and predictor columns.
    2. Normalize categorical values and convert the target labels to binary form.
    3. Load the trained preprocessing-and-classifier pipelines from the `model/` folder.
    4. Generate predictions and positive-class scores.
    5. Evaluate models using Accuracy, Precision, Recall, F1, AUC, and MCC.
    6. Inspect confusion matrices, classification reports, and row-level predictions.

    **Data and model notes**

    The app expects `test_data.csv` beside `app.py` and five saved `.joblib`
    pipelines inside a `model/` folder. Models are loaded as saved and are not
    retrained by the dashboard. AUC uses predicted probabilities when available;
    the LinearSVC model uses its decision scores.
    """)
    st.markdown("#### Required predictor features")
    st.write(", ".join(FEATURES))

st.divider()
st.caption("Adult Income Classifier Dashboard · Evaluation results depend on the selected test dataset.")
