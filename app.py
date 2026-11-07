# Adult Income Classifier Lab - the Streamlit front end.
# It only LOADS the saved joblib models from model/ and scores whatever csv it's given -
# nothing gets trained or retrained in here.

from __future__ import annotations

import io
from pathlib import Path

import joblib
import matplotlib

# No display on the server, so draw straight into image buffers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"

TARGET = "income"
NEGATIVE_LABEL = "<=50K"
POSITIVE_LABEL = ">50K"

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

MODEL_BLURBS = {
    "Logistic Regression": "A linear classification model that estimates the probability of earning more than $50K. The saved pipeline includes preprocessing.",
    "Decision Tree": "A tree-based model that makes predictions through a sequence of feature-based decision rules.",
    "KNN": "K-Nearest Neighbors predicts income using the labels of the five nearest training observations. The saved pipeline includes preprocessing.",
    "Naive Bayes": "Gaussian Naive Bayes applies Bayes' theorem with a conditional independence assumption between input features.",
    "SVM": "Linear Support Vector Classifier finds a separating decision boundary. Its AUC is calculated from decision scores.",
}

METRIC_LABELS = [
    ("Accuracy", "Accuracy"),
    ("AUC", "AUC"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("F1", "F1"),
    ("MCC", "MCC"),
]

METRIC_HELP = {
    "Accuracy": "Share of rows predicted correctly.",
    "AUC": "Area under the ROC curve for the positive class (>50K).",
    "Precision": "Proportion of predicted high-income cases that are correct.",
    "Recall": "Proportion of actual high-income cases correctly identified.",
    "F1": "Harmonic mean of precision and recall for the positive class.",
    "MCC": "Matthews correlation coefficient. 1 = perfect, 0 = coin flip, -1 = total disagreement.",
}

# Dark palette styling
BG, PANEL, BORDER = "#0D1117", "#161B22", "#30363D"
TEXT, MUTED = "#E6EDF3", "#8B949E"
AMBER, TEAL = "#F0A050", "#3FB8AF"

# Dark -> brown -> amber -> cream colormap for confusion matrix heatmap
HEATMAP_CMAP = LinearSegmentedColormap.from_list("adult_income", [PANEL, "#4A2F14", AMBER, "#FFE3B8"])

st.set_page_config(
    page_title="Adult Income Classifier Lab",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib parameters to match dark theme
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "grid.color": BORDER, "legend.facecolor": PANEL, "legend.edgecolor": BORDER,
    "font.size": 10,
})

# Custom CSS styling
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
.nav {{
    position: sticky; top: 3.2rem; z-index: 50; display: flex; gap: 8px; flex-wrap: wrap;
    background: {BG}; padding: 8px 0 10px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 6px;
}}
.nav a {{
    text-decoration: none; color: {TEXT}; background: {PANEL}; border: 1px solid {BORDER};
    border-radius: 999px; padding: 5px 16px; font-size: 0.9rem; font-weight: 600;
}}
.nav a:hover {{ border-color: {AMBER}; color: {AMBER}; }}
.section-head {{
    font-size: 1.45rem; font-weight: 800; color: {TEXT}; margin: 28px 0 12px 0; padding-bottom: 6px;
    border-bottom: 2px solid {AMBER}; scroll-margin-top: 7rem;
}}
.github-link {{
    position: fixed; top: 0.7rem; right: 1.25rem; z-index: 1000;
    display: inline-flex; align-items: center; padding: 0.45rem 0.8rem;
    border: 1px solid #46515E; border-radius: 999px; background: #161B22;
    color: #E6EDF3 !important; font-size: 0.82rem; font-weight: 650;
    text-decoration: none !important; box-shadow: 0 2px 10px rgba(0,0,0,0.22);
}}
.github-link:hover {{ border-color: #F0A050; color: #F0A050 !important; background: #1C2430; }}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<a class="github-link" href="https://github.com/yadavalliharika321-dev/MLAssignment2" '
    'target="_blank" rel="noopener noreferrer">↗ GitHub Repository</a>',
    unsafe_allow_html=True,
)

# ============================================================
# LOADING & PREPROCESSING UTILITIES
# ============================================================

@st.cache_resource
def load_models():
    missing = [
        filename
        for filename in MODEL_FILES.values()
        if not (MODEL_DIR / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): "
            + ", ".join(missing)
            + ". Make sure the model .joblib files are inside the model/ folder."
        )
    return {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }


def normalize_input(raw):
    """Normalize UCI Adult categorical values and income labels."""
    df = raw.copy()
    for column in CATEGORICAL_FEATURES + [TARGET]:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].str.rstrip(".")

    for column in CATEGORICAL_FEATURES:
        if column in df.columns:
            df[column] = df[column].replace("?", "Unknown")

    return df


def validate_and_prepare(raw):
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        return None, "The CSV is missing required column(s): " + ", ".join(missing)

    df = normalize_input(raw[required])

    valid_labels = {NEGATIVE_LABEL, POSITIVE_LABEL}
    found_labels = set(df[TARGET].dropna().astype(str))
    unknown = sorted(found_labels - valid_labels)
    if unknown:
        return None, (
            f"The '{TARGET}' column must contain only '{NEGATIVE_LABEL}' "
            f"or '{POSITIVE_LABEL}'. Found: {', '.join(unknown)}"
        )

    if df[TARGET].isna().any():
        return None, f"The '{TARGET}' column contains missing values."

    X = df[FEATURES].copy()

    try:
        X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].apply(
            pd.to_numeric, errors="raise"
        )
    except (ValueError, TypeError):
        return None, "These columns must contain numeric values: " + ", ".join(NUMERIC_FEATURES)

    if X.isna().any().any():
        return None, "Feature columns contain missing values."

    y = (df[TARGET].astype(str) == POSITIVE_LABEL).astype(int)
    if y.nunique() != 2:
        return None, "The uploaded test data must contain both income classes to calculate AUC."

    return (X, y, df), None


def get_scores(model, X):
    """Return continuous probability or decision score for positive income class."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = list(model.classes_)
        positive_index = classes.index(1)
        return probabilities[:, positive_index]

    scores = model.decision_function(X)
    if np.ndim(scores) == 2:
        classes = list(model.classes_)
        positive_index = classes.index(1)
        return scores[:, positive_index]

    if list(model.classes_)[1] == 1:
        return scores
    return -scores


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    scores = get_scores(model, X)

    return {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, scores),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }, predictions, scores


def fmt(v, nd=4):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap=HEATMAP_CMAP)
    ax.set_xticks([0, 1], labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    ax.set_yticks([0, 1], labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    ax.set_xlabel("Predicted income")
    ax.set_ylabel("Actual income")
    thresh = cm.max() * 0.55 if cm.max() else 0
    row_totals = cm.sum(axis=1, keepdims=True)
    row_percentages = np.divide(
        cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0
    ) * 100
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({row_percentages[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color=BG if cm[i, j] > thresh else TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_edgecolor(BORDER)
    fig.tight_layout()
    return fig


def plot_roc(y, scores, auc_val):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    fpr, tpr, _ = roc_curve(y, scores)
    ax.plot(fpr, tpr, color=AMBER, lw=1.8, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def board_frame(results_dict):
    tab = pd.DataFrame([
        {"Model": name, **metrics}
        for name, metrics in results_dict.items()
    ])
    tab = tab.sort_values(["F1", "MCC", "Accuracy"], ascending=False).reset_index(drop=True)
    tab.index = tab.index + 1
    return tab


def render_board(tab, caption):
    best = tab.iloc[0]
    st.success(f"🏆 Winner: **{best['Model']}** - F1 {best['F1']:.4f}, MCC {best['MCC']:.4f}")
    num = [lbl for lbl, _ in METRIC_LABELS]
    st.dataframe(tab.style.format({c: "{:.4f}" for c in num}, na_rep="n/a")
                 .highlight_max(subset=num, props=f"background-color:#4A2F14;color:{AMBER};font-weight:700"),
                 height=min(35 * (len(tab) + 1) + 3, 600))
    st.caption(caption)
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(tab) + 0.9))
    t2 = tab.iloc[::-1]
    bars = ax.barh(t2["Model"], t2["F1"], color=AMBER)
    for b, v in zip(bars, t2["F1"]):
        ax.text(v + 0.001, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=8, color=TEXT)
    lo = max(0.0, float(np.nanmin(t2["F1"])) - 0.03)
    ax.set_xlim(lo, min(1.0, float(np.nanmax(t2["F1"])) + 0.02))
    ax.set_xlabel("F1 score")
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    show(fig)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 💼 How to use")
    st.markdown(
        "1. **Upload** a test CSV in panel ① (or just use the bundled one)\n"
        "2. **Pick a model** from the dropdown in panel ②\n"
        "3. Scroll down (or use the jump bar) for results, leaderboard and predictions")
    st.divider()
    st.markdown("### 📦 Dataset at a glance")
    st.markdown(
        "- UCI **Adult Census Income** dataset\n"
        "- Predicts whether annual income exceeds $50K\n"
        "- 14 predictor features (6 numeric, 8 categorical)\n"
        "- Binary classification classes (`<=50K`, `>50K`)")
    st.divider()
    st.caption("Every model is trained offline as a scikit-learn pipeline; "
               "this app only loads the saved joblib files and never retrains.")

# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
<div class="hero">
  <h1>💼 Adult Income Classifier Lab</h1>
  <p>Explore five trained machine-learning models that predict whether an individual's
  annual income is above $50K. Upload labelled test data, select a model, and inspect
  its performance and predictions.</p>
  <span class="chip">14 features</span><span class="chip">2 income classes</span>
  <span class="chip">5 models</span><span class="chip teal">Saved Joblib pipelines</span>
  <span class="chip teal">no retraining in the app</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PANEL CONTROLS
# ============================================================

c_data, c_model = st.columns([1.15, 1], gap="large")
with c_data:
    with st.container(border=True):
        st.markdown('<div class="step"><span class="num">1</span>Load your test data</div>'
                    '<div class="step-sub">CSV with all 14 feature columns and the income target. '
                    'Test data only - 10 MB max.</div>', unsafe_allow_html=True)
        data_source = st.radio(
            "Test data source",
            ["Bundled test_data.csv", "Upload CSV"],
            horizontal=True,
            label_visibility="collapsed",
        )
        uploaded_file = None
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload labelled test CSV", type=["csv"],
                help="Must include the 14 predictor columns and income target.",
            )
        dl, note = st.columns([1, 1.6])
        if TEST_CSV.is_file():
            dl.download_button("⬇️ Sample test_data.csv", TEST_CSV.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")
            note.caption("No file? The bundled hold-out set is used automatically.")
        else:
            note.caption("Bundled test_data.csv is not present. Upload a labelled CSV.")

with c_model:
    with st.container(border=True):
        st.markdown('<div class="step"><span class="num">2</span>Pick a model</div>'
                    '<div class="step-sub">Choose one of the five fitted classifiers.</div>',
                    unsafe_allow_html=True)
        chosen = st.selectbox(
            "Model", list(MODEL_FILES.keys()), label_visibility="collapsed")
        st.markdown(f'<div class="blurb">{MODEL_BLURBS.get(chosen, "")}</div>', unsafe_allow_html=True)

# ============================================================
# DATA VALIDATION & EXECUTION
# ============================================================

if data_source == "Upload CSV":
    if uploaded_file is None:
        raw_bytes = TEST_CSV.read_bytes() if TEST_CSV.is_file() else b""
        source_name = "the bundled <b>test_data.csv</b>"
    else:
        raw_bytes = uploaded_file.getvalue()
        source_name = f"your file <b>{uploaded_file.name}</b>"
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("The uploaded file exceeds the 10 MB limit.")
            st.stop()
else:
    raw_bytes = TEST_CSV.read_bytes() if TEST_CSV.is_file() else b""
    source_name = "the bundled <b>test_data.csv</b>"

if not raw_bytes:
    st.error("No test data found. Please upload a valid CSV file.")
    st.stop()

try:
    df_in = pd.read_csv(io.BytesIO(raw_bytes), skipinitialspace=True)
except Exception as e:
    st.error(f"Couldn't read that as a CSV: {e}")
    st.stop()

prepared, err = validate_and_prepare(df_in)
if err:
    st.error(err)
    st.stop()

X, y, clean_data = prepared

try:
    models = load_models()
except Exception as e:
    st.error(str(e))
    st.stop()

st.markdown(f'<div class="status">Evaluating <b>{chosen}</b> on {source_name} · '
            f'<b>{len(X):,}</b> rows · <b>{len(FEATURES)}</b> features · '
            f'<b>{(y == 0).sum():,}</b> actual {NEGATIVE_LABEL} · '
            f'<b>{(y == 1).sum():,}</b> actual {POSITIVE_LABEL}</div>', unsafe_allow_html=True)

all_results = {}
all_predictions = {}
all_scores = {}

with st.spinner("Scoring five saved models..."):
    for m_name, model_obj in models.items():
        m_res, m_pred, m_scores = evaluate_model(model_obj, X, y)
        all_results[m_name] = m_res
        all_predictions[m_name] = m_pred
        all_scores[m_name] = m_scores

selected_metrics = all_results[chosen]
selected_pred = all_predictions[chosen]
selected_scores = all_scores[chosen]

# Dynamic Navigation Sticky Bar
SECTIONS = [("results", "📊 Model results"), ("leaderboard", "🏆 Leaderboard"),
            ("predictions", "🔮 Predictions"), ("about", "ℹ️ About")]
st.markdown('<div class="nav">' + "".join(f'<a href="#{a}">{t}</a>' for a, t in SECTIONS) + "</div>",
            unsafe_allow_html=True)


def section(anchor):
    title = dict(SECTIONS)[anchor]
    st.markdown(f'<div class="section-head" id="{anchor}">{title}</div>', unsafe_allow_html=True)

# ============================================================
# RESULTS SECTION
# ============================================================

with st.container():
    section("results")
    cols = st.columns(6)
    for c, (key, label) in zip(cols, METRIC_LABELS):
        c.metric(label, fmt(selected_metrics[key]), help=METRIC_HELP[key], border=True)
    st.caption("Precision, Recall, and F1 are for the positive class (>50K). "
               "AUC uses predicted probabilities or decision scores.")

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Confusion matrix** · rows = actual, columns = predicted")
            cm = confusion_matrix(y, selected_pred, labels=[0, 1])
            show(plot_confusion(cm))
    with right:
        with st.container(border=True):
            st.markdown("**ROC curve** · positive class = >50K")
            show(plot_roc(y, selected_scores, selected_metrics["AUC"]))

    st.markdown("#### Classification report")
    rep_dict = classification_report(
        y, selected_pred, labels=[0, 1],
        target_names=[NEGATIVE_LABEL, POSITIVE_LABEL],
        output_dict=True, zero_division=0,
    )
    rep = pd.DataFrame(rep_dict).T
    st.dataframe(rep.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}"}))

    with st.expander("Model description and evaluation details", expanded=True):
        st.write(MODEL_BLURBS[chosen])
        st.write("The saved preprocessing and classifier pipeline is loaded from the model/ directory.")

# ============================================================
# LEADERBOARD SECTION
# ============================================================

with st.container():
    section("leaderboard")
    st.markdown("**All 5 models evaluated on the test set**")
    render_board(board_frame(all_results), f"Evaluated live on {len(X):,} test rows.")

# ============================================================
# PREDICTIONS SECTION
# ============================================================

with st.container():
    section("predictions")
    mistakes = selected_pred != y.to_numpy()
    correct_count = int((~mistakes).sum())
    n_wrong = int(mistakes.sum())

    a, b, c = st.columns([1, 1, 2])
    a.metric("Rows evaluated", f"{len(y):,}", border=True)
    b.metric("Mistakes", f"{n_wrong:,}", border=True)
    with c:
        only_wrong = st.toggle("Show only the mistakes")

    out = clean_data[FEATURES + [TARGET]].copy()
    out["predicted_income"] = np.where(selected_pred == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
    out["correct"] = np.where(mistakes, "No", "Yes")
    out.insert(0, "row", np.arange(1, len(out) + 1))

    if only_wrong:
        out = out[~out["correct"].eq("Yes")]

    st.dataframe(out, height=460, hide_index=True)
    st.download_button("⬇️ Download these predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{chosen.lower().replace(' ', '_')}.csv", mime="text/csv")

# ============================================================
# ABOUT SECTION
# ============================================================

with st.container():
    section("about")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### About the data")
        st.markdown(
            """
**UCI Adult Census Income dataset**: predict whether annual income exceeds $50K/yr based on census data.

* **Target classes**: `<=50K` and `>50K`
* **Features**: 14 attributes (6 numeric, 8 categorical)
""")
        feature_table = pd.DataFrame({
            "Feature": FEATURES,
            "Type": ["Numeric" if f in NUMERIC_FEATURES else "Categorical" for f in FEATURES],
        })
        st.dataframe(feature_table, hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### How the metrics are computed")
        st.markdown(
            """
* **Accuracy** - share of rows predicted right.
* **AUC** - area under the ROC curve for the positive class (>50K).
* **Precision / Recall / F1** - calculated for the positive class (>50K).
* **MCC** - Matthews correlation coefficient, 1 = perfect, 0 = no better than chance.
""")
    st.caption("Trained classifiers loaded from the model/ directory. No retraining is performed inside the app.")
