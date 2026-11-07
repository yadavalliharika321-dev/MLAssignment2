    for column, metric in zip(leader_columns, METRICS[:3]):
        column.metric(metric, leaders[metric])
    leader_columns = st.columns(3)
    for column, metric in zip(leader_columns, METRICS[3:]):
        column.metric(metric, leaders[metric])

    csv_bytes = leaderboard.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download leaderboard CSV",
        data=csv_bytes,
        file_name="adult_income_model_leaderboard.csv",
        mime="text/csv",
    )

# ============================================================
# PREDICTIONS
# ============================================================

with predictions_tab:
    st.subheader("🔮 Predictions")
    mistakes = selected_predictions != y.to_numpy()
    correct_count = int((~mistakes).sum())
    mistake_count = int(mistakes.sum())

    prediction_columns = st.columns(3)
    prediction_columns[0].metric("Rows evaluated", f"{len(y):,}")
    prediction_columns[1].metric("Correct predictions", f"{correct_count:,}")
    prediction_columns[2].metric("Mistakes", f"{mistake_count:,}")

    show_mistakes = st.checkbox("Show only the mistakes")
    prediction_view = clean_data[FEATURES + [TARGET]].copy()
    prediction_view["Predicted income"] = np.where(
        selected_predictions == 1, POSITIVE_LABEL, NEGATIVE_LABEL
    )
    prediction_view["Correct"] = np.where(mistakes, "No", "Yes")
    prediction_view.insert(0, "Row", np.arange(1, len(prediction_view) + 1))

    if show_mistakes:
        prediction_view = prediction_view.loc[mistakes]

    st.dataframe(
        prediction_view,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download predictions CSV",
        data=prediction_view.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )

# ============================================================
# ABOUT
# ============================================================

with about_tab:
    st.subheader("ℹ️ About the project")

    st.markdown(
        """
        **Problem statement**

        Predict whether an individual's annual income is greater than $50K
        using demographic and employment-related attributes from the UCI
        Adult Census Income dataset.

        **Target classes**
        - `<=50K` — income is at most $50K
        - `>50K` — income is greater than $50K

        **Input features**
        """
    )

    feature_table = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Type": [
                "Numeric" if feature in NUMERIC_FEATURES else "Categorical"
                for feature in FEATURES
            ],
        }
    )
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Models**
        1. Logistic Regression
        2. Decision Tree
        3. K-Nearest Neighbors
        4. Gaussian Naive Bayes
        5. Linear Support Vector Machine

        **Evaluation metrics**
        - Accuracy: proportion of correct predictions.
        - AUC: ability to rank positive cases above negative cases.
        - Precision: proportion of predicted high-income cases that are correct.
        - Recall: proportion of actual high-income cases correctly identified.
        - F1: harmonic mean of precision and recall.
        - MCC: balanced measure based on all four confusion-matrix outcomes.

        **Application behaviour**

        All models are trained offline in the notebook and saved as Joblib
        pipelines. This application loads those files and evaluates them on
        the selected labelled test CSV. It does not train or retrain models.
        """
    )

    st.caption(
        "Project: Adult Income Classification · Dataset: UCI Adult Census Income"
    )
