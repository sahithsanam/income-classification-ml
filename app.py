import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Income Classification App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Income Classification ML Application")
st.markdown("download test dataset and evaluate selected machine learning model.")

# -------------------------------
# Download Sample Test Data
# -------------------------------
st.subheader("📥 Download Sample Test Data")

# Load original dataset
try:
    sample_df = pd.read_csv("adult.csv").sample(200, random_state=42)
    csv = sample_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Sample Test CSV",
        data=csv,
        file_name="sample_test_data.csv",
        mime="text/csv"
    )
except:
    st.info("Sample dataset not found. Please upload test CSV manually.")
# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV only)",
    type=["csv"]
)

# -------------------------------
# Model Selection
# -------------------------------
model_option = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

model_files = {
    "Logistic Regression": "model/logistic_model.pkl",
    "Decision Tree": "model/dt_model.pkl",
    "KNN": "model/knn_model.pkl",
    "Naive Bayes": "model/nb_model.pkl",
    "Random Forest": "model/rf_model.pkl",
    "XGBoost": "model/xgb_model.pkl"
}

# -------------------------------
# Main Logic
# -------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "income" not in df.columns:
        st.error("Uploaded dataset must contain 'income' column.")
    else:
        # Load preprocessing
        le_dict = joblib.load("model/label_encoders.pkl")
        scaler = joblib.load("model/scaler.pkl")

        # Encoding
        for col in df.select_dtypes(include='object').columns:
            if col in le_dict:
                df[col] = le_dict[col].transform(df[col])

        X = df.drop("income", axis=1)
        y = df["income"]

        # Scaling
        X = scaler.transform(X)

        model = joblib.load(model_files[model_option])
        y_pred = model.predict(X)

        # -------------------------------
        # Metrics
        # -------------------------------
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.subheader("📈 Model Performance")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("AUC", f"{auc:.3f}")
        col3.metric("Precision", f"{prec:.3f}")
        col4.metric("Recall", f"{rec:.3f}")
        col5.metric("F1 Score", f"{f1:.3f}")
        col6.metric("MCC", f"{mcc:.3f}")

       # -------------------------------
        # Ultra Small Confusion Matrix
        # -------------------------------
        st.subheader("🔎 Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        # Create very small figure
        fig, ax = plt.subplots(figsize=(2, 1.6))   # MUCH smaller

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap="Blues",
            cbar=False,
            linewidths=1,
            linecolor='black',
            square=True,
            annot_kws={"size": 8},   # smaller numbers
            ax=ax
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(["0", "1"], fontsize=8)
        ax.set_yticklabels(["0", "1"], fontsize=8)

        # Center it using columns
        col1, col2, col3 = st.columns([2,1,2])
        with col2:
            st.pyplot(fig, use_container_width=False)
        # -------------------------------
        # Bordered Metric Summary
        # -------------------------------
        st.markdown("### 📌 Model Summary")

        st.markdown(
            f"""
            <div style="
                border:2px solid #4CAF50;
                border-radius:10px;
                padding:15px;
                background-color:#f9f9f9;
            ">
                <b>Selected Model:</b> {model_option} <br><br>
                <b>Accuracy:</b> {acc:.3f} &nbsp;&nbsp;
                <b>AUC:</b> {auc:.3f} &nbsp;&nbsp;
                <b>F1 Score:</b> {f1:.3f} <br><br>
                <b>Precision:</b> {prec:.3f} &nbsp;&nbsp;
                <b>Recall:</b> {rec:.3f} &nbsp;&nbsp;
                <b>MCC:</b> {mcc:.3f}
            </div>
            """,
            unsafe_allow_html=True
        )
        # -------------------------------
        # Styled Classification Report
        # -------------------------------
        st.subheader("📄 Classification Report")

        report_dict = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        styled_report = report_df.style \
            .set_table_styles([
                {'selector': 'th',
                'props': [('border', '1px solid black'),
                        ('background-color', '#f0f2f6'),
                        ('padding', '6px')]},
                {'selector': 'td',
                'props': [('border', '1px solid black'),
                        ('padding', '6px')]}
            ]) \
            .format(precision=2)

        st.dataframe(styled_report, use_container_width=True)