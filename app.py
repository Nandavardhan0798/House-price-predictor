import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# ----------------------------
# Load model and deployment info
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    categorical_columns = joblib.load("categorical_columns.joblib")
    with open("deployment_info.json") as f:
        deployment_info = json.load(f)
    shift = deployment_info["shift"]
    return model, feature_columns, categorical_columns, shift

model, feature_columns, categorical_columns, shift = load_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🏠 House Price Prediction")

uploaded_file = st.file_uploader("Upload CSV with features only", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Input Data Preview")
    st.dataframe(data.head())

    # Ensure correct columns and order
    try:
        data = data[feature_columns]
    except KeyError:
        st.error("Uploaded CSV does not contain the required features.")
        st.stop()

    # Convert categorical columns
    for col in categorical_columns:
        data[col] = data[col].astype("category")

    # Predict using trained model
    y_pred_log = model.predict(data)
    y_pred = np.expm1(y_pred_log) - shift

    # Show predictions
    data["Predicted_Price"] = y_pred
    st.subheader("Predictions")
    st.dataframe(data)

    # Download predictions as CSV
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

    # Feature Importance
    st.subheader("Top Feature Importances")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    st.dataframe(feat_df.head(15))

    # Plot top 15 features
    plt.figure(figsize=(10,6))
    plt.barh(feat_df["Feature"][:15], feat_df["Importance"][:15])
    plt.gca().invert_yaxis()
    plt.title("Top 15 Feature Importances")
    st.pyplot(plt)
