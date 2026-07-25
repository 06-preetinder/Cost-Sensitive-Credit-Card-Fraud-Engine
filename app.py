import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(
    page_title="Cost-Sensitive Fraud Engine", page_icon="💳", layout="wide"
)

st.title("💳 Cost-Sensitive Credit Card Fraud Detection Engine")
st.markdown(
    """
This app evaluates fraud risks based on **Direct Dollar Costs**:
* **Missed Fraud (False Negative):** Costs the full transaction amount.
* **False Alarm (False Positive):** Costs a $10 operational verification fee.
"""
)


# Load saved model artifact
@st.cache_resource
def load_artifact():
  return joblib.load("fraud_detection_model.joblib")


try:
  artifact = load_artifact()
  pipeline = artifact["pipeline"]
  optimal_threshold = artifact["best_threshold"]
  feature_names = artifact["feature_names"]
except FileNotFoundError:
  st.error('❌ "fraud_detection_model.joblib" not found in current folder.')
  st.stop()

# Sidebar configuration
st.sidebar.header("⚙️ Threshold Tuning")
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.01,
    max_value=0.99,
    value=float(optimal_threshold),
    step=0.01,
)

if abs(threshold - optimal_threshold) < 0.001:
  st.sidebar.success(
      f"🎯 Currently using Optimal Business Threshold ({optimal_threshold:.2f})"
  )

# Interactive Simulation Inputs
st.subheader("🧪 Single Transaction Risk Simulator")

col1, col2 = st.columns(2)
with col1:
  amount = st.number_input(
      "Transaction Amount ($)", min_value=1.0, max_value=10000.0, value=250.00
  )
with col2:
  v14 = st.slider(
      "V14 PCA Anomaly Level (Key Fraud Indicator)", -20.0, 5.0, -5.0
  )

# Create input vector matching trained feature layout
input_data = pd.DataFrame(
    np.zeros((1, len(feature_names))), columns=feature_names
)
input_data["Amount"] = amount
input_data["V14"] = v14

# Inference
fraud_prob = pipeline.predict_proba(input_data)[0, 1]
is_flagged = fraud_prob >= threshold

st.divider()

# Output Display
res1, res2, res3 = st.columns(3)
res1.metric("Predicted Fraud Risk", f"{fraud_prob * 100:.2f}%")
res2.metric(
    "Engine Decision",
    "🚨 BLOCK / FLAG" if is_flagged else "✅ APPROVE",
    delta_color="inverse",
)

if is_flagged:
  res3.metric("Estimated Operational Fee", "$10.00")
else:
  res3.metric("Potential Uncaptured Loss", f"${amount:.2f}")

st.info(
    f"Transactions with calculated fraud probability **≥ {threshold:.2f}** are"
    " flagged for manual review."
)