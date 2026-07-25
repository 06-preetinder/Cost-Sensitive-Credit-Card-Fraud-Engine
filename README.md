<div align="center">

# 💳 Cost-Sensitive Credit Card Fraud Engine
### *Translating Machine Learning Probabilities into Direct Dollar Business Metrics*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 📌 Executive Summary

Predictive accuracy in fraud detection is a trap. In highly imbalanced datasets (~0.17% fraud), a model predicting "No Fraud" 100% of the time achieves **99.83% accuracy**—while letting **millions of dollars in fraud slide through**.

This engine abandons generic metrics (Accuracy, ROC-AUC) in favor of **Cost-Sensitive Learning**. It treats fraud detection as an economic optimization problem, balancing the financial impact of missed fraud against the operational costs of customer friction and manual review.

> 💰 **Key Highlight:** By sweeping decision thresholds ($\tau \in [0.01, 0.99]$) against a custom financial loss function, this engine **drastically reduces total monetary losses** compared to standard ML default thresholds ($0.50$).

---

## ⚡ Business Cost Architecture

Rather than treating all classification errors equally, this model evaluates predictions against real-world bank operating expenses:

| Outcome | Decision | Operational Real-World Cost |
| :--- | :--- | :--- |
| **False Negative (FN)** | Missed Fraud | **Full Transaction Amount ($)** — High Loss |
| **False Positive (FP)** | False Alarm | **$10.00** — Fixed Verification/Friction Cost |
| **True Positive (TP)** | Fraud Stopped | **$10.00** — Fixed Verification Cost |
| **True Negative (TN)** | Clean Transaction | **$0.00** — Approved |
