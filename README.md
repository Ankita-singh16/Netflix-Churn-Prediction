# Netflix-Churn-Prediction
Netflix Customer Churn Prediction using XGBoost &amp; Streamlit — Predicts whether a customer will cancel their subscription based on viewing behavior and account details.

A Machine Learning project that predicts whether a Netflix customer 
will churn (cancel subscription) based on their viewing behavior, 
account details, and engagement patterns.

Problem Statement
Customer churn is one of the biggest challenges for streaming services. 
This project builds a predictive model to identify customers who are 
likely to cancel their subscription — enabling the business to take 
proactive retention actions.

Dataset
- 5000 customer records
- 14 features including watch hours, login frequency, 
  subscription type, device, region, and more
- Target variable: `churned` (0 = stayed, 1 = churned)

  Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Joblib

  ML Pipeline
1. Data Loading & Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
   - `is_inactive` — user inactive for 30+ days
   - `fee_per_profile` — cost per profile
   - `watch_intensity` — engagement quality score
4. Label Encoding & One-Hot Encoding
5. Train Test Split (80/20)
6. Model Training with XGBoost
7. Model Evaluation (Accuracy, F1, AUC-ROC, Confusion Matrix)
8. Feature Importance with SHAP
9. Streamlit Web App Deployment

    odel Performance
| Metric | Score |
|--------|-------|
| Accuracy | ~82% |
| AUC-ROC | ~0.89 |
| F1 Score | ~0.81 |

Author
Ankita Kumari
- LinkedIn: https://www.linkedin.com/in/ankita-kumari-24a538289/
- GitHub: https://github.com/Ankita-singh16

