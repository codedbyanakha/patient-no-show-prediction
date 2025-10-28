# Patient No-Show Prediction Project (MSc Internship, July–September 2025)
Welcome to my project repository for the No-Show Prediction mini-project, conducted as part of my MSc Data Science program. The goal was to predict whether a patient would attend or miss their medical appointment using machine learning, while ensuring fairness, interpretability, and deployability through a Streamlit web app.

**LIVE APP:** https://patient-no-show-prediction-nacmezuzsqtpudkfzjklki.streamlit.app/

📋 PROJECT OVERVIEW

Role: Data Science Intern
Duration: July 2025 – September 2025
Mode: Remote | Research-Based
Dataset: Medical Appointment No-Show Dataset (Kaggle)

🚀TASK STRUCTURE
🧩Level 1: Data Preprocessing & Exploration
| **Task** | **Title**                 | **Description**                                                                                          |
| -------- | ------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1        | Data Cleaning             | Removed irrelevant IDs, handled missing and invalid values, standardized date columns.                   |
| 2        | Feature Engineering       | Derived new features: `waiting_days` (gap between scheduling and appointment) and `appointment_weekday`. |
| 3        | Encoding & Scaling        | Applied one-hot encoding to categorical variables and standardized numerical features.                   |
| 4        | Exploratory Data Analysis | Visualized age, gender, and waiting days against no-show patterns to identify behavioral insights.       |

🤖Level 2: Model Development & Evaluation
| **Task** | **Title**       | **Description**                                                                   |
| -------- | --------------- | --------------------------------------------------------------------------------- |
| 1        | Model Training  | Built Logistic Regression, Random Forest, and XGBoost classifiers.                |
| 2        | Evaluation      | Compared models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics. |
| 3        | Model Selection | Chose **XGBoost** as the final model due to superior overall performance.         |
| 4        | Artifact Saving | Saved model, scaler, feature metadata, and neighborhood encoding for deployment.  |

⚖️Level 3: Fairness & Explainability
| **Task** | **Title**           | **Description**                                                                                            |
| -------- | ------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1        | Fairness Assessment | Checked for gender and age bias using **Statistical Parity Difference** and **Equalized Odds Difference**. |
| 2        | SHAP Analysis       | Explained global and local feature importance contributing to no-show behavior.                            |
| 3        | LIME Analysis       | Generated local explanations for individual patient predictions.                                           |
| 4        | Comparison          | Found **SHAP** to be more reliable and consistent than LIME for interpretation.                            |

🤖Level 4: Model Deployment – Streamlit App
| **Task** | **Title**       | **Description**                                                                                                     |
| -------- | --------------- | ------------------------------------------------------------------------------------------------------------------- |
| 1        | App Development | Built a **Streamlit dashboard** for real-time no-show prediction using the trained XGBoost model.                   |
| 2        | Features        | Supports **single patient input** and **batch CSV uploads**, with automatic preprocessing and result visualization. |
| 3        | User Output     | Displays prediction outcomes, model confidence, and interpretable summary charts.                                   |

🧰TOOLS USED
| **Category**            | **Tools / Libraries**     |
| ----------------------- | ------------------------- |
| Programming             | Python                    |
| Data Processing         | Pandas, NumPy             |
| Modeling                | Scikit-learn, XGBoost     |
| Visualization           | Matplotlib, Seaborn       |
| Explainability          | SHAP, LIME                |
| App Development         | Streamlit                 |
| Development Environment | Google Colab, VS Code |

🎯 OUTCOME :

This project delivered a complete end-to-end ML solution for predicting patient appointment no-shows — from preprocessing, modeling, fairness assessment, and explainability to deployment via a user-friendly Streamlit app. The system enables healthcare providers to proactively reduce no-shows through data-driven decision-making and transparent AI insights.
