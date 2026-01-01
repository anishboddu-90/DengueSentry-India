# DengueSentry-India 🦟📊
**An Interpretable Early Warning System (EWS) for Dengue Outbreak Prediction in Urban India**

## 📌 Project Overview
DengueSentry-India is a hybrid computational framework designed to predict statistically significant dengue outbreaks in Delhi and Mumbai. By combining traditional epidemiological surveillance algorithms with Explainable AI (XAI), this project aims to provide actionable insights for public health resource allocation.

## 🚀 Technical Architecture
The system operates on a dual-layer logic:
1. **The Statistical Baseline:** Uses the **Farrington Flexible Algorithm** (via R `surveillance`) to identify historical "abnormality" thresholds.
2. **The Predictive Model:** A **Random Forest Classifier** trained on multi-modal environmental data to predict "Alarm" status.
3. **Interpretability:** **SHAP (SHapley Additive exPlanations)** is used to attribute outbreak risks to specific climate drivers (e.g., 3-week lagged precipitation).

## 📂 Repository Structure
* `data/`: Weekly epidemiological records and merged climate variables (IMD/NASA POWER).
* `notebooks/`: Exploratory Data Analysis (EDA) and Model Training.
* `src/`: Core logic including the Python-to-R bridge (`rpy2`) and feature engineering scripts.
* `validation/`: Verification against the IDSP Outbreak Registry (2015-2024). (Not Currently Present)

## 🛠️ Requirements & Installation
- Python 3.9+
- R 4.0+ (with `surveillance` package)
- `pip install rpy2 scikit-learn pandas shap matplotlib numpy seaborn`

## 📈 Key Methodology: The Farrington-ML Hybrid
Unlike standard regression models, this project uses the Farrington algorithm to define the **Target Variable**, ensuring that the model learns to predict "Outbreaks" rather than just raw case counts.
