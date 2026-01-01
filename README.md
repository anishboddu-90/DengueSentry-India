# DengueSentry-India 🦟📊
**An Interpretable Early Warning System (EWS) for Dengue Outbreak Prediction.**

[![Python 3.10.19](https://img.shields.io/badge/python-3.10.19-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
**DengueSentry-India** is a hybrid computational framework designed to predict statistically significant dengue outbreaks. This project bridges the gap between traditional epidemiology and Machine Learning by using the **Farrington Flexible Algorithm** to define outbreaks and **Explainable AI (XAI)** to justify predictions.

**Current Status:** Phase 1 (Benchmark Validation on San Juan dataset) is complete. Phase 2 (Delhi/Mumbai deployment) is in development.

---

## 🚀 Technical Architecture
The system operates on a dual-layer logic:
1.  **The Statistical Baseline:** Uses the **Farrington Flexible Algorithm** (via R surveillance) to identify historical "abnormality" thresholds.
2.  **The Predictive Model:** A **Decision Tree** pipeline (transitioning to Random Forest) trained on multi-modal environmental data (NASA POWER) to predict "Spike Risk."
3.  **Interpretability:** **SHAP (SHapley Additive exPlanations)** is used to attribute risks to specific climate drivers, such as 4-week lagged precipitation.

---

## 🛠️ Installation & Setup
This project requires both **Python** and **R**. We recommend using **Conda** to manage the cross-language dependencies (especially for rpy2).

### Option A: Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate dengue_project
```

### Option B: Using Pip
# Ensure R 4.0+ is installed on your system first
```bash
pip install -r requirements.txt
```
---

## 📂 Repository Structure

* **data/**: Weekly epidemiological records and merged climate variables.
* **notebooks/**: Phase 1 Validation Notebook (San Juan benchmark).
* **src/**: Core logic including the Python-to-R bridge (rpy2).
* **environment.yml**: Full Conda environment specification (Python 3.10.19).
* **requirements.txt**: Minimal Python package list.

---

## 📈 Key Methodology: The Farrington-ML Hybrid
Unlike standard regression models, this project uses the Farrington algorithm to define the **Target Variable**. 

* **The Problem:** Raw case counts don't tell you if a true outbreak is occurring, only if cases are high relative to the previous week.
* **The Solution:** We calculate a "Spike" as any week where cases exceed the **95% Upper Bound** of predicted historical seasonality. The ML model learns to predict this "Alarm" status with a **7-day lead time**.

---

## 🔬 Phase 1 Results (San Juan Benchmark)
* **F1-Score:** 0.614
* **Lead Time:** 7 Days (Fixed)
* **Top Predictors:** Case Momentum, Temperature (Lag 1), Precipitation (Lag 4).
* **Key Finding:** Identified **"Momentum Dominance,"** where the model relies on recent trends. Phase 2 will implement 8–12 week lags to better capture the mosquito life cycle.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.