# DengueSentry-India 🦟📊
**An interpretable Early Warning System (EWS) for dengue outbreak prediction in India.**

[![Python 3.10.19](https://img.shields.io/badge/python-3.10.19-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
**DengueSentry-India** is a hybrid framework that combines statistical surveillance with machine learning to forecast dengue outbreak risk. We use the **Farrington Flexible Algorithm** (R surveillance) to define outbreak labels and a **Random Forest** classifier to predict next‑week spikes using weather, momentum, and lag features. **SHAP** is used for interpretability.

**Current Status:** Phase 2 validation is focused on Indian data (Kerala + Karnataka, 2023–2025) with synthetic historical augmentation to satisfy Farrington baselines.

---

## 🚀 Technical Architecture
The system operates on a dual‑layer pipeline:
1. **Statistical Baseline:** **Farrington Flexible** identifies outbreak alarms from weekly cases.
2. **Predictive Model:** **Random Forest** predicts next‑week spike risk using weather, lags, momentum, and seasonality features.
3. **Interpretability:** **SHAP** validates biological drivers (e.g., 4–8 week rainfall lags).

---

## 🧭 Project Evolution (Phase 1 → Phase 2)
**Phase 1: Benchmark Foundation (San Juan)**
* **Data:** DengAI San Juan benchmark.
* **Outbreak definition:** 2‑sigma anomaly vs. 52‑week rolling mean.
* **Model:** Decision Tree with a slim feature set.
* **Baseline:** Momentum‑only heuristic for comparison.
* **Key finding:** “Momentum trap” and limited independent value from short precipitation lags.
* **Notebook:** [notebooks/01_exploratory_analysis.ipynb](notebooks/01_exploratory_analysis.ipynb)

**Phase 2: India Validation (Kerala + Karnataka)**
* **Data:** Digitized weekly cases from Kerala and Karnataka (2023–2025) + synthetic history extension.
* **Outbreak definition:** Farrington Flexible (R surveillance) with sensitivity tuning.
* **Model:** Random Forest with class imbalance handling and threshold optimization.
* **Signals:** NASA POWER weather, momentum, seasonal features, optional Google Trends.
* **Goal:** Improve recall while preserving biological interpretability.
* **Notebook:** [notebooks/03_phase2_validation.ipynb](notebooks/03_phase2_validation.ipynb)

---

## 🛠️ Installation & Setup
This project requires both **Python** and **R**. We recommend using **Conda** to manage the cross-language dependencies (especially for rpy2).

### Option A: Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate dengue_project
```

### Option B: Using Pip
### Ensure R 4.0+ is installed on your system first
```bash
pip install -r requirements.txt
```
---

## 📂 Repository Structure

* **data/**: Weekly epidemiological records and merged climate variables.
* **notebooks/**: Phase 1 benchmark, Indian data acquisition, and Phase 2 validation notebooks.
* **src/**: Core pipeline (preprocessing, weather, outbreak detection, modeling, synthetic augmentation).
* **environment.yml**: Full Conda environment specification (Python 3.10.19 + R).
* **requirements.txt**: Minimal Python package list.

---

## 📈 Key Methodology: The Farrington‑ML Hybrid
Unlike standard regression models, this project uses Farrington to define the **target variable**.

* **The Problem:** Raw case counts alone don’t indicate true outbreaks.
* **The Solution:** A “spike” is defined when cases exceed the modelled seasonal upper bound. The ML model predicts that alarm with a **7‑day lead time**.

---

## 🔬 Phase 2 Focus (India Validation)
* **Regions:** Kerala + Karnataka (2023–2025) with synthetic historical extension.
* **Signals:** NASA POWER weather + momentum + lag features.
* **Goal:** Achieve strong recall on outbreak detection while preserving biological interpretability.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.