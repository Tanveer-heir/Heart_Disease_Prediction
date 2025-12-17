# Heart_Disease_Prediction
# 🫀 Heart Disease Risk Prediction using Machine Learning

This project implements an **end-to-end, interpretable machine learning pipeline** to predict heart disease risk using clinical and physiological data.  
The emphasis is on **data understanding, transparent modeling, and robust evaluation**, rather than black-box performance.

---

## 📌 Project Overview

- **Objective:** Predict whether a patient has heart disease  
- **Dataset:** UCI Heart Disease Dataset (Cleveland)  
- **Approach:**  
  Exploratory Data Analysis → Preprocessing → Model Training → Cross-Validation → Model Comparison → Final Model Selection  
- **Final Model:** Default Logistic Regression (Pipeline-based)

---

## 📊 Dataset Description

- **Samples:** ~1,000 patients  
- **Features:** 13 clinical & physiological attributes  
- **Target Variable:**
  - `0` → No heart disease
  - `1` → Heart disease

### Data Quality
- No missing values
- Medically realistic value ranges
- Balanced target distribution (~54% disease, ~46% no disease)

---

## 🧠 Feature Understanding

Features are grouped based on their semantic meaning:

### Numerical Features (Risk Intensity)
- `age`
- `trestbps`
- `chol`
- `thalach`
- `oldpeak`

These represent continuous physiological measurements.  
Outliers are retained, as they correspond to genuine medical extremes.

### Categorical Features (Diagnostic Indicators)
- `sex`
- `cp`
- `fbs`
- `restecg`
- `exang`
- `slope`
- `ca`
- `thal`

These act as clinical flags and decision rules.

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed step-by-step to build a clear mental model before modeling:

- Distribution analysis using histograms and boxplots
- Class dominance analysis for categorical features
- Feature vs target analysis:
  - Boxplots for numerical features
  - Count plots for categorical features
- Outlier analysis (no removal)
- Correlation analysis to assess feature relationships

### Key Insight
Heart disease risk emerges from the **interaction of diagnostic categorical indicators and physiological stress-response measurements**, rather than from any single feature alone.

---

## ⚙️ Preprocessing Pipeline

A reproducible preprocessing pipeline was built using `scikit-learn`:

- **Numerical features:** StandardScaler
- **Categorical features:** OneHotEncoder (`drop='first'`)
- **Tooling:** ColumnTransformer + Pipeline

This ensures:
- No data leakage
- Consistent transformations
- Deployment-ready workflow

---

## 🤖 Models Used

### Logistic Regression (Default)
- Used as an interpretable baseline
- Strong linear clinical signals captured effectively
- Feature influence interpreted using coefficients and odds ratios

### Random Forest (Default)
- Used to assess non-linear modeling capability
- No hyperparameter tuning applied (fair baseline comparison)

---
## 📈 Cross-Validation Results (5-Fold Stratified)

### Default Logistic Regression
Accuracy : 0.844 ± 0.034
Precision : 0.846 ± 0.045
Recall : 0.878 ± 0.086
F1 Score : 0.858 ± 0.038
ROC-AUC : 0.917 ± 0.019

### Default Random Forest
Accuracy : 0.778 ± 0.041
Precision : 0.799 ± 0.042
Recall : 0.799 ± 0.106
F1 Score : 0.793 ± 0.053
ROC-AUC : 0.893 ± 0.017


---

## 🏆 Final Model Selection

The **default Logistic Regression model** was selected as the final model because it:

- Outperformed Random Forest across all evaluation metrics
- Achieved high recall, critical for medical screening
- Demonstrated lower variance and better stability
- Provided strong interpretability for healthcare use

---

## 🔎 Model Interpretability Summary

- **Strong predictors:** `cp`, `ca`, `thal`, `oldpeak`, `exang`, `thalach`
- **Moderate contributors:** `age`, `sex`, `slope`
- **Weak individually but useful in combination:** `chol`, `trestbps`, `fbs`, `restecg`

The model aligns well with medical intuition and EDA findings.

---

## 💾 Model Saving

The complete pipeline (preprocessing + classifier) is saved using `joblib`:

```python
joblib.dump(final_model, "models/final_logistic_regression_model.pkl")


