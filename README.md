# 📘 Student Performance Prediction Using Explainable and Fair Machine Learning
### *Final Year Research Project — BSc (Hons) Computer Science (Data Science Pathway)*  
### **St Mary’s University, Twickenham, London**

---

## 📋 Overview
This project investigates how machine learning models can be used to predict secondary school student performance and, crucially, how **Explainable Artificial Intelligence (XAI)** can make these predictions **interpretable, transparent, and fair** across different student subgroups (e.g., gender, parental education).

Unlike existing research that focuses mainly on predictive accuracy, this study aims to produce **transparent and fair** machine learning insights that can support educators in understanding *why* predictions occur and whether they behave equitably.

This repository contains:
- Data preprocessing scripts  
- Machine learning model implementations  
- SHAP & LIME explainability analyses  
- Fairness evaluation tools  
- Visualisations and experiment results  
- Supporting documentation and dissertation artefacts  

---

## 🎯 Research Aim
To develop an interpretable and fair machine learning framework for predicting student academic performance, using Explainable AI techniques (SHAP and LIME) and subgroup fairness evaluation.

---


---

## 💾 Dataset
**UCI Student Performance Dataset**  
- Publicly available and anonymised  
- Two datasets: `student-mat.csv` and `student-por.csv`  
Link: https://archive.ics.uci.edu/dataset/320/student+performance

*Ethics:* No identifiable information. Does not require formal ethics approval.

---

## 🧹 Data Pre-processing
Includes:
- Cleaning & missing value handling  
- Label encoding  
- Normalisation  
- Feature engineering  
- Subgroup creation (gender, parental education, address)  
- Exploratory Data Analysis (EDA)  

---

## 🤖 Machine Learning Models

### 1️⃣ Logistic Regression
Simple, interpretable baseline.

### 2️⃣ Random Forest
Non-linear benchmark with strong performance.

### 3️⃣ XGBoost
High-performing gradient boosting model.

---

## ✨ Explainable AI (XAI)

### 🔹 SHAP  
- Global & local feature attributions  
- Clear visualisations of model reasoning  

### 🔹 LIME  
- Local explanations  
- Human-understandable approximations of model decisions  

---

## ⚖️ Fairness Analysis
Assessed across subgroups:
- Gender  
- Parental education  
- Address (Urban vs Rural)  

Metrics include:
- Error rate parity  
- Precision/recall comparison  
- SHAP disparity evaluation  

---

## 📈 Expected Outputs
- Cleaned dataset  
- Baseline ML models  
- Explainability visualisations  
- Fairness evaluation report  
- Complete dissertation  

---

## 🧪 Installation

```bash
git clone https://github.com/your-username/student-performance-xai.git
cd student-performance-xai
pip install -r requirements.txt

