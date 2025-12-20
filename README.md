# Student Academic Performance Prediction — XAI Pipeline

**Author:** Prabin Pokhrel | 2308806  
**Course:** CPS6001 | **Supervisor:** Elisabetta Canetta

---

## What This Project Does

This project builds a supervised machine learning pipeline that predicts whether a secondary school student will pass or fail based on demographic, family background, and behavioural data. Unlike most existing work in this area, the focus is not only on accuracy — the pipeline also explains *why* the model makes each prediction (using SHAP and LIME) and checks whether the model performs fairly across different student groups.

Three classification models are trained and compared:

- **Logistic Regression** — interpretable baseline
- **Random Forest** — handles non-linear patterns
- **XGBoost** — typically highest performing on tabular data

Explainability is provided at two levels:

- **SHAP** (global) — which features drive predictions across the whole dataset
- **LIME** (local) — why the model predicted what it did for a specific student

Fairness is assessed across four demographic attributes: gender, address type, parental education level, and subject.

---

## Dataset

The project uses the **UCI Student Performance Dataset** (Cortez and Silva, 2008):

- Source: https://archive.ics.uci.edu/ml/datasets/student+performance
- Two files: `student-mat.csv` (Mathematics, 395 records) and `student-por.csv` (Portuguese Language, 649 records)
- Combined: 1,044 records, 33 original features + 1 subject identifier

**You must download these files manually** and place them in the `data/` folder before running the notebook.

---

## Requirements

Python 3.9 or later is required. All dependencies are standard and installable via pip.

### Install dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt includes

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
xgboost>=2.0.0
shap>=0.44.0
lime>=0.2.0.1
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

---

## How to Run

### Step 1 — Place dataset files

```
final_project/
  data/
    student-mat.csv    ← download from UCI link above
    student-por.csv    ← download from UCI link above
```

### Step 2 — Run the main notebook

```bash
jupyter notebook XAI_Student_Performance.ipynb
```

Open the notebook in the browser and select **Kernel → Restart & Run All**.

The notebook runs end-to-end without any manual modifications. All output figures are saved automatically to `results/figures/` and all result tables to `results/tables/`.

**Expected runtime:** approximately 5–10 minutes with `N_ITER = 15` (the default). Increase `N_ITER` in Cell 1.1 for more thorough hyperparameter search.

### Step 3 — Launch the dashboard (optional)

```bash
streamlit run dashboard/app.py
```

The dashboard requires the notebook to have been run first so that `results/artefacts.pkl` exists.

---

## Project Structure

```
final_project/
│
├── XAI_Student_Performance.ipynb    ← Main research notebook (run this first)
├── requirements.txt                  ← Python dependencies
├── README.md                         ← This file
│
├── data/
│   ├── student-mat.csv              ← UCI Maths dataset (download separately)
│   └── student-por.csv              ← UCI Portuguese dataset (download separately)
│
├── dashboard/
│   └── app.py                       ← Streamlit interactive dashboard
│
└── results/                         ← Auto-generated when notebook is run
    ├── artefacts.pkl                ← Trained models and preprocessed data
    ├── figures/                     ← 17 output figures (PNG)
    │   ├── fig01_class_distribution.png
    │   ├── fig02_pass_rate_by_subject.png
    │   ├── fig03_correlation_heatmap.png
    │   ├── fig04_feature_distributions.png
    │   ├── fig05_demographic_pass_rates.png
    │   ├── fig06_engineered_features.png
    │   ├── fig07_confusion_matrices.png
    │   ├── fig08_roc_pr_curves.png
    │   ├── fig09_experiment_progression.png
    │   ├── fig10_shap_beeswarm.png
    │   ├── fig11_shap_importance.png
    │   ├── fig12_shap_dependence.png
    │   ├── fig13_lime_explanations.png
    │   ├── fig14_shap_vs_lime.png
    │   ├── fig15_fairness_subgroups.png
    │   ├── fig16_fairness_heatmap.png
    │   └── fig_learning_curves.png
    └── tables/                      ← CSV result tables
        ├── table01_experiment_comparison.csv
        ├── table02_shap_feature_importance.csv
        ├── table03_lime_instance_0.csv
        ├── table04_fairness_sex.csv
        ├── table04_fairness_address.csv
        ├── table04_fairness_Medu.csv
        ├── table04_fairness_subject.csv
        ├── table05_fairness_summary.csv
        └── table06_cross_subject.csv
```

---

## Notebook Sections

| Section | Title | Purpose |
|---------|-------|---------|
| 1 | Environment Setup | Imports, random seed, output directories |
| 2 | Dataset Loading and Integration | Load both UCI files, add subject column, combine |
| 3 | Target Variable and Leakage Removal | Binary pass/fail from G3; remove G1, G2, G3 |
| 4 | Data Quality Validation | Missing values, duplicates, range checks |
| 5 | Exploratory Data Analysis | Class balance, correlations, demographic pass rates |
| 6 | Feature Engineering | Seven derived features with educational justification |
| 7 | Preprocessing Pipeline | ColumnTransformer (scaling + encoding), stratified split |
| 8 | Experimental Design | Three experiments defined with hypotheses |
| 9 | Model Training and Tuning | GridSearchCV (LR), RandomizedSearchCV (RF, XGBoost) |
| 10 | Model Evaluation | F1, ROC-AUC, confusion matrices, McNemar's test, learning curves |
| 11 | SHAP — Global Explainability | Beeswarm, bar chart, dependence plots for XGBoost |
| 12 | LIME — Local Explainability | Per-student explanations for four representative cases |
| 13 | SHAP vs LIME Comparison | Side-by-side comparison of attribution agreement |
| 14 | Fairness Analysis | Subgroup metrics across sex, address, Medu, subject |
| 15 | Cross-Subject Generalisation | Maths-only vs combined dataset model comparison |
| 16 | Results Summary and Conclusions | Consolidated findings and limitations |

---

## Reproducibility

- A fixed `RANDOM_STATE = 42` is set at the top of the notebook and passed to all stochastic operations (train-test split, cross-validation, model initialisation, hyperparameter search).
- All output files are overwritten on each full run — delete `results/artefacts.pkl` if you want to force a clean re-run of the dashboard.
- The notebook is designed to run without any manual edits. The only configuration variable you may want to adjust is `N_ITER` in Cell 1.1, which controls the number of RandomizedSearchCV iterations.

---

## Dashboard

The Streamlit dashboard (`dashboard/app.py`) provides an interactive interface over the trained models and results. It requires `results/artefacts.pkl` to exist (generated by running the notebook).

```bash
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501` and includes:

- Model performance comparison across all three experiments
- SHAP global feature importance visualisation
- LIME individual student explanation viewer
- Fairness subgroup metric summary

---

## Reference

Cortez, P. and Silva, A. (2008) *Using data mining to predict secondary school student performance*. Proceedings of the 5th Annual Future Business Technology Conference, Porto, Portugal, pp. 5–12.
