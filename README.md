<div align="center">

# 🎓 Interpretable & Fair Student Performance Prediction
### *Explainable AI for Educational Decision-Making*

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-00C4B4?style=for-the-badge)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic-8A2BE2?style=for-the-badge)](LICENSE)

<br/>

> *"A model that cannot explain itself cannot be trusted — and in education, trust is everything."*

**Prabin Pokhrel · 2308806 · CPS6001 · Supervisor: Elisabetta Canetta**

</div>

---

## 🌟 Why This Project Matters

Every year, thousands of students fall through the cracks — not because they lack potential, but because educators lack the tools to identify who needs help **early enough** to make a difference.

This project tackles that problem head-on. It builds a machine learning system that predicts whether a secondary school student will **pass or fail** — and more importantly, **explains exactly why** that prediction was made and **proves the system is fair** across demographic groups.

This is not just a prediction engine. It is a **decision-support framework** that puts interpretability and fairness at the centre — the way any AI system touching human lives should.

---

## ✨ What Makes This Different

| Traditional ML Pipeline | This Project |
|---|---|
| Optimises only for accuracy | Balances accuracy, interpretability, and fairness |
| Black-box predictions | Every prediction explained with SHAP + LIME |
| One-size-fits-all model | Audited across gender, address, parental education, subject |
| Results stuck in notebooks | Live interactive dashboard for educators |
| Single experiment | Three progressive experiments with clear hypotheses |

---

## 🔬 Research Question

> **How can Explainable AI techniques — specifically SHAP and LIME — be used to improve the interpretability and fairness of machine learning models that predict student academic performance from demographic, behavioural, and academic data?**

---

## 🏆 Key Results

<div align="center">

| Metric | Best Model (XGBoost) |
|---|---|
| **Accuracy** | 83.3% |
| **F1-Score** | 82.1% |
| **ROC-AUC** | 0.891 |
| **Precision** | 81.6% |
| **Recall** | 82.7% |

</div>

**Top findings from XAI analysis:**
- 📉 **Past failures** is the single strongest negative predictor — two or more failures sharply increases predicted failure risk
- 📉 **Absences** are the second strongest negative factor, compounding with poor health
- 📈 **Study time** and **parental education** are the strongest positive protective factors
- ✅ **Fairness audit**: no demographic attribute exceeds the 0.10 disparity threshold on accuracy or F1

---

## 🛠️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│   UCI Student Performance Dataset (Maths + Portuguese)      │
│   1,044 records · 33 features · Binary pass/fail target     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  ML PIPELINE                                │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │  Logistic   │  │    Random    │  │     XGBoost        │ │
│  │ Regression  │  │    Forest    │  │  ← Best Model      │ │
│  └─────────────┘  └──────────────┘  └────────────────────┘ │
│         3 experiments · GridSearchCV · RandomizedSearchCV   │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
┌─────────▼──────────┐           ┌──────────▼────────────────┐
│  XAI LAYER         │           │  FAIRNESS AUDIT           │
│  SHAP (global)     │           │  sex · address · Medu     │
│  LIME (local)      │           │  subject · 0.10 threshold │
└─────────┬──────────┘           └──────────┬────────────────┘
          │                                 │
          └────────────┬────────────────────┘
                       │
          ┌────────────▼─────────────────┐
          │   STREAMLIT DASHBOARD        │
          │   Live · Interactive · Fair  │
          └──────────────────────────────┘
```

---

## 📁 Project Structure

```
final_project/
│
├── 📓 XAI_Student_Performance.ipynb   ← Main research notebook (start here)
├── 📋 requirements.txt                ← All Python dependencies
├── 📖 README.md                       ← You are here
├── 🧪 conftest.py                     ← Pytest shared fixtures
├── ⚙️  pytest.ini                     ← Test configuration
│
├── 📂 data/
│   ├── student-mat.csv                ← UCI Mathematics dataset (395 records)
│   └── student-por.csv                ← UCI Portuguese dataset (649 records)
│
├── 📂 dashboard/
│   └── app.py                         ← Streamlit interactive dashboard (828 lines)
│
├── 📂 results/                        ← Pre-computed outputs (ready to use)
│   ├── artefacts.pkl                  ← Trained models, preprocessor, feature names
│   ├── figures/                       ← 17 publication-quality figures (PNG)
│   │   ├── fig01_class_distribution.png
│   │   ├── fig02_pass_rate_by_subject.png
│   │   ├── fig03_correlation_heatmap.png
│   │   ├── fig04_feature_distributions.png
│   │   ├── fig05_demographic_pass_rates.png
│   │   ├── fig06_engineered_features.png
│   │   ├── fig07_confusion_matrices.png
│   │   ├── fig08_roc_pr_curves.png
│   │   ├── fig09_experiment_progression.png
│   │   ├── fig10_shap_beeswarm.png
│   │   ├── fig11_shap_importance.png
│   │   ├── fig12_shap_dependence.png
│   │   ├── fig13_lime_explanations.png
│   │   ├── fig14_shap_vs_lime.png
│   │   ├── fig15_fairness_subgroups.png
│   │   ├── fig16_fairness_heatmap.png
│   │   └── fig_learning_curves.png
│   └── tables/                        ← 9 result tables (CSV)
│       ├── table01_experiment_comparison.csv
│       ├── table02_shap_feature_importance.csv
│       ├── table03_lime_instance_0.csv
│       ├── table04_fairness_sex.csv
│       ├── table04_fairness_address.csv
│       ├── table04_fairness_Medu.csv
│       ├── table04_fairness_subject.csv
│       ├── table05_fairness_summary.csv
│       └── table06_cross_subject.csv
│
└── 📂 tests/                          ← Full pytest test suite (99 tests)
    ├── __init__.py
    ├── test_data_loading.py           ← 15 tests — UCI schema & integrity
    ├── test_feature_engineering.py    ← 20 tests — 7 engineered feature formulas
    ├── test_model_pipeline.py         ← 8 tests  — ML pipeline contracts
    ├── test_fairness.py               ← 10 tests — disparity & threshold logic
    └── test_dashboard_utils.py        ← 46 tests — dashboard pure logic
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or later
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Praaabin/Student-Performance-Prediction-Using-Explainable-and-Fair-Machine-Learning.git
cd Student-Performance-Prediction-Using-Explainable-and-Fair-Machine-Learning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard immediately

The `results/` folder is pre-computed and included in the repository. You can launch the interactive dashboard right away — **no need to run the notebook first**:

```bash
streamlit run dashboard/app.py
```

Open your browser at **`http://localhost:8501`** ✨

---

## 🔁 Reproducing Results from Scratch

If you want to re-run the full ML pipeline and regenerate all outputs:

```bash
jupyter notebook XAI_Student_Performance.ipynb
```

Open the notebook and select **Kernel → Restart & Run All**.

> ⏱️ **Expected runtime:** 5–10 minutes (default `N_ITER = 15`). Increase `N_ITER` in Cell 1.1 for a more thorough hyperparameter search.

All figures are saved to `results/figures/` and all tables to `results/tables/` automatically.

---

## 📊 Notebook Walkthrough

| # | Section | What Happens |
|---|---------|-------------|
| 1 | **Environment Setup** | Imports, random seed (`RANDOM_STATE = 42`), output directories |
| 2 | **Dataset Loading** | Load both UCI files, add `subject` column, combine to n=1,044 |
| 3 | **Target Variable** | Binary `pass/fail` from G3 ≥ 10; remove G1, G2, G3 (leakage prevention) |
| 4 | **Data Quality** | Missing values, duplicates, range validation |
| 5 | **EDA** | Class balance, correlations, demographic pass-rate breakdown |
| 6 | **Feature Engineering** | 7 domain-informed derived features |
| 7 | **Preprocessing** | ColumnTransformer (StandardScaler + OneHotEncoder), 80/20 stratified split |
| 8 | **Experimental Design** | 3 progressive phases with testable hypotheses |
| 9 | **Model Training** | GridSearchCV (LR), RandomizedSearchCV (RF, XGBoost) |
| 10 | **Evaluation** | F1, ROC-AUC, confusion matrices, McNemar's test, learning curves |
| 11 | **SHAP** | Beeswarm, bar chart, dependence plots — global model behaviour |
| 12 | **LIME** | Per-student local explanations for four representative cases |
| 13 | **SHAP vs LIME** | Side-by-side attribution agreement comparison |
| 14 | **Fairness Audit** | Subgroup metrics across sex, address, Medu, subject |
| 15 | **Generalisation** | Maths-only vs combined dataset model comparison |
| 16 | **Conclusions** | Consolidated findings, limitations, future directions |

---

## 🧪 Running Tests

The test suite validates every critical component independently — no Streamlit launch required.

```bash
# Run all 99 tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=term-missing

# Run a specific module
pytest tests/test_feature_engineering.py -v

# Run only fast unit tests (no real data needed)
pytest -m "not data" -v
```

**Test breakdown:**

| Module | Tests | Covers |
|--------|-------|--------|
| `test_data_loading.py` | 15 | UCI schema, row counts, null checks, grade/age bounds |
| `test_feature_engineering.py` | 20 | All 7 feature formulas, edge cases, NaN safety |
| `test_model_pipeline.py` | 8 | Preprocessor, predict contracts, probabilities, model selection |
| `test_fairness.py` | 10 | Subgroup metrics, disparity calculation, 0.10 threshold |
| `test_dashboard_utils.py` | 46 | Risk labels, bullet logic, student profile, helper formulas |
| **Total** | **99** | **All passing ✅** |

> Data-dependent tests (`-m data`) skip automatically if the CSV files are not present.

---

## 🎛️ Dashboard Pages

The Streamlit dashboard (`dashboard/app.py`) provides five interactive views:

| Page | Description |
|------|-------------|
| **Overview** | Research context, objectives, and key results at a glance |
| **Predict Student Outcome** | Enter a student profile → get a PASS/FAIL prediction, probability, risk level, LIME explanation, and a plain-English intervention summary |
| **Model Performance** | Side-by-side metric table, ROC/PR curves, confusion matrices, and experiment progression chart |
| **Feature Importance** | SHAP beeswarm, global importance bar, dependence plots, and SHAP vs LIME comparison |
| **Fairness Analysis** | Demographic fairness heatmap, subgroup performance charts, and per-attribute interpretations |

---

## ⚙️ Engineered Features

Seven domain-informed features were derived to capture non-linear academic signals invisible to raw data:

| Feature | Formula | Educational Rationale |
|---------|---------|----------------------|
| `study_efficiency` | `studytime / (failures + 1)` | How productively the student uses study time relative to past setbacks |
| `academic_risk_score` | `failures×2 + absences/10 + (5−studytime)` | Composite early-warning indicator |
| `studytime_x_failures` | `studytime × failures` | Interaction: high study time after many failures signals struggling students |
| `social_engagement` | `goout + freetime − Dalc − Walc` | Net social capital vs. risky behaviour |
| `parent_edu_avg` | `(Medu + Fedu) / 2` | Home academic environment proxy |
| `support_index` | `schoolsup + famsup + paid` (count) | Total active support channels (0–3) |
| `health_x_absences` | `health × absences` | Captures health-driven absenteeism |

---

## 🔒 Reproducibility Guarantee

- **Fixed seed:** `RANDOM_STATE = 42` applied to all stochastic operations
- **Deterministic splits:** Stratified 80/20 train-test split preserves class balance
- **No leakage:** G1 and G2 (intermediate grades) removed before any model sees data
- **Self-contained:** All results pre-computed and committed — no external downloads needed
- **Full re-run:** Delete `results/artefacts.pkl` and re-run the notebook to regenerate everything from scratch

---

## 🌿 Git Branch Strategy

| Branch | Purpose | Period |
|--------|---------|--------|
| `main` | Stable, always-deployable state | Continuous |
| `feature/eda-and-preprocessing` | EDA, data loading, feature engineering | Jan 2026 |
| `feature/model-training` | 3-phase training and evaluation | Jan–Feb 2026 |
| `feature/xai-fairness` | SHAP, LIME, and fairness audit | Feb–Mar 2026 |
| `feature/streamlit-dashboard` | Interactive Streamlit application | Mar 2026 |
| `feature/testing` | Full pytest suite (99 tests) | Apr 2026 |

All feature branches are merged into `main` via `--no-ff` merge commits to preserve a clean, readable development history.

---

## 📚 References

- Cortez, P. and Silva, A. (2008) *Using data mining to predict secondary school student performance.* Proceedings of the 5th Annual Future Business Technology Conference, Porto, pp. 5–12.
- Lundberg, S.M. and Lee, S.I. (2017) *A unified approach to interpreting model predictions.* NeurIPS 2017.
- Ribeiro, M.T., Singh, S. and Guestrin, C. (2016) *"Why should I trust you?": Explaining the predictions of any classifier.* KDD 2016.
- Chouldechova, A. (2017) *Fair prediction with disparate impact: A study of bias in recidivism prediction instruments.* Big Data, 5(2), pp. 153–163.
- UCI Machine Learning Repository: [Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)

---

<div align="center">

**Built with 🧠 by Prabin Pokhrel · 2308806**

*Interpretable AI · Fairness-Aware ML · Educational Analytics*

</div>
