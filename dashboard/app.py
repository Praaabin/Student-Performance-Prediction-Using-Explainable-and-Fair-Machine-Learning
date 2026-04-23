"""
dashboard/app.py
================
Interpretable AI System for Student Performance Prediction
Streamlit Dashboard — matches the design mockup

Author  : Prabin Pokhrel | 2308806
Project : XAI for Student Performance Prediction
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                              accuracy_score, recall_score, f1_score)

warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = 'XAI Student Performance',
    page_icon   = '',
    layout      = 'wide',
    initial_sidebar_state = 'expanded',
)

# ── Design tokens ──────────────────────────────────────────────────────────
C_DARK  = '#1a3a5c'
C_BLUE  = '#2E75B6'
C_RED   = '#C0504D'
C_GREEN = '#70AD47'
C_GOLD  = '#F79646'
C_LIGHT = '#EEF3F8'

# ── Inject global CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&family=Source+Serif+Pro:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Source Sans Pro', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2E75B6 100%);
    padding: 28px 32px 22px;
    border-radius: 8px;
    color: white;
    margin-bottom: 24px;
}
.main-header h1 {
    font-family: 'Source Serif Pro', serif;
    font-size: 1.7em;
    font-weight: 600;
    margin: 0 0 4px 0;
    line-height: 1.25;
}
.main-header p {
    font-size: 0.9em;
    opacity: 0.82;
    margin: 0;
    font-style: italic;
}

.section-card {
    background: white;
    border: 1px solid #dce4ed;
    border-radius: 6px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.section-card h3 {
    color: #1a3a5c;
    font-size: 0.95em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin: 0 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #2E75B6;
}

.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 12px;
}
.metric-box {
    flex: 1;
    background: #EEF3F8;
    border-radius: 5px;
    padding: 12px 14px;
    text-align: center;
}
.metric-box .label {
    font-size: 0.72em;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 4px;
}
.metric-box .value {
    font-size: 1.5em;
    font-weight: 700;
    color: #1a3a5c;
}

.result-pass {
    background: #d4edda;
    border: 2px solid #28a745;
    border-radius: 6px;
    padding: 14px 18px;
    text-align: center;
}
.result-fail {
    background: #f8d7da;
    border: 2px solid #C0504D;
    border-radius: 6px;
    padding: 14px 18px;
    text-align: center;
}
.result-label {
    font-size: 1.1em;
    font-weight: 700;
    margin-bottom: 6px;
}
.result-prob { font-size: 1.4em; font-weight: 700; }
.result-risk { font-size: 0.85em; font-weight: 600; margin-top: 4px; }

.insight-box {
    background: #fff8e1;
    border-left: 4px solid #F79646;
    padding: 10px 14px;
    border-radius: 0 4px 4px 0;
    font-size: 0.87em;
    color: #444;
    margin-top: 10px;
}
.insight-box strong { color: #1a3a5c; }

.fairness-flag {
    background: #fdecea;
    border-left: 4px solid #C0504D;
    padding: 10px 14px;
    border-radius: 0 4px 4px 0;
    font-size: 0.87em;
    color: #444;
    margin-top: 8px;
}
.fairness-ok {
    background: #e8f5e9;
    border-left: 4px solid #70AD47;
    padding: 10px 14px;
    border-radius: 0 4px 4px 0;
    font-size: 0.87em;
    color: #444;
    margin-top: 8px;
}

.rec-item {
    background: #f0f7ff;
    border-radius: 4px;
    padding: 9px 14px;
    margin-bottom: 7px;
    font-size: 0.87em;
    color: #1a3a5c;
    border-left: 3px solid #2E75B6;
}

.stSelectbox label, .stSlider label, .stNumberInput label {
    font-size: 0.88em !important;
    color: #333 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load artefacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    path = os.path.join(os.path.dirname(__file__), '..', 'results', 'artefacts.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

art = load_artefacts()


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Interpretable AI System for Student Performance Prediction</h1>
    <p>A Fairness-Aware Machine Learning Framework for Educational Decision-Making
       &nbsp;|&nbsp; Prabin Pokhrel &nbsp;|&nbsp; 2308806</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="background:{C_DARK};padding:18px;border-radius:6px;color:white;margin-bottom:18px;">
        <div style="font-size:1.1em;font-weight:700;font-family:'Source Serif Pro',serif;">XAI Dashboard</div>
        <div style="font-size:0.8em;opacity:0.75;margin-top:4px;">Student Performance Prediction</div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio('Navigation', [
        'Overview',
        'Predict Student Outcome',
        'Model Performance',
        'Feature Importance',
        'Fairness Analysis',
    ])

    if art:
        st.markdown('---')
        st.markdown(f"""
        <div style="font-size:0.8em;color:#666;">
        <b>Best Model:</b> {art.get('best_model_name','—')}<br/>
        <b>Dataset:</b> Mathematics + Portuguese<br/>
        <b>Total records:</b> 1,044
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════
if page == 'Overview':
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-card"><h3>Research Context</h3>', unsafe_allow_html=True)
        st.markdown("""
        **A Fairness-Aware Machine Learning Framework for Educational Decision-Making**
        *Prabin Pokhrel | 2308806*

        This dashboard presents the outputs of a project investigating whether
        machine learning models can predict student academic pass/fail outcomes while remaining
        interpretable and demographically fair.

        **Dataset:** UCI Student Performance Dataset (Cortez and Silva, 2008) — publicly
        available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/student+performance)
        and [Kaggle](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set).
        Mathematics and Portuguese subjects combined (n = 1,044 records, 33 features).

        **Approach:** Three classifiers (Logistic Regression, Random Forest, XGBoost) were
        trained across a three-phase experimental framework. Predictions are explained globally
        using SHAP and locally using LIME. A dedicated fairness audit evaluates demographic parity
        across gender, address type, parental education, and academic subject.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card"><h3>Research Question & Objectives</h3>', unsafe_allow_html=True)
        st.markdown("""
        **How can Explainable AI techniques — specifically SHAP and LIME — be used to improve
        the interpretability and fairness of machine learning models that predict student
        academic performance from demographic, behavioural, and academic data?**
        """)
        st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
        for q in [
            '① Train and compare Logistic Regression, Random Forest, and XGBoost classifiers',
            '② Apply SHAP (global) and LIME (local) for model interpretability',
            '③ Evaluate performance using F1, ROC-AUC, precision, recall, and accuracy',
            '④ Assess fairness across demographic subgroups (gender, address, parental education, subject)',
            '⑤ Summarise findings and produce output visualisations for educational stakeholders',
        ]:
            st.markdown(f'<div class="rec-item">{q}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if art:
        st.markdown('<div class="section-card"><h3>Key Results at a Glance</h3>', unsafe_allow_html=True)
        m = art['metrics_df']
        bn = art['best_model_name']
        best = m.loc[bn] if bn in m.index else m.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, val in [
            (c1, 'Accuracy',  f"{best['Accuracy']:.1%}"),
            (c2, 'F1-Score',  f"{best['F1']:.1%}"),
            (c3, 'ROC-AUC',   f"{best['ROC-AUC']:.3f}"),
            (c4, 'Precision', f"{best['Precision']:.1%}"),
            (c5, 'Recall',    f"{best['Recall']:.1%}"),
        ]:
            col.markdown(f"""
            <div class="metric-box">
              <div class="label">{label}</div>
              <div class="value">{val}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box"><strong>Best model:</strong> {bn} — selected by ROC-AUC on the held-out test set.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════
elif page == 'Model Performance':
    st.markdown('<div class="section-card"><h3>Model Performance Comparison</h3>', unsafe_allow_html=True)

    if art:
        m = art['metrics_df'].copy()
        bn = art['best_model_name']
        m.index = [n.replace('_',' ') for n in m.index]
        bn_disp = bn.replace('_',' ')

        # Metrics table
        def style_table(df, best):
            styled = df.style\
                .format('{:.4f}')\
                .apply(lambda row: [
                    f'background-color:#d4edda;font-weight:700;' if row.name == best else ''
                    for _ in row], axis=1)\
                .set_table_styles([
                    {'selector':'th','props':[
                        ('background-color',C_DARK),('color','white'),
                        ('font-size','0.88em'),('text-align','center'),('padding','8px 12px')]},
                    {'selector':'td','props':[
                        ('text-align','center'),('padding','7px 12px'),('font-size','0.9em')]},
                ])
            return styled
        st.dataframe(style_table(m, bn_disp), use_container_width=True)
        st.markdown(f'<div class="insight-box"><strong>Best model:</strong> {bn_disp} achieves the highest ROC-AUC ({m.loc[bn_disp,"ROC-AUC"]:.3f}), indicating superior discriminative ability across all classification thresholds.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Figures
    col1, col2 = st.columns(2)
    fdir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')

    for col, fname, caption in [
        (col1, 'fig08_roc_pr_curves.png', 'Figure 8. ROC and Precision-Recall Curves'),
        (col2, 'fig07_confusion_matrices.png', 'Figure 7. Confusion Matrices'),
    ]:
        path = os.path.join(fdir, fname)
        with col:
            st.markdown(f'<div class="section-card"><h3>{caption}</h3>', unsafe_allow_html=True)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.info('Run the notebook first to generate this figure.')
            st.markdown('</div>', unsafe_allow_html=True)

    path = os.path.join(fdir, 'fig09_experiment_progression.png')
    st.markdown('<div class="section-card"><h3>Figure 6. Experimental Comparison: Baseline vs Feature Engineering vs Tuned Models</h3>', unsafe_allow_html=True)
    if os.path.exists(path):
        st.image(path, use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Interpretation:</strong> Feature engineering produces a consistent F1 improvement across all three model types. Tuning provides a further increment, confirming that both pipeline components contribute independently to performance.</div>', unsafe_allow_html=True)
    else:
        st.info('Run the notebook to generate this figure.')
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════
elif page == 'Feature Importance':
    fdir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    bn   = art['best_model_name'] if art else '—'

    st.markdown(f'<div class="section-card"><h3>SHAP Global Explanations — {bn}</h3>', unsafe_allow_html=True)
    st.markdown("""
    SHAP (SHapley Additive exPlanations) assigns each feature a Shapley value — its average
    marginal contribution to the prediction across all possible feature coalitions. These values
    are theoretically grounded, uniquely determined, and satisfy four desirable properties:
    efficiency, symmetry, dummy, and linearity.
    """)

    col1, col2 = st.columns(2)
    for col, fname, cap in [
        (col1, 'fig10_shap_beeswarm.png',  'Figure 10. SHAP Summary (Beeswarm)'),
        (col2, 'fig11_shap_importance.png', 'Figure 11. Global Feature Importance'),
    ]:
        path = os.path.join(fdir, fname)
        with col:
            if os.path.exists(path):
                st.image(path, use_container_width=True, caption=cap)
            else:
                st.info('Run the notebook to generate this figure.')

    st.markdown('<div class="insight-box"><strong>Key finding:</strong> Past failures and absences are the strongest negative predictors. Students with more than two prior failures receive consistently large negative SHAP values. Study time and parental education exert positive influence, reflecting their established roles as protective factors in educational research.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    path_dep = os.path.join(fdir, 'fig12_shap_dependence.png')
    st.markdown('<div class="section-card"><h3>Figure 12. SHAP Dependence Plots — Top 3 Features</h3>', unsafe_allow_html=True)
    if os.path.exists(path_dep):
        st.image(path_dep, use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Interpretation:</strong> The dependence plots reveal non-linear relationships not visible from correlation analysis alone. The negative impact of failures accelerates beyond two prior failures; study time shows diminishing protective returns at the highest level. These patterns have direct implications for the timing of educational interventions.</div>', unsafe_allow_html=True)
    else:
        st.info('Run the notebook to generate this figure.')
    st.markdown('</div>', unsafe_allow_html=True)

    # SHAP vs LIME comparison
    col1, col2 = st.columns(2)
    path_sv = os.path.join(fdir, 'fig14_shap_vs_lime.png')
    path_lm = os.path.join(fdir, 'fig13_lime_explanations.png')

    with col1:
        st.markdown('<div class="section-card"><h3>Figure 14. SHAP vs LIME Comparison</h3>', unsafe_allow_html=True)
        if os.path.exists(path_sv):
            st.image(path_sv, use_container_width=True)
        else:
            st.info('Run the notebook to generate this figure.')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card"><h3>Figure 13. LIME Local Explanations</h3>', unsafe_allow_html=True)
        if os.path.exists(path_lm):
            st.image(path_lm, use_container_width=True)
        else:
            st.info('Run the notebook to generate this figure.')
        st.markdown('</div>', unsafe_allow_html=True)

    # Method comparison table
    st.markdown('<div class="section-card"><h3>SHAP vs LIME: Method Comparison</h3>', unsafe_allow_html=True)
    cmp_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables', 'shap_lime_comparison.csv')
    if os.path.exists(cmp_path):
        cmp = pd.read_csv(cmp_path, index_col=0)
        st.dataframe(cmp, use_container_width=True)
    else:
        cmp_data = {
            'Attribute'   :['Theoretical basis','Scope','Consistency','Primary use case'],
            'SHAP'        :['Shapley values (game theory)','Global and local','Unique and consistent','Global audit; project'],
            'LIME'        :['Local linear approximation','Local only','Stochastic','Individual student explanation'],
        }
        st.dataframe(pd.DataFrame(cmp_data).set_index('Attribute'), use_container_width=True)
    st.markdown('<div class="insight-box"><strong>Recommendation:</strong> SHAP is preferred for academic reporting and global model auditing due to its theoretical foundations and uniqueness. LIME is more appropriate for communicating individual predictions to educators in plain language.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: PREDICT STUDENT OUTCOME
# ════════════════════════════════════════════════════════════════════
elif page == 'Predict Student Outcome':
    # ── How-to guide ────────────────────────────────────────────────
    st.markdown('<div class="section-card"><h3>How to Use This Page</h3>', unsafe_allow_html=True)
    st.markdown("""
    This tool uses the best-performing trained model to predict whether a student is likely to
    **pass or fail** based on their personal, academic, and behavioural profile.
    It is designed to support — not replace — educator judgement.
    """)
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        st.markdown("""
**① Fill in the student profile**

Use the three columns below to enter the student's details:
- **Academic** — study time, past failures, absences, extra support
- **Demographic** — age, gender, address, parental education, subject
- **Behavioural** — going out, alcohol use, health, free time

All fields have sensible defaults. Only change what is relevant to the student you are assessing.
        """)
    with c_b:
        st.markdown("""
**② Click "Generate Prediction"**

The model returns:
- A **PASS** or **FAIL** prediction
- The **probability of passing** (e.g. 72% = likely to pass)
- A **risk level**: LOW · MODERATE · HIGH · VERY HIGH

A higher probability means the model is more confident the student will pass. A probability below 50% means the model predicts failure.
        """)
    with c_c:
        st.markdown("""
**③ Read the LIME explanation**

The bar chart shows *why* the model made this prediction:
- 🔵 **Blue bars** push the prediction towards **PASS**
- 🔴 **Red bars** push the prediction towards **FAIL**
- Longer bars = stronger influence on the outcome

Use this to identify which factors are most critical for this student and where targeted support could make the most difference.
        """)
    st.markdown('<div class="insight-box"><strong>Important:</strong> This model was trained on historical student data and should be used as a decision-support aid only. Predictions carry inherent uncertainty and must always be reviewed alongside an educator\'s direct knowledge of the student. The model does not replace professional judgement.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><h3>Student Outcome Prediction</h3>', unsafe_allow_html=True)
    st.markdown('Complete the student profile below, then click **Generate Prediction** to see the outcome, risk level, and a personalised explanation of the key contributing factors.')

    if not art:
        st.error('Artefacts not found. Run the notebook to train models first.')
        st.stop()

    model     = art['models'][art['best_model_name']]
    pre       = art['preprocessor']
    eng_sc    = art['eng_scaler']
    num_feats = art['numeric_features']
    cat_feats = art['categorical_features']
    all_feats = art['feature_names']
    eng_feats = art['eng_features']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('**Academic Factors**')
        study_time  = st.selectbox('Study Time (weekly)', [1,2,3,4],
                                   format_func=lambda x: {1:'<2 hours',2:'2-5 hours',3:'5-10 hours',4:'>10 hours'}[x])
        failures    = st.selectbox('Past Failures', [0,1,2,3])
        absences    = st.number_input('Absences', 0, 93, 4)
        higher      = st.selectbox('Wants Higher Education', ['yes','no'])
        paid        = st.selectbox('Extra Paid Classes', ['yes','no'])
        schoolsup   = st.selectbox('School Support', ['yes','no'])
        famsup      = st.selectbox('Family Support', ['yes','no'])

    with col2:
        st.markdown('**Demographic Factors**')
        age      = st.slider('Age', 15, 22, 17)
        sex      = st.selectbox('Gender', ['F','M'])
        address  = st.selectbox('Address Type', ['U','R'],
                                format_func=lambda x: {'U':'Urban','R':'Rural'}[x])
        Medu     = st.slider("Mother's Education (0–4)", 0, 4, 2)
        Fedu     = st.slider("Father's Education (0–4)", 0, 4, 2)
        subject  = st.selectbox('Subject', ['mathematics','portuguese'])
        famsize  = st.selectbox('Family Size', ['GT3','LE3'])

    with col3:
        st.markdown('**Behavioural Factors**')
        health   = st.slider('Health (1=poor, 5=excellent)', 1, 5, 3)
        goout    = st.slider('Going Out (1–5)', 1, 5, 3)
        Dalc     = st.slider('Weekday Alcohol (1–5)', 1, 5, 1)
        Walc     = st.slider('Weekend Alcohol (1–5)', 1, 5, 2)
        freetime = st.slider('Free Time (1–5)', 1, 5, 3)
        famrel   = st.slider('Family Relations (1–5)', 1, 5, 4)
        traveltime = st.selectbox('Travel Time', [1,2,3,4],
                                  format_func=lambda x:{1:'<15 min',2:'15-30 min',3:'30-60 min',4:'>60 min'}[x])

    predict_btn = st.button('Generate Prediction', type='primary')

    if predict_btn:
        student_raw = {
            'age':age,'Medu':Medu,'Fedu':Fedu,'traveltime':traveltime,
            'studytime':study_time,'failures':failures,'famrel':famrel,
            'freetime':freetime,'goout':goout,'Dalc':Dalc,'Walc':Walc,
            'health':health,'absences':absences,
            'school':'GP','sex':sex,'address':address,'famsize':famsize,
            'Pstatus':'T','Mjob':'other','Fjob':'other','reason':'course',
            'guardian':'mother','schoolsup':schoolsup,'famsup':famsup,
            'paid':paid,'activities':'no','nursery':'yes','higher':higher,
            'internet':'yes','romantic':'no','subject':subject,
        }

        try:
            df_s    = pd.DataFrame([student_raw])
            df_s_num = df_s[[c for c in num_feats if c in df_s.columns]]
            df_s_cat = df_s[[c for c in cat_feats if c in df_s.columns]]
            df_s_all = df_s[[c for c in num_feats+cat_feats if c in df_s.columns]]
            X_raw   = pre.transform(df_s_all)

            # Engineered features
            def eng_student(d):
                r = {}
                r['study_efficiency']     = d['studytime'] / (d['failures'] + 1)
                r['academic_risk_score']  = d['failures']*2 + d['absences']/10 + (5-d['studytime'])
                r['studytime_x_failures'] = d['studytime'] * d['failures']
                r['social_engagement']    = d['goout'] + d['freetime'] - d['Dalc'] - d['Walc']
                r['parent_edu_avg']       = (d['Medu'] + d['Fedu']) / 2
                r['support_index']        = sum(1 for c in ['schoolsup','famsup','paid'] if d.get(c)=='yes')
                r['health_x_absences']    = d['health'] * d['absences']
                return r

            eng_vals = eng_student(student_raw)
            eng_arr  = eng_sc.transform([[eng_vals[f] for f in eng_feats]])
            X_full   = np.hstack([X_raw, eng_arr])

            y_pred   = int(model.predict(X_full)[0])
            y_proba  = float(model.predict_proba(X_full)[0, 1])
            risk_pct = (1 - y_proba) * 100

            st.markdown('---')
            r1, r2, r3 = st.columns([2, 1, 2])

            with r2:
                if y_pred == 1:
                    risk_label = 'LOW' if risk_pct < 30 else 'MODERATE'
                    st.markdown(f"""
                    <div class="result-pass">
                        <div class="result-label" style="color:#155724;">Prediction: PASS</div>
                        <div class="result-prob" style="color:#155724;">{y_proba:.1%}</div>
                        <div class="result-risk">P(PASS) | Risk Level: {risk_label}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    risk_label = 'VERY HIGH' if y_proba < 0.2 else 'HIGH'
                    st.markdown(f"""
                    <div class="result-fail">
                        <div class="result-label" style="color:#721c24;">Prediction: FAIL</div>
                        <div class="result-prob" style="color:#721c24;">{y_proba:.1%}</div>
                        <div class="result-risk">P(PASS) | Risk Level: {risk_label}</div>
                    </div>""", unsafe_allow_html=True)

            # LIME explanation
            with r1:
                st.markdown('<div class="section-card"><h3>Key Contributing Factors (LIME)</h3>', unsafe_allow_html=True)
                try:
                    from lime import lime_tabular
                    X_train_full = art.get('X_train_full', art.get('X_test_full'))  # training data as LIME background
                    lime_exp_obj = lime_tabular.LimeTabularExplainer(
                        training_data=X_train_full, feature_names=all_feats,
                        class_names=['FAIL','PASS'], mode='classification',
                        random_state=42, discretize_continuous=True)
                    exp = lime_exp_obj.explain_instance(
                        X_full[0], model.predict_proba, num_features=8, labels=[1])
                    items = exp.as_list(label=1)

                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    features = [x[0][:25] for x in items]
                    weights  = [x[1] for x in items]
                    colors   = ['#2E75B6' if w>0 else '#C0504D' for w in weights]
                    ax.barh(features[::-1], weights[::-1], color=colors[::-1], alpha=0.85)
                    ax.axvline(0, color='black', lw=0.8)
                    ax.set_xlabel('LIME Weight')
                    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.markdown('<div style="font-size:0.82em;color:#666;">Blue = pushes towards PASS | Red = pushes towards FAIL</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f'LIME explanation unavailable: {e}')
                st.markdown('</div>', unsafe_allow_html=True)

            with r3:
                st.markdown('<div class="section-card"><h3>Student Profile Summary</h3>', unsafe_allow_html=True)
                risk_score = failures*2 + absences/10 + (5-study_time)
                study_eff  = study_time / (failures + 1)
                metrics_disp = [
                    ('Academic Risk Score', f'{risk_score:.2f}', 'Higher = greater risk'),
                    ('Study Efficiency', f'{study_eff:.2f}', 'Study time / (failures+1)'),
                    ('Absences', str(absences), 'Class sessions missed'),
                    ('Past Failures', str(failures), 'Prior failed courses'),
                    ('Support Channels', str(sum(1 for v in [schoolsup,famsup,paid] if v=='yes')), 'Of 3 available'),
                ]
                for label, val, note in metrics_disp:
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:7px 0;border-bottom:1px solid #eee;font-size:0.88em;">
                      <span><b>{label}</b><br/><span style="color:#888;font-size:0.85em;">{note}</span></span>
                      <span style="font-weight:700;color:{C_DARK};font-size:1.1em;">{val}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Dynamic plain-English summary ────────────────────────
            st.markdown('<div class="section-card"><h3>What This Result Means</h3>', unsafe_allow_html=True)

            outcome_word = "pass" if y_pred == 1 else "fail"
            outcome_color = "#155724" if y_pred == 1 else "#721c24"
            risk_score_val = failures*2 + absences/10 + (5-study_time)
            study_eff_val  = study_time / (failures + 1)
            support_count  = sum(1 for v in [schoolsup, famsup, paid] if v == 'yes')
            parent_edu_avg = (Medu + Fedu) / 2

            # Headline sentence
            st.markdown(f"""
<p style="font-size:1.05em;color:{outcome_color};font-weight:600;margin-bottom:8px;">
The model predicts this student will <strong>{outcome_word.upper()}</strong> with a
{y_proba:.0%} probability of passing — classified as <strong>{risk_label} risk</strong>.
</p>""", unsafe_allow_html=True)

            # Build context-aware bullet points
            points = []

            # Failures
            if failures == 0:
                points.append(("✅", "No past failures", "This student has no record of previously failed courses, which is the single strongest positive indicator in this model."))
            elif failures == 1:
                points.append(("⚠️", "1 past failure", "One prior failed course increases the predicted risk moderately. Early academic support is advisable."))
            else:
                points.append(("🔴", f"{failures} past failures", f"Past failures are the strongest negative predictor in this model. {failures} failures significantly raises the risk of not passing."))

            # Absences
            if absences <= 3:
                points.append(("✅", f"Low absences ({absences})", "Good attendance is a protective factor — the student is unlikely to have missed critical instruction time."))
            elif absences <= 10:
                points.append(("⚠️", f"Moderate absences ({absences})", "Some missed sessions may have affected learning continuity. Monitoring attendance going forward is recommended."))
            else:
                points.append(("🔴", f"High absences ({absences})", f"{absences} missed sessions is well above average and is a strong contributor to predicted failure risk."))

            # Study time
            study_labels = {1: "less than 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "more than 10 hours"}
            if study_time >= 3:
                points.append(("✅", f"Strong study time ({study_labels[study_time]} per week)", "Above-average study time is a positive signal and offsets some negative risk factors."))
            elif study_time == 2:
                points.append(("⚠️", f"Moderate study time ({study_labels[study_time]} per week)", "Study time is at the lower end. Increasing structured study could improve the outcome."))
            else:
                points.append(("🔴", f"Very low study time ({study_labels[study_time]} per week)", "Less than 2 hours per week is the lowest study band and significantly raises predicted risk."))

            # Study efficiency
            if study_eff_val >= 2.0:
                points.append(("✅", f"High study efficiency ({study_eff_val:.2f})", "The student's study time is well-used relative to their failure history."))
            elif study_eff_val >= 1.0:
                points.append(("⚠️", f"Moderate study efficiency ({study_eff_val:.2f})", "There is room to improve how study time translates into academic outcomes."))
            else:
                points.append(("🔴", f"Low study efficiency ({study_eff_val:.2f})", "Study time is heavily diluted by past failures, indicating the student may need targeted academic support rather than just more hours."))

            # Support
            if support_count == 3:
                points.append(("✅", "Maximum support (school + family + paid classes)", "Having all three support channels active is a strong protective factor."))
            elif support_count == 2:
                points.append(("✅", f"{support_count} of 3 support channels active", "Good level of academic support in place."))
            elif support_count == 1:
                points.append(("⚠️", "Only 1 support channel active", "Consider activating additional school or family support to reduce risk."))
            else:
                points.append(("🔴", "No support channels active", "No school support, family support, or paid tutoring is active. This is a key area for intervention."))

            # Parental education
            if parent_edu_avg >= 3:
                points.append(("✅", f"High parental education (avg {parent_edu_avg:.1f}/4)", "Higher parental education is associated with better home learning environments."))
            elif parent_edu_avg >= 2:
                points.append(("⚠️", f"Moderate parental education (avg {parent_edu_avg:.1f}/4)", "Parental education is at a mid level — home academic support may be limited."))
            else:
                points.append(("🔴", f"Low parental education (avg {parent_edu_avg:.1f}/4)", "Lower parental education is associated with reduced home academic support. School-based interventions are especially important for this student."))

            # Render bullets
            for icon, title, detail in points:
                bg  = "#e8f5e9" if icon == "✅" else ("#fff3cd" if icon == "⚠️" else "#fdecea")
                bdr = "#70AD47"  if icon == "✅" else ("#F79646"  if icon == "⚠️" else "#C0504D")
                st.markdown(f"""
<div style="background:{bg};border-left:4px solid {bdr};padding:9px 14px;
            border-radius:0 4px 4px 0;margin-bottom:7px;font-size:0.88em;color:#333;">
  <strong>{icon} {title}</strong><br/>{detail}
</div>""", unsafe_allow_html=True)

            # Closing recommendation
            if y_pred == 1:
                st.markdown('<div class="fairness-ok"><strong>Recommendation:</strong> This student is on track. Continue monitoring attendance and study habits. Maintaining current support levels should sustain positive outcomes.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="fairness-flag"><strong>Recommendation:</strong> This student is at risk. Priority actions should focus on the red-flagged factors above — particularly past failures and absences. Early intervention (tutoring, attendance monitoring, counselling) is recommended before the situation becomes harder to reverse.</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f'Prediction error: {e}')

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: FAIRNESS ANALYSIS
# ════════════════════════════════════════════════════════════════════
elif page == 'Fairness Analysis':
    st.markdown('<div class="section-card"><h3>Demographic Fairness Audit</h3>', unsafe_allow_html=True)
    st.markdown("""
    This section evaluates whether the best-performing model produces equitable outcomes across
    four sensitive attributes: gender, address type, mother's education level, and academic subject.
    The 0.10 threshold is adopted from the fairness literature (Chouldechova, 2017) to identify
    potentially meaningful performance disparities.
    """)

    fdir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    tdir = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables')

    path_hm = os.path.join(fdir, 'fig16_fairness_heatmap.png')
    if os.path.exists(path_hm):
        st.image(path_hm, use_container_width=True, caption='Figure 16. Fairness Metrics Heatmap')
        st.markdown('<div class="insight-box"><strong>How to read:</strong> Green cells indicate the difference between best and worst subgroup is within the 0.10 threshold. Red cells indicate a potentially meaningful disparity requiring further investigation.</div>', unsafe_allow_html=True)
    else:
        st.info('Run the notebook to generate the fairness heatmap.')
    st.markdown('</div>', unsafe_allow_html=True)

    path_sub = os.path.join(fdir, 'fig15_fairness_subgroups.png')
    st.markdown('<div class="section-card"><h3>Figure 15. Subgroup Performance Analysis</h3>', unsafe_allow_html=True)
    if os.path.exists(path_sub):
        st.image(path_sub, use_container_width=True)
    else:
        st.info('Run the notebook to generate subgroup figures.')
    st.markdown('</div>', unsafe_allow_html=True)

    # Summary table
    summary_path = os.path.join(tdir, 'table05_fairness_summary.csv')
    st.markdown('<div class="section-card"><h3>Fairness Metrics Summary Table</h3>', unsafe_allow_html=True)
    if os.path.exists(summary_path):
        fair_sum = pd.read_csv(summary_path, index_col=0)

        def highlight_threshold(val):
            try:
                v = float(val)
                if v > 0.10:
                    return 'background-color:#f8d7da;color:#721c24;font-weight:600;'
                elif v > 0.05:
                    return 'background-color:#fff3cd;color:#856404;'
                else:
                    return 'background-color:#d4edda;color:#155724;'
            except:
                return ''

        st.dataframe(
            fair_sum.style.applymap(highlight_threshold).format('{:.4f}'),
            use_container_width=True
        )

        # Per-attribute interpretation
        for attr, row in fair_sum.iterrows():
            flags = [m for m, v in row.items() if float(v) > 0.10]
            if flags:
                st.markdown(f'<div class="fairness-flag"><strong>{attr}:</strong> Exceeds threshold on {flags}. This may indicate the model performs inconsistently across subgroups defined by this attribute and should be investigated before deployment.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="fairness-ok"><strong>{attr}:</strong> All fairness metrics within the 0.10 threshold. Performance appears broadly equitable across subgroups.</div>', unsafe_allow_html=True)
    else:
        st.info('Run the notebook to generate the fairness summary table.')

    st.markdown('</div>', unsafe_allow_html=True)

    # Educational recommendations
    st.markdown('<div class="section-card"><h3>Educational Recommendations</h3>', unsafe_allow_html=True)
    recs = [
        'Focus early intervention support on students with two or more prior academic failures, as SHAP analysis identifies this as the strongest negative predictor of passing.',
        'Attendance monitoring should be prioritised: absenteeism is the second strongest predictor of failure, and its impact compounds with poor health (the health_x_absences interaction feature).',
        'Students from households with lower parental education levels show both lower baseline pass rates and may receive less equitable model predictions — targeted outreach to these families is warranted.',
        'Predictions should be used to support — not replace — educator judgement. Model outputs carry inherent uncertainty and should be reviewed in the context of individual student circumstances.',
        'The model should be periodically retrained as student cohort characteristics evolve, and fairness metrics should be re-evaluated with each update.',
    ]
    for rec in recs:
        st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown('---')
st.markdown(f"""
<div style="text-align:center;font-size:0.78em;color:#888;padding:8px 0;">
Prabin Pokhrel | 2308806 | XAI for Student Performance Prediction
</div>
""", unsafe_allow_html=True)
