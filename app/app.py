import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve,
                             precision_recall_curve, roc_auc_score,
                             average_precision_score)
import plotly.express as px
import plotly.graph_objects as go
import shap
import os

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# ── load assets ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")

@st.cache_resource
def load_model_assets():
    model         = joblib.load("models/fraud_model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    eval_data     = joblib.load("models/eval_data.pkl")
    return model, scaler, feature_names, eval_data

df = load_data()
model, scaler, feature_names, eval_data = load_model_assets()

X_test      = eval_data['X_test']
y_test      = eval_data['y_test']
y_proba     = eval_data['y_proba']
shap_values = eval_data['shap_values']
X_sample    = eval_data['X_sample']

# ── FIX: medians from X_test (has scaled columns), not raw df ──────────────
feature_medians = {f: float(X_test[f].median()) for f in feature_names}

# ── sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("💳 Fraud Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Dataset Overview",
    "🔍 Live Transaction Checker",
    "📈 Model Performance",
    "🧠 SHAP Explainability"
])

threshold = st.sidebar.slider(
    "Decision threshold", 0.01, 0.99, 0.50, 0.01,
    help="Lower = catch more fraud (more false alarms). Higher = fewer false alarms (miss more fraud)."
)

y_pred_thresh = (y_proba >= threshold).astype(int)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** XGBoost + SMOTE")
st.sidebar.markdown("**Dataset:** Kaggle Credit Card Fraud")
st.sidebar.markdown(f"**Transactions:** {len(df):,}")
st.sidebar.markdown(f"**Fraud rate:** {df['Class'].mean()*100:.4f}%")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("Exploring the Kaggle Credit Card Fraud dataset — 284,807 transactions over 2 days.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total transactions", f"{len(df):,}")
    col2.metric("Fraud cases",        f"{df['Class'].sum():,}")
    col3.metric("Fraud rate",         f"{df['Class'].mean()*100:.4f}%")
    col4.metric("Avg fraud amount",   f"${df[df['Class']==1]['Amount'].mean():.2f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Class distribution")
        fig = px.bar(
            x=["Legitimate", "Fraud"],
            y=[df['Class'].value_counts()[0], df['Class'].value_counts()[1]],
            color=["Legitimate", "Fraud"],
            color_discrete_map={"Legitimate": "#378ADD", "Fraud": "#E24B4A"},
            labels={"x": "Class", "y": "Count"}
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("Transaction amount by class")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=df[df['Class']==0]['Amount'],
            name='Legitimate', marker_color='#378ADD', opacity=0.7, nbinsx=60))
        fig2.add_trace(go.Histogram(x=df[df['Class']==1]['Amount'],
            name='Fraud', marker_color='#E24B4A', opacity=0.7, nbinsx=60))
        fig2.update_layout(barmode='overlay', height=300,
            xaxis_title="Amount (USD)", yaxis_title="Count")
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Fraud rate by hour of day")
    df['Hour'] = (df['Time'] // 3600) % 24
    hourly = df.groupby(['Hour','Class']).size().unstack().fillna(0)
    hourly['fraud_rate'] = hourly[1] / (hourly[0] + hourly[1]) * 100
    fig3 = px.bar(hourly, y='fraud_rate',
        labels={'fraud_rate': 'Fraud rate (%)', 'Hour': 'Hour of day'},
        color_discrete_sequence=['#E24B4A'])
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, width='stretch')

    st.subheader("Raw data preview")
    st.dataframe(df.head(20), width='stretch')

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE TRANSACTION CHECKER
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Transaction Checker":
    st.title("🔍 Live Transaction Checker")
    st.markdown("Adjust the sliders to simulate a transaction and get a real-time fraud prediction.")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.slider("Transaction amount ($)", 0.0, 5000.0, 150.0, 1.0)
        hour   = st.slider("Hour of day", 0, 23, 14)
        v1  = st.slider("V1",  float(df['V1'].min()),  float(df['V1'].max()),  0.0, 0.01)
        v2  = st.slider("V2",  float(df['V2'].min()),  float(df['V2'].max()),  0.0, 0.01)
        v3  = st.slider("V3",  float(df['V3'].min()),  float(df['V3'].max()),  0.0, 0.01)
    with col2:
        v4  = st.slider("V4",  float(df['V4'].min()),  float(df['V4'].max()),  0.0, 0.01)
        v14 = st.slider("V14", float(df['V14'].min()), float(df['V14'].max()), 0.0, 0.01)
        v17 = st.slider("V17", float(df['V17'].min()), float(df['V17'].max()), 0.0, 0.01)
        v12 = st.slider("V12", float(df['V12'].min()), float(df['V12'].max()), 0.0, 0.01)
        v10 = st.slider("V10", float(df['V10'].min()), float(df['V10'].max()), 0.0, 0.01)

    if st.button("🔍 Analyse Transaction", type="primary"):
        # FIX: build input from X_test medians (correct scaled feature space)
        input_dict = feature_medians.copy()
        input_dict.update({
            'Amount_scaled': float(scaler.transform([[amount]])[0][0]),
            'Time_scaled':   float(scaler.transform([[hour * 3600]])[0][0]),
            'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4,
            'V14': v14, 'V17': v17, 'V12': v12, 'V10': v10
        })
        input_df = pd.DataFrame([input_dict])[feature_names]

        prob = model.predict_proba(input_df)[0][1]
        pred = int(prob >= threshold)

        st.markdown("---")
        col_r, col_g = st.columns(2)
        with col_r:
            if pred == 1:
                st.error(f"🚨 **FRAUD DETECTED**\n\nProbability: **{prob*100:.1f}%**")
            else:
                st.success(f"✅ **LEGITIMATE TRANSACTION**\n\nProbability of fraud: **{prob*100:.1f}%**")

        with col_g:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                title={'text': "Fraud probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#E24B4A" if pred == 1 else "#378ADD"},
                    'steps': [
                        {'range': [0,  30], 'color': "#E8F5E9"},
                        {'range': [30, 60], 'color': "#FFF9C4"},
                        {'range': [60, 100],'color': "#FFEBEE"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 3},
                                  'thickness': 0.75, 'value': threshold * 100}
                }
            ))
            fig_g.update_layout(height=250)
            st.plotly_chart(fig_g, width='stretch')

        st.subheader("Why this prediction? (SHAP explanation)")
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(input_df)
        fig_w, ax = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig_w)
        plt.close()
        st.caption("Red bars push toward fraud. Blue bars push toward legitimate.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown(f"Evaluating at threshold = **{threshold:.2f}** (adjust in sidebar)")

    from sklearn.metrics import precision_score, recall_score, f1_score

    tp = int(((y_pred_thresh == 1) & (y_test == 1)).sum())
    fp = int(((y_pred_thresh == 1) & (y_test == 0)).sum())
    fn = int(((y_pred_thresh == 0) & (y_test == 1)).sum())
    tn = int(((y_pred_thresh == 0) & (y_test == 0)).sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("ROC-AUC",   f"{roc_auc_score(y_test, y_proba):.4f}")
    col2.metric("PR-AUC",    f"{average_precision_score(y_test, y_proba):.4f}")
    col3.metric("Precision", f"{precision_score(y_test, y_pred_thresh):.4f}")
    col4.metric("Recall",    f"{recall_score(y_test, y_pred_thresh):.4f}")
    col5.metric("F1 Score",  f"{f1_score(y_test, y_pred_thresh):.4f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, y_pred_thresh)
        fig_cm = px.imshow(cm, text_auto=True,
            x=['Predicted Legit', 'Predicted Fraud'],
            y=['Actual Legit',    'Actual Fraud'],
            color_continuous_scale='Blues')
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, width='stretch')
        st.caption(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

    with col_b:
        st.subheader("ROC curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
            name=f'XGBoost (AUC={roc_auc_score(y_test, y_proba):.3f})',
            line=dict(color='#378ADD', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
            name='Random', line=dict(color='gray', dash='dash')))
        fig_roc.update_layout(xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate', height=350)
        st.plotly_chart(fig_roc, width='stretch')

    st.subheader("Precision-Recall curve")
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode='lines',
        name=f'XGBoost (PR-AUC={average_precision_score(y_test, y_proba):.3f})',
        line=dict(color='#E24B4A', width=2)))
    fig_pr.update_layout(xaxis_title='Recall',
        yaxis_title='Precision', height=300)
    st.plotly_chart(fig_pr, width='stretch')
    st.caption("PR curve is more informative than ROC for imbalanced datasets.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainability":
    st.title("🧠 SHAP Explainability")
    st.markdown("Understanding **why** the model flags transactions — not just that it does.")

    tab1, tab2 = st.tabs(["🌍 Global importance", "🔬 Beeswarm plot"])

    with tab1:
        st.subheader("Feature importance (mean |SHAP|)")
        if os.path.exists("reports/shap_importance.png"):
            st.image("reports/shap_importance.png", width='stretch')
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
            st.pyplot(fig)
            plt.close()
        st.markdown("""
        **How to read this:** Features with longer bars have more influence on predictions.
        - **V14, V17, V12, V10** are consistently the strongest fraud signals
        - These correspond to specific spending pattern anomalies in the original (anonymized) data
        """)

    with tab2:
        st.subheader("SHAP beeswarm — feature impact direction")
        if os.path.exists("reports/shap_beeswarm.png"):
            st.image("reports/shap_beeswarm.png", width='stretch')
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(fig)
            plt.close()
        st.markdown("""
        **How to read this:**
        - Each dot = one transaction
        - **Red dots** = high feature value, **Blue dots** = low feature value
        - Position on X axis = impact on fraud prediction
        - Red dot on the right = high value of that feature pushes toward fraud
        """)
