import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no popup windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import joblib
import os

# ── paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/creditcard.csv"
MODEL_PATH  = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"
REPORT_DIR  = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 50)
print("STEP 1 — Loading data")
print("=" * 50)
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")

# ── EDA charts ─────────────────────────────────────────────────────────────
print("\nSTEP 2 — Saving EDA charts")

# Class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df, palette=['#378ADD','#E24B4A'])
plt.xticks([0,1], ['Legitimate','Fraud'])
plt.title('Class Distribution')
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/class_distribution.png", dpi=150)
plt.close()
print("  ✓ class_distribution.png")

# Amount by class
fig, axes = plt.subplots(1, 2, figsize=(12,4))
df[df['Class']==0]['Amount'].plot(kind='hist', bins=50, ax=axes[0],
    color='#378ADD', alpha=0.7, title='Legitimate — Amount')
df[df['Class']==1]['Amount'].plot(kind='hist', bins=50, ax=axes[1],
    color='#E24B4A', alpha=0.7, title='Fraud — Amount')
for ax in axes:
    ax.set_xlabel('Amount (USD)')
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/amount_by_class.png", dpi=150)
plt.close()
print("  ✓ amount_by_class.png")

# Fraud by hour
df['Hour'] = (df['Time'] // 3600) % 24
hourly = df.groupby(['Hour','Class']).size().unstack().fillna(0)
hourly['fraud_rate'] = hourly[1] / (hourly[0] + hourly[1]) * 100
plt.figure(figsize=(12,4))
plt.bar(hourly.index, hourly['fraud_rate'], color='#E24B4A', alpha=0.8)
plt.xlabel('Hour of day')
plt.ylabel('Fraud rate (%)')
plt.title('Fraud Rate by Hour of Day')
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/fraud_by_hour.png", dpi=150)
plt.close()
print("  ✓ fraud_by_hour.png")

# ── preprocessing ──────────────────────────────────────────────────────────
print("\nSTEP 3 — Preprocessing")

scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled']   = scaler.fit_transform(df[['Time']])

features = [c for c in df.columns if c not in ['Class','Amount','Time','Hour']]
X = df[features]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

# SMOTE — fix class imbalance on training set only
print("  Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"  After SMOTE — fraud: {y_train_sm.sum()}, legit: {(y_train_sm==0).sum()}")

# ── baseline model ─────────────────────────────────────────────────────────
print("\nSTEP 4 — Baseline: Logistic Regression")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_sm, y_train_sm)
lr_pred = lr.predict(X_test)
print(classification_report(y_test, lr_pred, target_names=['Legit','Fraud']))

# ── XGBoost ────────────────────────────────────────────────────────────────
print("\nSTEP 5 — XGBoost model")
scale_pos = (y_train_sm == 0).sum() / y_train_sm.sum()
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos,
    use_label_encoder=False,
    eval_metric='aucpr',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_sm, y_train_sm,
          eval_set=[(X_test, y_test)],
          verbose=50)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("\n── XGBoost Results ──")
print(classification_report(y_test, y_pred, target_names=['Legit','Fraud']))
print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR-AUC:   {average_precision_score(y_test, y_proba):.4f}")

# confusion matrix chart
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legit','Fraud'],
            yticklabels=['Legit','Fraud'])
plt.title('Confusion Matrix — XGBoost')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/confusion_matrix.png", dpi=150)
plt.close()
print("  ✓ confusion_matrix.png")

# ── SHAP ───────────────────────────────────────────────────────────────────
print("\nSTEP 6 — SHAP explainability (this takes ~1 min)")
explainer   = shap.TreeExplainer(model)
X_sample    = X_test.sample(500, random_state=42)
shap_values = explainer.shap_values(X_sample)

# Global feature importance
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/shap_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ shap_importance.png")

# Beeswarm
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ shap_beeswarm.png")

# ── save model ─────────────────────────────────────────────────────────────
print("\nSTEP 7 — Saving model and scaler")
joblib.dump(model,  MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(X_test.columns.tolist(), "models/feature_names.pkl")
joblib.dump({'X_test': X_test, 'y_test': y_test,
             'y_proba': y_proba, 'shap_values': shap_values,
             'X_sample': X_sample}, "models/eval_data.pkl")
print(f"  ✓ Model saved → {MODEL_PATH}")
print(f"  ✓ Eval data saved → models/eval_data.pkl")
print("\n Training complete! Run: streamlit run app/app.py")