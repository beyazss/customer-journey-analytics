import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data/clean_data.csv', parse_dates=['InvoiceDate'])
df = df.sort_values('InvoiceDate')

# Churn definition - temporal split
split_date = df['InvoiceDate'].quantile(0.75)

early = df[df['InvoiceDate'] <= split_date]
late = df[df['InvoiceDate'] > split_date]

# Features from early period
early_customers = early.groupby('Customer ID').agg(
    Frequency=('Invoice', 'nunique'),
    Monetary=('TotalPrice', 'sum'),
    avg_order_value=('TotalPrice', 'mean'),
    unique_products=('StockCode', 'nunique'),
    avg_quantity=('Quantity', 'mean'),
    std_days_between=('InvoiceDate', lambda x: x.sort_values().diff().dt.days.std())
).reset_index()

early_customers['std_days_between'] = early_customers['std_days_between'].fillna(0)

# Label: did they buy in the late period?
late_buyers = set(late['Customer ID'].unique())
early_customers['churned'] = (~early_customers['Customer ID'].isin(late_buyers)).astype(int)

print(f"Churned: {early_customers['churned'].sum()} ({early_customers['churned'].mean():.1%})")
print(f"Active : {(early_customers['churned']==0).sum()} ({(early_customers['churned']==0).mean():.1%})")

# Train/test split
feature_cols = ['Frequency', 'Monetary', 'avg_order_value',
                'unique_products', 'avg_quantity', 'std_days_between']

X = early_customers[feature_cols]
y = early_customers['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
axes[0].barh(importances.index, importances.values, color='#534AB7')
axes[0].set_title('Feature Importance')

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
axes[1].plot(fpr, tpr, color='#534AB7', label=f'AUC = {auc:.3f}')
axes[1].plot([0, 1], [0, 1], '--', color='gray')
axes[1].set_title('ROC Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

# Churn probability distribution
early_customers['churn_prob'] = model.predict_proba(
    scaler.transform(early_customers[feature_cols])
)[:, 1]

axes[2].hist(early_customers[early_customers['churned']==0]['churn_prob'],
             bins=30, alpha=0.6, color='#1D9E75', label='Active')
axes[2].hist(early_customers[early_customers['churned']==1]['churn_prob'],
             bins=30, alpha=0.6, color='#D85A30', label='Churned')
axes[2].set_title('Churn Probability Distribution')
axes[2].set_xlabel('Churn Probability')
axes[2].legend()

plt.tight_layout()
plt.savefig('data/churn_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# Save scores
early_customers[['Customer ID', 'Frequency', 'Monetary', 'churn_prob', 'churned']].to_csv(
    'data/churn_scores.csv', index=False)
print("\nSaved: data/churn_scores.csv")