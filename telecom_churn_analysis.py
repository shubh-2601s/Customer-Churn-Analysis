import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import eli5
from eli5.sklearn import PermutationImportance
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for plots
os.makedirs("plots", exist_ok=True)

# Step 1: Load Data
df = pd.read_csv('telecom_customers.csv')

# Convert 'last_recharge_date' to datetime and create recency feature
df['last_recharge_date'] = pd.to_datetime(df['last_recharge_date'], errors='coerce')
df['recency_days'] = (pd.Timestamp.today() - df['last_recharge_date']).dt.days
df['recency_days'].fillna(df['recency_days'].median(), inplace=True)

print("Data Sample:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nChurn distribution:")
print(df['churn_flag'].value_counts())

# Step 2: Data Preprocessing

numeric_cols = ['age', 'monthly_charges', 'recency_days', 'calls_made', 'call_duration_total', 
                'num_complaints', 'internet_usage_GB', 'support_tickets', 'total_recharges']

for col in numeric_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

categorical_cols = ['gender', 'region', 'plan_type', 'contract_type']
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Features and target
X = df.drop(columns=['customer_id', 'churn_flag', 'last_recharge_date', 'churn'], errors='ignore')
y = df['churn_flag']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# Step 3: Model Training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 4: Prediction and Evaluation
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Step 5: ELI5 Permutation Importance
print("\nPermutation Importance (ELI5):")
perm = PermutationImportance(clf, random_state=42).fit(X_test, y_test)
with open("plots/eli5_feature_weights.html", "w", encoding="utf-8") as f:
    f.write(eli5.format_as_html(eli5.explain_weights(perm, feature_names=X.columns.tolist())))

# Step 6: SHAP Explainability
print("\nGenerating SHAP summary plot...")
explainer = shap.Explainer(clf, X_train)
shap_values = explainer(X_test)

# Save SHAP summary (dot) plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("plots/shap_summary_dot.png", bbox_inches="tight")
plt.close()

# Save SHAP summary (bar) plot
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("plots/shap_summary_bar.png", bbox_inches="tight")
plt.close()

# Step 7: Customer Segmentation
df_test = X_test.copy()
df_test['churn_proba'] = y_proba
df_test['churn_flag'] = y_test.values

def segment_customer(prob):
    if prob > 0.7:
        return 'At Risk'
    elif prob < 0.3:
        return 'Loyal'
    else:
        return 'Dormant'

df_test['segment'] = df_test['churn_proba'].apply(segment_customer)

print("\nCustomer Segment Distribution:")
print(df_test['segment'].value_counts())

# Plot segment distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df_test, x='segment', order=['Loyal', 'Dormant', 'At Risk'], palette='coolwarm')
plt.title("Customer Segments Based on Churn Probability")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("plots/customer_segments.png")
plt.close()

# Step 8: Churn Rate by Region (Optional but insightful)
if 'region' in label_encoders:
    df['region'] = label_encoders['region'].inverse_transform(df['region'])
    region_churn = df.groupby('region')['churn_flag'].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    region_churn.plot(kind='bar', color='teal')
    plt.title("Churn Rate by Region")
    plt.ylabel("Churn Rate")
    plt.xlabel("Region")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/churn_by_region.png")
    plt.close()
