import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from shap import TreeExplainer, summary_plot
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('gym_churn_us.csv')

# Basic data info
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# Convert binary columns to boolean
binary_cols = ['gender', 'Near_Location', 'Partner', 'Promo_friends', 'Phone', 'Group_visits', 'Churn']
df[binary_cols] = df[binary_cols].astype(bool)

# EDA
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Distribution of numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
for i, col in enumerate(numeric_cols):
    sns.histplot(data=df, x=col, hue='Churn', multiple='stack', ax=axes[i//3, i%3])
plt.tight_layout()
plt.show()

# Churn rate by categorical features
cat_cols = ['gender', 'Near_Location', 'Partner', 'Promo_friends', 'Phone', 'Group_visits']
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
for i, col in enumerate(cat_cols):
    sns.barplot(x=col, y='Churn', data=df, ax=axes[i//3, i%3])
plt.tight_layout()
plt.show()

# Feature Engineering
df['Contract_length'] = df['Contract_period'].map({1: 'Short', 6: 'Medium', 12: 'Long'})
df['Time_left_ratio'] = df['Month_to_end_contract'] / df['Contract_period']
df['Avg_additional_charges_per_month'] = df['Avg_additional_charges_total'] / df['Lifetime']
df['Class_frequency_change'] = df['Avg_class_frequency_current_month'] - df['Avg_class_frequency_total']
df['Is_regular'] = df['Avg_class_frequency_total'] > df['Avg_class_frequency_total'].median()

# Prepare data for modeling
X = df.drop(['Churn', 'Contract_length'], axis=1)
y = df['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Selection and Training
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))

# Hyperparameter Tuning for the best model (assuming XGBoost performs better)
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Model Interpretation
explainer = TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled)
summary_plot(shap_values, X_test_scaled, plot_type="bar", feature_names=X.columns)
plt.show()

# Customer Segmentation
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_train_scaled)

# Analyze segments
segment_profile = df.groupby('Cluster').mean()
print("\nSegment Profiles:")
print(segment_profile)

# Churn rate by segment
churn_by_segment = df.groupby('Cluster')['Churn'].mean()
print("\nChurn Rate by Segment:")
print(churn_by_segment)