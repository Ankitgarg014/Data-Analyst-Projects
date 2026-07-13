# Importing recquired libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")


# Load Dataset
# Dataset is taken from kaggle

df = pd.read_csv("credit_risk_dataset.csv")

print("=" * 60)
print("CREDIT RISK DATASET OVERVIEW")
print("=" * 60)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all"))

# Exploratory Data Analysis
# Missing and Duplicate Values

print("Missing Values:\n")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Target Variable Distribution
print("\nLoan Status Distribution:\n")
print(df["loan_status"].value_counts())

print("\nNumerical Summary:\n")
print(df.describe())

# Plot Target Distribution
plt.figure(figsize=(5,4))
df["loan_status"].value_counts().plot(kind="bar")
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
plt.imshow(df.corr(numeric_only=True), cmap="Blues")
plt.colorbar()

plt.xticks(range(len(df.corr(numeric_only=True).columns)),
           df.corr(numeric_only=True).columns,
           rotation=90)

plt.yticks(range(len(df.corr(numeric_only=True).columns)),
           df.corr(numeric_only=True).columns)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Data Preprocessing

df = df.dropna()

# Encode Categorical Columns
encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# Features and Target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Set:", X_train.shape)
print("Testing Set :", X_test.shape)

# Handle Imbalanced Data
print("Before SMOTE:\n", y_train.value_counts())

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:\n", y_train.value_counts())

# Model Building & Evaluation
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = []

for name, model in models.items():
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, accuracy, precision, recall, f1])

# Display Results
results = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

print(results)

import shap
# Explain Random Forest Model
explainer = shap.TreeExplainer(models["Random Forest"])
shap_values = explainer.shap_values(X_test, check_additivity=False)
shap.summary_plot(shap_values, X_test)

print("\nProject Completed Successfully!")
best_model = results.sort_values("Accuracy", ascending=False).iloc[0]
print("\nBest Model:")
print(best_model)