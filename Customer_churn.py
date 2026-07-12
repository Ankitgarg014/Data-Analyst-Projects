# IMPORT 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Data Loading, Data Cleaning, Initial Exploration
import joblib

# Dataset is taken from Kaggle  
train = pd.read_csv("customer_churn_dataset-training-master.csv")
test = pd.read_csv("customer_churn_dataset-testing-master.csv")
print(train.head())
print(train.tail())

# Checking how many rows and columns each dataset has
print("Training Dataset Shape :", train.shape)
print("Testing Dataset Shape :", test.shape)
train.columns
train.info()
train.describe()
train.isnull().sum()
(train.isnull().sum()/len(train))*100
train.dropna(inplace=True)
train.isnull().sum()
train.duplicated().sum()
train.drop_duplicates(inplace=True)
train.shape
train.dtypes

# Looking at how many unique/different values are in each column -
for col in train.columns:
    print(f"{col} : {train[col].nunique()}")
train.drop("CustomerID",axis=1,inplace=True)
test.drop("CustomerID",axis=1,inplace=True)
encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

for col in categorical_columns:
    train[col] = encoder.fit_transform(train[col])
    test[col] = encoder.transform(test[col])
train.head()
train.corr()
train.to_csv("clean_train.csv",index=False)

# Data in visualisation form 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.figure(figsize=(6,5))
sns.countplot(x='Churn', data=train)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(x='Gender', hue='Churn', data=train)
plt.title("Gender vs Churn")
plt.xlabel("Gender (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(7,5))
sns.countplot(x='Subscription Type', hue='Churn', data=train)
plt.title("Subscription Type vs Churn")
plt.xlabel("Subscription Type")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(7,5))
sns.countplot(x='Contract Length', hue='Churn', data=train)
plt.title("Contract Length vs Churn")
plt.xlabel("Contract Length")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(train['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(train['Tenure'], bins=30, kde=True)
plt.title("Tenure Distribution")
plt.xlabel("Tenure")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(train['Total Spend'], bins=30, kde=True)
plt.title("Total Spend Distribution")
plt.xlabel("Total Spend")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x='Churn', y='Payment Delay', data=train)
plt.title("Payment Delay vs Churn")
plt.xlabel("Churn")
plt.ylabel("Payment Delay")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x='Churn', y='Support Calls', data=train)
plt.title("Support Calls vs Churn")
plt.xlabel("Churn")
plt.ylabel("Support Calls")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x='Churn', y='Usage Frequency', data=train)
plt.title("Usage Frequency vs Churn")
plt.xlabel("Churn")
plt.ylabel("Usage Frequency")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x='Churn', y='Last Interaction', data=train)
plt.title("Last Interaction vs Churn")
plt.xlabel("Churn")
plt.ylabel("Last Interaction")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

train.hist(figsize=(16,12), bins=20)
plt.suptitle("Distribution of Numerical Features", fontsize=18)
plt.show()

churn_percent = train['Churn'].value_counts(normalize=True) * 100

plt.figure(figsize=(6,6))
plt.pie(
    churn_percent,
    labels=['Stayed','Churned'],
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Customer Churn Percentage")
plt.show()

#Comparing the average value of every column between churned and
# non-churned customers - a fast way to spot obvious differences

average_values = train.groupby('Churn').mean(numeric_only=True)
print("\nAverage Feature Values by Churn:\n")
print(average_values)

sample = train.sample(500, random_state=42)

sns.pairplot(
    sample,
    vars=['Age','Tenure','Total Spend','Usage Frequency'],
    hue='Churn'
)

plt.show()

print("="*60)
print("Exploratory Data Analysis Completed Successfully")
print("="*60)

# MODEL BUILDING 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X = train.drop("Churn", axis=1)
y = train["Churn"]

print("Feature Matrix Shape :", X.shape)
print("Target Shape :", y.shape)

# Splitting the data into a training portion (80%) and a testing portion (20%).
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000, random_state=42)

tree_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=8,
    random_state=42
)

forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)


# Training (teaching) each model using the training data
print("\nTraining Logistic Regression...")
log_model.fit(X_train_scaled, y_train)

print("Training Decision Tree...")
tree_model.fit(X_train, y_train)

print("Training Random Forest...")
forest_model.fit(X_train, y_train)

print("\nAll Models Trained Successfully!")

# Getting the probability each model gives for "this customer will churn".
log_pred = log_model.predict(X_test_scaled)
tree_pred = tree_model.predict(X_test)
forest_pred = forest_model.predict(X_test)

log_prob = log_model.predict_proba(X_test_scaled)[:,1]
tree_prob = tree_model.predict_proba(X_test)[:,1]
forest_prob = forest_model.predict_proba(X_test)[:,1]

def evaluate_model(name, actual, prediction, probability):

    accuracy = accuracy_score(actual, prediction)
    precision = precision_score(actual, prediction)
    recall = recall_score(actual, prediction)
    f1 = f1_score(actual, prediction)
    roc = roc_auc_score(actual, probability)
# how well the model separates churners from non-churners overall
    print("="*60)
    print(f"{name}")
    print("="*60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc:.4f}")

    print("\nClassification Report\n")
    print(classification_report(actual, prediction))

    return [
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc
    ]
# Running the evaluation for each of the three models and storing the results.
results = []
results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        log_pred,
        log_prob
    )
)
results.append(
    evaluate_model(
        "Decision Tree",
        y_test,
        tree_pred,
        tree_prob
    )
)
results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        forest_pred,
        forest_prob
    )
)

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ]
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

print("\n")
print("="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)

print(results_df)
# Keeping a reference to the actual best model object (not just its name),
# so I can use it directly in the next steps
best_model_name = results_df.iloc[0]["Model"]

print("\nBest Performing Model :", best_model_name)

if best_model_name == "Logistic Regression":
    best_model = log_model
elif best_model_name == "Decision Tree":
    best_model = tree_model
else:
    best_model = forest_model

print("\nFeature Columns Used:")
print(list(X.columns))

print("\nTraining Completed Successfully.")
print("="*70)

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import joblib

model = forest_model

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=importance, x="Importance", y="Feature")
plt.title("Feature Importance")
plt.show()

print("\nTop 10 Important Features")
print(importance.head(10))
