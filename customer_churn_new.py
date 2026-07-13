# Importing recquired Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")

# Load Dataset
# Dataset is taken from Kaggle 
train = pd.read_csv("customer_churn_dataset-training-master.csv")
test = pd.read_csv("customer_churn_dataset-testing-master.csv")

print("Train Shape :", train.shape)
print("Test Shape  :", test.shape)

print("\nFirst 5 Rows")
print(train.head())

print("\nDataset Info")
train.info()

# Data Preprocessing

train.dropna(inplace=True)
train.drop_duplicates(inplace=True)

train.drop("CustomerID", axis=1, inplace=True)
test.drop("CustomerID", axis=1, inplace=True)

encoder = LabelEncoder()

for col in ["Gender", "Subscription Type", "Contract Length"]:
    train[col] = encoder.fit_transform(train[col])
    test[col] = encoder.transform(test[col])

print("\nChurn Distribution:\n")
print(train["Churn"].value_counts())

# Churn Distribution
plt.figure(figsize=(5,4))
sns.countplot(x="Churn", data=train)
plt.title("Customer Churn Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(train.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print(train.dtypes)


# Train-Test Split & Models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Features and Target
X = train.drop("Churn", axis=1)
y = train["Churn"]

print(X.dtypes)
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Model Training & Evaluation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model_results = []

for name, model in models.items():

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    model_results.append([
        name,
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    ])

results = pd.DataFrame(
    model_results,
    columns=["Model","Accuracy","Precision","Recall","F1-Score"]
)

print(results)

# Feature Importance & Conclusion
# Select Best Model
best_model = models["Random Forest"]

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=importance.head(10), x="Importance", y="Feature")
plt.title("Top 10 Important Features")
plt.show()

print("\nModel Performance")
print(results)

print("\nBest Model:")
print(results.sort_values(by="Accuracy", ascending=False).iloc[0])