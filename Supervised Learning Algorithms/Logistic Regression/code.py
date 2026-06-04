import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("telecom_churn.csv")

print("="*50)
print("DATASET OVERVIEW")
print("="*50)

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:\n")
print(df.head())

# ==========================================
# 2. FEATURE SELECTION
# ==========================================

features = [
    "age",
    "num_dependents",
    "estimated_salary",
    "calls_made",
    "sms_sent",
    "data_used"
]

X = df[features]

y = df["churn"]

# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# 4. MODEL TRAINING
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ==========================================
# 5. PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 6. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n")
print("="*50)
print("MODEL PERFORMANCE")
print("="*50)

print(f"Accuracy : {accuracy*100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# ==========================================
# GRAPH 1
# CHURN DISTRIBUTION
# ==========================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="churn"
)

plt.title("Customer Churn Distribution")

plt.savefig("graph1_churn_distribution.png")

plt.show()

# ==========================================
# GRAPH 2
# AGE VS CHURN
# ==========================================

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="churn",
    y="age"
)

plt.title("Age vs Customer Churn")

plt.savefig("graph2_age_vs_churn.png")

plt.show()

# ==========================================
# GRAPH 3
# SALARY VS CHURN
# ==========================================

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="churn",
    y="estimated_salary"
)

plt.title("Salary vs Customer Churn")

plt.savefig("graph3_salary_vs_churn.png")

plt.show()

# ==========================================
# GRAPH 4
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("graph4_confusion_matrix.png")

plt.show()

# ==========================================
# GRAPH 5
# LOGISTIC REGRESSION CURVE
# ==========================================

X_single = df[["estimated_salary"]]

y_single = df["churn"]

curve_model = LogisticRegression(
    max_iter=1000
)

curve_model.fit(
    X_single,
    y_single
)

salary_range = np.linspace(
    X_single.min().values[0],
    X_single.max().values[0],
    500
).reshape(-1,1)

probabilities = curve_model.predict_proba(
    salary_range
)[:,1]

plt.figure(figsize=(10,6))

sns.scatterplot(
    x=df["estimated_salary"],
    y=df["churn"],
    alpha=0.25
)

plt.plot(
    salary_range,
    probabilities,
    linewidth=3
)

plt.title(
    "Logistic Regression Sigmoid Curve"
)

plt.xlabel(
    "Estimated Salary"
)

plt.ylabel(
    "Probability of Churn"
)

plt.grid(True)

plt.savefig(
    "graph5_logistic_regression_curve.png"
)

plt.show()

# ==========================================
# SAMPLE PREDICTION
# ==========================================

sample_customer = pd.DataFrame({
    "age":[35],
    "num_dependents":[2],
    "estimated_salary":[60000],
    "calls_made":[120],
    "sms_sent":[50],
    "data_used":[15]
})

prediction = model.predict(
    sample_customer
)[0]

probability = model.predict_proba(
    sample_customer
)[0][1]

print("\n")
print("="*50)
print("SAMPLE PREDICTION")
print("="*50)

print(
    f"Probability of Churn: {probability:.2%}"
)

if prediction == 1:
    print("Prediction: Customer Will Churn")
else:
    print("Prediction: Customer Will Stay")
