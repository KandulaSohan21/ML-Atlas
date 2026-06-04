# 📊 Results Explainationpp

## Dataset Overview

The telecom churn dataset was used to predict whether a customer would leave the service.

### Target Variable

| Value | Meaning         |
| ----- | --------------- |
| 0     | Customer Stays  |
| 1     | Customer Churns |

---

## Generated Visualizations

### Customer Churn Distribution

![Customer Churn Distribution](https://github.com/KandulaSohan21/ML-Atlas/blob/7873f445ed77e097942a9c68a28cb2c27bc300bc/Supervised%20Learning%20Algorithms/Logistic%20Regression/Results/Custom%20churn%20Distribution.png)

#### Observation

The dataset is highly imbalanced.

* Most customers remain with the company.
* A smaller portion of customers churn.

This imbalance can make classification difficult because the model tends to favor the majority class.

---

### Age vs Customer Churn

![Age vs Customer Churn](https://github.com/KandulaSohan21/ML-Atlas/blob/7eb4908f397b4c9c2c85dcf3881f745e4916218e/Supervised%20Learning%20Algorithms/Logistic%20Regression/Results/Age%20vs%20Customer%20churn.png)

#### Observation

The age distribution of churned and non-churned customers is very similar.

This suggests that age alone is not a strong predictor of customer churn.

---

### Salary vs Customer Churn

![Salary vs Customer Churn](https://github.com/KandulaSohan21/ML-Atlas/blob/e86e0f209dc9df6742872c39c89535c1068fa508/Supervised%20Learning%20Algorithms/Logistic%20Regression/Results/salary%20vs%20customer%20churn.png).

#### Observation

Estimated salary shows a similar distribution across both classes.

This indicates that salary has limited influence on churn behavior in this dataset.

---

### Confusion Matrix

![Confusion Matrix](https://github.com/KandulaSohan21/ML-Atlas/blob/8ec7359f4aec2881c57df0b33f639904266d5caa/Supervised%20Learning%20Algorithms/Logistic%20Regression/Results/Confusion%20Matrix.png)

#### Observation

The model correctly identified a large number of non-churning customers.

However, it failed to identify customers who actually churned.

This indicates that the model is biased toward the majority class.

### Confusion Matrix Summary

| Actual    | Predicted Stay | Predicted Churn |
| --------- | -------------- | --------------- |
| Stay (0)  | 38,928         | 0               |
| Churn (1) | 9,783          | 0               |

---

### Logistic Regression Sigmoid Curve

![Logistic Regression Curve](results/graph5_logistic_regression_curve.png)

#### Observation

The sigmoid curve appears nearly flat.

This suggests that estimated salary alone is not strongly correlated with customer churn.

The model struggles to separate churning customers from non-churning customers using this single feature.

---

## Key Findings

### ✅ Logistic Regression Successfully Trained

The model was successfully trained using customer demographic and usage information.

### ⚠ Dataset Imbalance Affected Performance

The dataset contains significantly more non-churning customers than churning customers.

As a result, the model learned to predict the majority class.

### ⚠ Selected Features Have Weak Predictive Power

Features such as age and estimated salary show little separation between churned and non-churned customers.

### ⚠ Churn Prediction Requires Additional Features

More informative features may improve performance, including:

* Customer tenure
* Contract type
* Customer support interactions
* Monthly charges
* Service subscriptions

---

## Conclusion

Logistic Regression provided a baseline classification model for customer churn prediction.

While the model successfully learned the dataset structure, the current feature set and class imbalance limited its ability to identify churning customers.

This experiment demonstrates an important machine learning lesson:

> A good algorithm cannot compensate for weak features or heavily imbalanced data.


