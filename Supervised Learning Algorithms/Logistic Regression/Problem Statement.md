# Customer Churn Prediction Using Logistic Regression

## Problem Statement

Telecommunication companies lose millions of dollars every year when customers switch to competitors. Acquiring a new customer is significantly more expensive than retaining an existing one.

The company wants to identify customers who are likely to discontinue their services in the near future so that retention teams can take proactive actions such as providing discounts, loyalty benefits, or personalized offers.

Using historical customer information such as age, salary, usage patterns, and service activity, we can build a machine learning model capable of predicting whether a customer is likely to churn.

---

## Business Objective

Predict whether a customer will leave the telecom service.

---

## Input Features

* Age
* Number of Dependents
* Estimated Salary
* Calls Made
* SMS Sent
* Data Used

---

## Target Variable

### Churn

```text
0 = Customer Stays
1 = Customer Leaves
```

---

## Success Criteria

Develop a classification model capable of accurately identifying customers at risk of leaving.

---

## Why This Problem Matters

Customer retention is one of the most important challenges in the telecom industry.

Accurate churn prediction helps companies:

* Reduce customer loss
* Increase revenue
* Improve customer satisfaction
* Lower acquisition costs

---

## Why Logistic Regression?

The target variable contains only two classes:

```text
Stay
Leave
```

Since the objective is binary classification rather than numerical prediction, Logistic Regression is a suitable algorithm.
