# Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for **classification problems**.

Despite its name, Logistic Regression is not used to predict continuous values. Instead, it predicts the probability that an observation belongs to a particular class.

The algorithm uses the **Sigmoid Function** to convert numerical outputs into probabilities ranging between 0 and 1.

These probabilities are then used to classify data into categories such as:

* Yes / No
* True / False
* Spam / Not Spam
* Survived / Did Not Survive

---

## Formula

### Linear Equation

```text
z = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

Where:

* z = Linear combination of input features
* b₀ = Intercept
* b₁...bₙ = Feature coefficients
* x₁...xₙ = Input features

### Sigmoid Function

```text
P(y=1) = 1 / (1 + e^(-z))
```

The sigmoid function transforms any value into a probability between 0 and 1.

---

## Learning Type

Supervised Learning

---

## Problem Type

Classification

---

## Classification Categories

### Binary Classification

Predicts one of two possible outcomes.

Examples:

* Spam or Not Spam
* Pass or Fail
* Fraudulent or Legitimate
* Survived or Not Survived

### Multi-Class Classification

Can be extended to classify multiple categories.

Examples:

* Animal Classification
* Handwritten Digit Recognition
* Product Category Prediction

---

## Output

Instead of predicting a numerical value, Logistic Regression predicts a probability.

Example:

```text
Probability = 0.92
```

Since the probability is greater than 0.5:

```text
Prediction = Positive Class
```

---

## Key Idea

Linear Regression predicts continuous values:

```text
House Price = ₹50,00,000
```

Logistic Regression predicts probabilities:

```text
Probability of Survival = 92%
```

This makes Logistic Regression suitable for classification tasks rather than regression tasks.

---

## Historical Background

Logistic Regression originated from statistical research in the 19th century and became one of the most widely used classification algorithms in machine learning.

Today, it serves as a foundational algorithm for binary classification problems and is often used as a baseline model before applying more complex techniques.

---

## Fun Fact

Many email spam filters, medical diagnosis systems, and customer churn prediction models begin with Logistic Regression because it is simple, interpretable, and surprisingly effective.
