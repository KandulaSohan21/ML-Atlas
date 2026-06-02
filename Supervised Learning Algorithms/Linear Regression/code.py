import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n")
print(df.head())

# ==========================
# Features and Target
# ==========================

X = df[
    [
        'Engine size (L)',
        'Cylinders',
        'City (L/100 km)',
        'Highway (L/100 km)',
        'Combined (L/100 km)'
    ]
]

y = df['CO2 emissions (g/km)']

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Model Training
# ==========================

model = LinearRegression()

model.fit(X_train, y_train)

# ==========================
# Predictions
# ==========================

y_pred = model.predict(X_test)

# ==========================
# Evaluation
# ==========================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-------------------")
print("MAE :", round(mae,2))
print("MSE :", round(mse,2))
print("RMSE:", round(rmse,2))
print("R²  :", round(r2,4))

# ==========================
# Graph 1
# Engine Size vs CO2
# ==========================

plt.figure(figsize=(8,5))
plt.scatter(
    df['Engine size (L)'],
    df['CO2 emissions (g/km)']
)

plt.title("Engine Size vs CO₂ Emissions")
plt.xlabel("Engine Size (L)")
plt.ylabel("CO₂ Emissions (g/km)")
plt.grid(True)

plt.savefig("graph1_engine_vs_co2.png")
plt.show()

# ==========================
# Graph 2
# Fuel Consumption vs CO2
# ==========================

plt.figure(figsize=(8,5))
plt.scatter(
    df['Combined (L/100 km)'],
    df['CO2 emissions (g/km)']
)

plt.title("Fuel Consumption vs CO₂ Emissions")
plt.xlabel("Fuel Consumption Combined")
plt.ylabel("CO₂ Emissions (g/km)")
plt.grid(True)

plt.savefig("graph2_fuel_vs_co2.png")
plt.show()

# ==========================
# Graph 3
# Actual vs Predicted
# ==========================

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Actual vs Predicted CO₂ Emissions")
plt.xlabel("Actual CO₂")
plt.ylabel("Predicted CO₂")
plt.grid(True)

plt.savefig("graph3_actual_vs_predicted.png")
plt.show()

# ==========================
# Sample Prediction
# ==========================

sample_vehicle = pd.DataFrame({
    'Engine size (L)':[2.0],
    'Cylinders':[4],
    'City (L/100 km)':[9.5],
    'Highway (L/100 km)':[7.1],
    'Combined (L/100 km)':[8.4]
})

prediction = model.predict(sample_vehicle)

print("\nPredicted CO₂ Emission:")
print(round(prediction[0],2),"g/km")
