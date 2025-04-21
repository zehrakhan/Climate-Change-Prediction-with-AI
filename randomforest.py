import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/Users/mehjabeen/Desktop/KION/Thesis/Model_Thesis/imputed_deu_data1.csv")

# Filter for Aachen
df_aachen = df[df["City"] == "Aachen"].copy()

# Clean Latitude and Longitude columns
df_aachen["Latitude"] = df_aachen["Latitude"].str.extract(r'([-+]?\d*\.\d+|\d+)').astype(float)
df_aachen["Longitude"] = df_aachen["Longitude"].str.extract(r'([-+]?\d*\.\d+|\d+)').astype(float)

# Create a new column for 'YearsSince1744'
df_aachen["YearsSince1744"] = df_aachen["Year"] - 1744

# Selecting features and target
features = ["YearsSince1744", "AverageTemperatureUncertainty", "Latitude", "Longitude",
            "TemperatureAnomaly", "TemperatureChange", "Yearly_Avg_Temperature"]
target = "AverageTemperature"

# Define X (features) and y (target)
X = df_aachen[features]
y = df_aachen[target]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training (70%), validation (15%), and testing (15%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.17, random_state=42)

# Initialize and train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest - MSE: {mse}, MAE: {mae}, R²: {r2}")

# Feature Importance
feature_importances = pd.Series(rf_model.feature_importances_, index=features)
feature_importances.sort_values().plot(kind="barh", title="Feature Importance in Random Forest")
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.legend()
plt.title("Random Forest: Actual vs Predicted Temperature")
plt.show()
