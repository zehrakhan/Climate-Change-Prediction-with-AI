import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
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

# Prepare time series data
time_series = df_aachen.set_index("YearsSince1744")["AverageTemperature"]

# Split into train and test (85% train, 15% test)
train_size = int(len(time_series) * 0.85)
train, test = time_series[:train_size], time_series[train_size:]

# Fit ARIMA model (manually set order)
arima_model = ARIMA(train, order=(5, 1, 0))  # Example order (p, d, q)
arima_fit = arima_model.fit()

# Forecast
arima_forecast = arima_fit.forecast(steps=len(test))

# Evaluate ARIMA
mse = mean_squared_error(test, arima_forecast)
mae = mean_absolute_error(test, arima_forecast)
r2 = r2_score(test, arima_forecast)

print(f"ARIMA - MSE: {mse}, MAE: {mae}, R²: {r2}")

# Plot ARIMA results
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, arima_forecast, label="ARIMA Forecast")
plt.legend()
plt.title("ARIMA Forecast vs Actual")
plt.xlabel("Years Since 1744")
plt.ylabel("Average Temperature")
plt.show()
