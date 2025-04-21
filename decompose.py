import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
df = pd.read_csv("/Users/mehjabeen/Desktop/KION/Thesis/Model_Thesis/imputed_deu_data1.csv")

# Filter for Aachen (since you're focusing on one city)
df_aachen = df[df["City"] == "Aachen"].copy()

# Convert Year and Month to Date format
df_aachen["Date"] = pd.to_datetime(df_aachen["Year"].astype(str) + "-" + df_aachen["Month"].astype(str))

# Set Date as index
df_aachen.set_index("Date", inplace=True)

# Sort by Date (important for time series)
df_aachen.sort_index(inplace=True)

# Perform decomposition
decomposition = seasonal_decompose(df_aachen["AverageTemperature"], model="additive", period=12)

# Plot decomposition results
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(decomposition.trend, label="Trend", color="blue")
plt.legend(loc="upper left")
plt.ylabel("Trend")


plt.subplot(2, 1, 2)
plt.plot(decomposition.resid, label="Residual", color="red")
plt.legend(loc="upper left")
plt.ylabel("Residuals")

plt.suptitle("Time Series Decomposition for Aachen", fontsize=14)
plt.tight_layout()
plt.show()

