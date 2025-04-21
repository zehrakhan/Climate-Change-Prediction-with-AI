import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import random
import tensorflow as tf

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Load the dataset
df = pd.read_csv("/Users/mehjabeen/Desktop/KION/Thesis/Model_Thesis/imputed_deu_data1.csv")

# Filter the dataset for Aachen
df_aachen = df[df["City"] == "Aachen"].copy()  # Make a copy to avoid setting on a view

# Clean Latitude and Longitude columns by removing N, S, E, W and converting to float
df_aachen['Latitude'] = df_aachen['Latitude'].replace(r'[^\d.-]', '', regex=True).astype(float)
df_aachen['Longitude'] = df_aachen['Longitude'].replace(r'[^\d.-]', '', regex=True).astype(float)

# Create a new column for 'YearsSince1744'
df_aachen.loc[:, "YearsSince1744"] = df_aachen["Year"] - 1744

# Selecting features
features = ['YearsSince1744', 'AverageTemperatureUncertainty', 'Latitude', 'Longitude', 
            'TemperatureAnomaly', 'TemperatureChange', 'Yearly_Avg_Temperature']
target = 'AverageTemperature'

# Normalize the feature set
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(df_aachen[features])
y_scaled = scaler_y.fit_transform(df_aachen[[target]])  # Scale the target separately

# Convert data into sequences (using last 10 years to predict the next year)
def create_sequences(X, y, time_steps=120):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i: i + time_steps])  # Collect past `time_steps` data
        y_seq.append(y[i + time_steps])  # Predict the next value
    return np.array(X_seq), np.array(y_seq)

# Prepare sequences
time_steps = 120
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# Split into training & validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Build the LSTM model
model = Sequential([
    LSTM(100, activation="relu", input_shape=(time_steps, X_seq.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation="relu"),
    Dropout(0.2),
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Plot loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# --- Future Predictions ---

# Function to make future predictions
def make_future_predictions(model, last_sequence, n_steps, num_features):
    future_predictions = []
    current_input = last_sequence
    for _ in range(n_steps):
        predicted = model.predict(current_input)
        future_predictions.append(predicted[0][0])
        
        # Reshape the predicted value to match the number of features
        predicted_reshaped = np.repeat(predicted[0], num_features).reshape(1, 1, num_features)
        
        # Append the predicted value to the input sequence (drop the first timestep)
        current_input = np.append(current_input[:, 1:, :], predicted_reshaped, axis=1)
        
    return future_predictions

# Get the last sequence from the training data
last_sequence = X_scaled[-time_steps:].reshape((1, time_steps, X_scaled.shape[1]))

# Make future predictions (next 10 years)
future_years = 10
future_predictions = make_future_predictions(model, last_sequence, future_years, num_features=7)

# Convert predictions back to the original scale
future_predictions = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate years for future predictions
future_years_range = np.arange(df_aachen['Year'].max() + 1, df_aachen['Year'].max() + future_years + 1)

# --- Trend Analysis using Linear Regression ---

# Reshape for regression
future_years_range_reshaped = future_years_range.reshape(-1, 1)
future_predictions_reshaped = future_predictions.reshape(-1, 1)

# Apply Linear Regression
reg = LinearRegression()
reg.fit(future_years_range_reshaped, future_predictions_reshaped)

# Compute trend slope
slope = reg.coef_[0][0]

# Generate trendline
trendline = reg.predict(future_years_range_reshaped)

# --- Visualization ---
plt.figure(figsize=(12, 6))

# Plot historical data (every 500th point)
plt.plot(df_aachen['Year'][::500], scaler_y.inverse_transform(y_scaled)[::500], label='Actual Temperature', color='blue')

# Plot future predictions
plt.plot(future_years_range, future_predictions, label='Future Predictions', color='red', linestyle='dashed')

# Plot trendline
plt.plot(future_years_range, trendline, label='Trendline', color='green', linestyle='dotted', linewidth=2)

# Labels and title
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.title('Climate Change - Future Temperature Trend')
plt.legend()
plt.show()

# --- Predicted vs Actual Values Plot (Validation) ---
plt.figure(figsize=(10, 6))
plt.plot(y_val_true[::500], label='Actual Temperature', color='blue', marker='o', linestyle='-')
plt.plot(y_val_pred_rescaled[::500], label='Predicted Temperature', color='red', marker='x', linestyle='--')
plt.xlabel('Time Steps')
plt.ylabel('Average Temperature (°C)')
plt.title('Predicted vs Actual Temperature on Validation Set')
plt.legend()
plt.grid(True)
plt.show()
