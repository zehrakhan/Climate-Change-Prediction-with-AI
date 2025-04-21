import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
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

# Convert data into required format for LSTM + Transformer (batch_size, timesteps, features)
X_train = X_train[:, np.newaxis, :]
X_val = X_val[:, np.newaxis, :]
X_test = X_test[:, np.newaxis, :]

# Build LSTM + Transformer Hybrid Model
def build_lstm_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # LSTM Layer
    lstm_out = layers.LSTM(128, return_sequences=True)(inputs)
    
    # Transformer block
    x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)
    x = layers.Dropout(0.1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Fully connected layers
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    
    return model

# Create the model
lstm_transformer_model = build_lstm_transformer_model(input_shape=(X_train.shape[1], X_train.shape[2]))

# Train the model
history = lstm_transformer_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Predict on test set
y_pred = lstm_transformer_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"LSTM + Transformer - MSE: {mse}, MAE: {mae}, R²: {r2}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.legend()
plt.title("LSTM + Transformer: Actual vs Predicted Temperature")
plt.show()

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()