import matplotlib.pyplot as plt

# Updated experiment data with new configuration and results
experiments = [
    {"exp": "Exp 1", "Batch": 100, "n1": 256, "n2": 256, "Dropout": 0.3, "d_state": 1024, "dconv": 4, "n_layers": 6, "MAE": 0.267, "MSE": 0.249, "R2": 0.723},
    {"exp": "Exp 2", "Batch": 64, "n1": 128, "n2": 256, "Dropout": 0.2, "d_state": 128, "dconv": 2, "n_layers": 3, "MAE": 0.260, "MSE": 0.254, "R2": 0.716},
    {"exp": "Exp 3", "Batch": 64, "n1": 128, "n2": 256, "Dropout": 0.2, "d_state": 128, "dconv": 2, "n_layers": 3, "MAE": 0.308, "MSE": 0.274, "R2": 0.694}
]

# Extract metrics for plotting
exp_names = [exp["exp"] for exp in experiments]
mae_values = [exp["MAE"] for exp in experiments]
mse_values = [exp["MSE"] for exp in experiments]
r2_values = [exp["R2"] for exp in experiments]

# Plotting all metrics in one graph
plt.figure(figsize=(10, 6))

# Plot MAE
plt.plot(exp_names, mae_values, marker='o', label="MAE", color='blue', linestyle='-', linewidth=2)

# Plot MSE
plt.plot(exp_names, mse_values, marker='s', label="MSE", color='red', linestyle='--', linewidth=2)

# Plot R²
plt.plot(exp_names, r2_values, marker='^', label="R²", color='green', linestyle='-.', linewidth=2)

# Labels and title
plt.xlabel('Experiments')
plt.ylabel('Metric Values')

plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
