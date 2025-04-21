import matplotlib.pyplot as plt

# Experiment data (you can add more experiments if needed)
experiments = [
    {"exp": "Exp 1", "MAE": 0.0439, "MSE": 0.0032, "R2": 0.9211},
    {"exp": "Exp 2", "MAE": 0.0432, "MSE": 0.0031, "R2": 0.9241},
    {"exp": "Exp 3", "MAE": 0.0445, "MSE": 0.0033, "R2": 0.9199},
    {"exp": "Exp 4", "MAE": 0.0743, "MSE": 0.0078, "R2": 0.8080},
    {"exp": "Exp 5", "MAE": 1.3971, "MSE": 3.1034, "R2": 0.9180},
    {"exp": "Exp 6", "MAE": 1.2881, "MSE": 2.824, "R2": 0.9250},
    {"exp": "Exp 7", "MAE": 1.4500, "MSE": 3.4000, "R2": 0.9100}
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
