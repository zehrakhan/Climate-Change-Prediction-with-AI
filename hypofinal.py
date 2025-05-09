import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Given LSTM predicted temperature values
predicted_temps = np.array([5.6485553, 9.693489, 13.8637085, 16.861849, 18.23563, 17.63588, 14.546166, 
                            9.601876, 5.334866, 2.1104887, 0.9822695, 1.0915952, 3.719997, 7.900401, 
                            12.447004, 15.743015, 17.34591, 16.804293, 13.619662, 8.4211035, 4.2748685, 
                            1.0408223, -0.06447846, 0.80758864, 4.047362, 8.776802, 13.200128, 16.29433, 
                            17.686687, 17.111433, 14.105318, 9.3446245, 5.395141, 2.6286607, 0.8204077, 
                            3.0992236, 5.4982386, 9.328004, 13.46404, 16.566359, 17.89852, 17.07197, 
                            13.351068, 8.102102, 3.9568946, 1.0454911, 0.6551963, 1.6421423, 4.4422183, 
                            8.906767, 13.190577, 16.360956, 17.785141, 17.109507, 13.97062, 9.030291, 
                            5.0715857, 2.1311648, 1.3602549, 2.9667504, 5.579561, 9.453882, 13.578567, 
                            16.686268, 18.13133, 17.46236, 14.235976, 9.220183, 5.183916, 2.1404388, 
                            1.2859223, 3.1682603, 5.578236, 9.587343, 13.64705, 16.77741, 18.306953, 
                            17.888523, 14.778997, 9.627788, 5.5179157, 2.4275692, 1.3281618, 2.4455502, 
                            5.025167, 9.128743, 13.37742, 16.494862, 17.972136, 17.437746, 14.280542, 
                            9.182638, 5.0970864, 2.046117, 1.1034032, 2.381082, 4.6456137, 8.83386, 
                            13.134291, 16.38444, 17.865316, 17.27713, 13.835349, 8.611878, 4.504883, 
                            1.4671092, 0.44104195, 2.0719707, 4.7731304, 8.933218, 13.15789, 16.411777, 
                            17.993576, 17.445051, 14.447151, 9.390767, 5.2641478, 2.2508435, 1.0904559, 
                            2.7625086, 5.1992774, 8.960957, 13.212959, 16.549215, 18.17846, 17.839922, 
                            14.918427, 9.748212, 5.665904, 2.689378, 1.7097878, 2.4140968, 5.1552567, 
                            9.238159, 13.313236, 16.364578, 17.74045, 17.115967, 13.655183, 8.486237, 
                            4.4823704, 1.6301258, 1.035097, 2.667167, 5.208766, 9.284983, 13.370792, 
                            16.541634, 18.090696, 17.550972, 14.377641, 9.081935, 4.857692, 1.8236831, 
                            1.1037618, 3.2265475, 5.5530295, 9.569123, 13.698986, 16.76373, 18.214844, 
                            17.713097, 14.793654, 9.674252, 5.5349183, 2.692198, 1.8017311, 3.4183424, 
                            5.656365, 9.534256, 13.61445, 16.736383, 18.277544, 17.858019, 14.824265, 
                            9.399429, 5.050974, 1.9245265, 1.0876306, 3.0005906, 5.394488, 9.319256, 
                            13.501489, 16.768427, 18.424732, 17.973228, 14.611146, 9.320943, 5.318397, 
                            2.4471197, 1.5931847, 3.4304333, 5.838497, 9.924119, 13.866544, 16.842695, 
                            18.243374, 17.70986, 14.652304, 9.591187, 5.557754, 2.5297537, 1.3779293, 
                            1.9790255, 4.516811, 8.7231045, 12.931574, 16.174068, 17.732584, 17.21085, 
                            14.044066, 9.017258, 4.9791884, 2.00571, 1.1245062, 1.7916092, 4.836957, 
                            9.08666, 13.294936, 16.479828, 18.003843, 17.508657, 14.562952, 9.722181, 
                            5.7212744, 2.76294, 1.7048578, 2.1005714, 4.961141, 8.879893, 12.990442, 
                            16.146421, 17.64064, 17.077322, 13.977041, 8.834596, 4.59185, 1.4976735, 
                            0.6271561, 2.3126686, 4.9176464, 9.290187, 13.472251, 16.606142, 18.045918, 
                            17.498049, 14.408676, 9.360221, 5.3041134, 2.4484882, 1.5636922, 3.501332, 
                            5.9212923, 9.980434, 14.0309305, 17.092428, 18.56851, 18.096083, 15.031314, 
                            9.995703, 5.7779183, 2.7732148, 1.5681881, 3.2329834, 5.457422, 9.487175, 
                            13.690582, 16.875805, 18.543926, 18.349392, 15.510229, 10.295961, 6.1034565, 
                            3.1494396, 2.007762, 2.8278003, 5.2046194, 9.180492, 13.335763, 16.4346, 
                            17.92974, 17.509703, 14.598654, 9.393094, 5.238241, 2.1400805, 0.9961486, 
                            1.0455948, 4.249683, 8.517261, 12.907064, 16.246552, 17.81104, 17.490463, 
                            14.639038, 9.760129, 5.697623, 2.6556273, 1.8257868, 2.7255895, 4.920624, 
                            9.289299, 13.447236, 16.654299, 18.184092, 17.67263, 14.640478, 9.647594, 
                            5.8281827, 3.024114, 1.9221015, 1.9025235])

# Generate monthly dates starting from Jan 2013
dates = pd.date_range(start="1/1/2013", periods=len(predicted_temps), freq='MS')

# Create DataFrame
df = pd.DataFrame({"Date": dates, "PredictedTemperature": predicted_temps})

# Extract year and compute yearly average temperature
df["Year"] = df["Date"].dt.year
yearly_avg = df.groupby("Year")["PredictedTemperature"].mean()

# Compute trend line using linear regression
years = yearly_avg.index.astype(float)  # Convert year to float
slope, intercept, r_value, p_value, std_err = linregress(years, yearly_avg.values)
trend_line = intercept + slope * years  # Compute trend values

# Display the results (p-value, slope, intercept, r-value, std error)
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-value: {r_value}")
print(f"P-value: {p_value}")
print(f"Standard error: {std_err}")

# Plot yearly average temperatures
plt.figure(figsize=(10,5))
plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linestyle='-', color='b', label="Yearly Avg Temperature")
plt.plot(yearly_avg.index, trend_line, linestyle='--', color='green', label="Trend Line (Linear Fit)")
plt.xlabel("Year")
plt.ylabel("Average Temperature (°C)")
plt.title("Yearly Average Predicted Temperature with Trend")
plt.legend()
plt.grid(True)
plt.show()
