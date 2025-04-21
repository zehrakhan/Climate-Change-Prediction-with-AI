import numpy as np
import math
from scipy.stats import norm

# Slope and Standard Errors
lstm_slope = 0.021370780803703693
xlstm_slope = 0.035944841269841224
lstm_se = 0.008895662669956431
xlstm_se = 0.018871659041137263

# Z-statistic
z_stat = (lstm_slope - xlstm_slope) / math.sqrt(lstm_se**2 + xlstm_se**2)

# p-value from normal distribution
p_value = 2 * (1 - norm.cdf(abs(z_stat)))

print("z-statistic:", z_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Reject the null hypothesis: The trends are significantly different.")
else:
    print("Fail to reject the null hypothesis: The trends are not significantly different.")
