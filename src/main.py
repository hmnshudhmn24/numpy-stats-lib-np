import numpy as np
from numpy_stats import compute_quantile, compute_skewness, compute_kurtosis, bootstrap_confidence_interval

np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=1000)

print("Quantile (0.5):", compute_quantile(data, 0.5))
print("Skewness:", compute_skewness(data))
print("Kurtosis:", compute_kurtosis(data))
ci_low, ci_high = bootstrap_confidence_interval(data)
print(f"95% Confidence Interval for Mean: [{ci_low:.2f}, {ci_high:.2f}]")
