import numpy as np

def compute_quantile(data, q):
    return np.quantile(data, q)

def compute_skewness(data):
    mean = np.mean(data)
    std = np.std(data)
    return np.mean(((data - mean) / std) ** 3)

def compute_kurtosis(data):
    mean = np.mean(data)
    std = np.std(data)
    return np.mean(((data - mean) / std) ** 4) - 3

def bootstrap_confidence_interval(data, func=np.mean, num_samples=1000, alpha=0.05):
    bootstraps = np.random.choice(data, (num_samples, len(data)), replace=True)
    stats = np.apply_along_axis(func, 1, bootstraps)
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower, upper
