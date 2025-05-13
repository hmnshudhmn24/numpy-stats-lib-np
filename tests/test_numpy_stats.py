import numpy as np
from src.numpy_stats import compute_quantile, compute_skewness, compute_kurtosis, bootstrap_confidence_interval

def test_all():
    data = np.array([1, 2, 3, 4, 5])
    assert np.isclose(compute_quantile(data, 0.5), 3)
    assert np.isclose(round(compute_skewness(data), 5), 0)
    assert np.isclose(round(compute_kurtosis(data), 5), -1.3)
    ci = bootstrap_confidence_interval(data)
    assert len(ci) == 2
