"""
conformal.py
Shared helper for the prediction intervals (scripts 10 and 11).
"""
import numpy as np


def conformal_margin(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected quantile of the calibration scores."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))
