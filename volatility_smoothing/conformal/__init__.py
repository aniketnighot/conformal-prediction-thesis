"""
Conformal prediction for implied volatility surfaces.

Components:
- nonconformity_scores: Score computation (spread and Vega normalized)
- rolling_window: Temporal calibration window
- prediction_bands: Uncertainty band construction
"""

from .nonconformity_scores import (
    NonconformityScore,
    SpreadNormalizedScore,
    VegaNormalizedScore,
    compute_scores_from_data
)


from .rolling_window import RollingWindowCalibration, CalibrationSurface



__all__ = [
    # Nonconformity scores
    'NonconformityScore',
    'SpreadNormalizedScore',
    'VegaNormalizedScore',
    'compute_scores_from_data',

    

    # Rolling window
    'RollingWindowCalibration',
    'CalibrationSurface',

    # Prediction bands
    'PredictionBands',

]

__version__ = "0.1.0"
