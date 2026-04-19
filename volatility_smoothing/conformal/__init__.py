"""
Conformal prediction for implied volatility surfaces.

Components:
- nonconformity_scores: Score computation (spread and Vega normalized)
- mondrian: Spatial binning for heterogeneous uncertainty
- rolling_window: Temporal calibration window
- prediction_bands: Uncertainty band construction
- no_arbitrage_projection: Arbitrage constraint enforcement
- conformal_predictor: Main prediction interface
"""

from .nonconformity_scores import (
    NonconformityScore,
    SpreadNormalizedScore,
    VegaNormalizedScore,
    compute_scores_from_data
)

from .mondrian import MondrianConformal, MondrianBin

from .rolling_window import RollingWindowCalibration, CalibrationSurface

from .prediction_bands import PredictionBands

from .no_arbitrage_projection import (
    ArbitrageConstraints,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    project_to_arbitrage_free
)

from .conformal_predictor import (
    ConformalConfig,
    ConformalVolatilityPredictor
)

__all__ = [
    # Nonconformity scores
    'NonconformityScore',
    'SpreadNormalizedScore',
    'VegaNormalizedScore',
    'compute_scores_from_data',

    # Mondrian binning
    'MondrianConformal',
    'MondrianBin',

    # Rolling window
    'RollingWindowCalibration',
    'CalibrationSurface',

    # Prediction bands
    'PredictionBands',

    # No-arbitrage projection
    'ArbitrageConstraints',
    'check_butterfly_arbitrage',
    'check_calendar_arbitrage',
    'project_to_arbitrage_free',

    # Main predictor
    'ConformalConfig',
    'ConformalVolatilityPredictor',
]

__version__ = "0.1.0"
