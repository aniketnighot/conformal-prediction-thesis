"""Score B calibration must invert the same weight used for its IV bands."""

import unittest

import numpy as np
import torch

from volatility_smoothing.conformal.nonconformity_scores import (
    compute_scores_from_data,
    prediction_vega_weights,
)
from volatility_smoothing.conformal.prediction_bands import PredictionBands


class ScoreBConsistencyTest(unittest.TestCase):
    def test_calibration_and_bands_share_prediction_based_weights(self):
        rho = np.array([0.25, 0.4, 0.7])
        z = np.array([-0.12, 0.0, 0.18])
        iv_pred = np.array([0.22, 0.25, 0.29])
        market_a = np.array([0.21, 0.23, 0.31])
        market_b = np.array([0.19, 0.27, 0.26])
        weights = prediction_vega_weights(rho, z, iv_pred)

        def scores_for(market):
            data = {
                'r': torch.tensor(rho),
                'z': torch.tensor(z),
                'implied_volatility': torch.tensor(market),
            }
            return compute_scores_from_data(
                'vega', data, torch.tensor(iv_pred), None,
                torch.device('cpu'), min_weight=1.0,
            )

        np.testing.assert_allclose(
            scores_for(market_a), weights * np.abs(iv_pred - market_a)
        )
        np.testing.assert_allclose(
            scores_for(market_b), weights * np.abs(iv_pred - market_b)
        )

        quantile = np.array([0.015, 0.02, 0.025])
        bands = PredictionBands(mode='vega_iv').compute(
            z=z, iv_pred=iv_pred, sqrt_tau=rho,
            discount=np.array([0.99, 0.97, 0.94]),
            forward=np.array([3900.0, 4000.0, 4100.0]),
            quantile=quantile,
        )
        np.testing.assert_allclose(iv_pred - bands['iv_lower'], quantile / weights)
        np.testing.assert_allclose(bands['iv_upper'] - iv_pred, quantile / weights)


if __name__ == '__main__':
    unittest.main()
