"""
Nonconformity score implementations for conformal prediction.

Score A: Spread-normalized price error
Score B: Vega-weighted IV error
"""

import numpy as np
import torch
from typing import Dict, Optional


def prediction_vega_weights(rho: np.ndarray,
                            z: np.ndarray,
                            iv_pred: np.ndarray,
                            min_weight: float = 1.0) -> np.ndarray:
    """Forward-normalized Vega weights computed only from GNO predictions.

    The mean is taken over the target points of one surface. Using the same
    function for calibration scores and prediction bands makes the bands an
    inversion of the calibrated nonconformity score.
    """
    rho = np.asarray(rho, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    iv_pred = np.asarray(iv_pred, dtype=float).reshape(-1)
    if not (rho.size == z.size == iv_pred.size) or rho.size == 0:
        raise ValueError("rho, z and iv_pred must have the same nonzero length")
    if not (np.all(np.isfinite(rho)) and np.all(np.isfinite(z))
            and np.all(np.isfinite(iv_pred))):
        raise ValueError("Vega-weight inputs must be finite")
    if min_weight <= 0:
        raise ValueError("min_weight must be positive")

    safe_iv = np.maximum(iv_pred, 1e-8)
    d1 = -z / safe_iv + 0.5 * safe_iv * rho
    vega_pred = rho * np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)
    mean_vega_pred = max(float(vega_pred.mean()), 1e-12)
    return np.maximum(vega_pred / mean_vega_pred, min_weight)


class NonconformityScore:
    """Base class for nonconformity scores."""

    def compute(self,
                iv_pred: np.ndarray,
                iv_market: np.ndarray,
                aux_data: Dict) -> np.ndarray:
        """
        Compute nonconformity scores.

        Args:
            iv_pred: Predicted implied volatility (N,)
            iv_market: Market implied volatility (N,)
            aux_data: Dictionary containing auxiliary data (spreads, vega, etc.)

        Returns:
            scores: Nonconformity scores (N,)
        """
        raise NotImplementedError


class SpreadNormalizedScore(NonconformityScore):
    """
    Score A: Spread-normalized price error.

    Normalizes price errors by the bid-ask spread, making them
    comparable across different strike/maturity combinations.
    """

    def __init__(self, min_spread: float = 0.10):
        self.min_spread = min_spread

    def compute(self,
                iv_pred: np.ndarray,
                iv_market: np.ndarray,
                aux_data: Dict) -> np.ndarray:
        price_pred = aux_data['price_pred']
        price_market = aux_data['price_market']
        bid = aux_data['bid']
        ask = aux_data['ask']

        price_residual = price_pred - price_market
        spread = ask - bid
        spread = np.maximum(spread, self.min_spread)
        scores = 2 * np.abs(price_residual) / spread

        return scores

    def __repr__(self):
        return f"SpreadNormalizedScore(min_spread={self.min_spread})"


class VegaNormalizedScore(NonconformityScore):
    """
    Score B: Vega-weighted IV error.

    Vega is evaluated at the GNO prediction for both calibration scores and
    prediction bands. The observed market IV enters only the residual.
    """

    def __init__(self, min_weight: float = 1.0):
        self.min_weight = min_weight

    def compute(self,
                iv_pred: np.ndarray,
                iv_market: np.ndarray,
                aux_data: Dict) -> np.ndarray:
        iv_pred = np.asarray(iv_pred, dtype=float).reshape(-1)
        iv_market = np.asarray(iv_market, dtype=float).reshape(-1)
        weights = prediction_vega_weights(
            aux_data['rho'], aux_data['z'], iv_pred, self.min_weight)

        iv_residual = iv_pred - iv_market
        scores = weights * np.abs(iv_residual)

        return scores

    def __repr__(self):
        return f"VegaNormalizedScore(min_weight={self.min_weight})"


def compute_scores_from_data(
    score_type: str,
    data: Dict,
    iv_predict: torch.Tensor,
    model,
    device: torch.device,
    **score_kwargs
) -> np.ndarray:
    from scipy.stats import norm

    iv_pred = iv_predict.cpu().numpy()
    iv_market = data['implied_volatility'].cpu().numpy()

    if score_type == 'spread':
        r = data['r'].cpu().numpy()
        z = data['z'].cpu().numpy()
        tau = r ** 2
        discount = data['discount_factor'].cpu().numpy()
        forward = data['underlying_forward'].cpu().numpy()

        k = r * z
        strikes = forward * np.exp(k)
        is_call = (k >= 0).astype(float)
        sqrt_tau = r

        def compute_d1_d2(z_arr, iv_arr, sqrt_tau_arr):
            a = -z_arr / iv_arr
            total_vol = iv_arr * sqrt_tau_arr
            d1 = a + total_vol / 2
            d2 = a - total_vol / 2
            return d1, d2

        d1_market, d2_market = compute_d1_d2(z, iv_market, sqrt_tau)
        price_market_calls = discount * forward * (
            norm.cdf(d1_market) - np.exp(k) * norm.cdf(d2_market)
        )
        price_market_puts = discount * forward * (
            np.exp(k) * norm.cdf(-d2_market) - norm.cdf(-d1_market)
        )
        price_market = is_call * price_market_calls + (1 - is_call) * price_market_puts

        # Predicted prices
        d1_pred, d2_pred = compute_d1_d2(z, iv_pred, sqrt_tau)
        price_pred_calls = discount * forward * (
            norm.cdf(d1_pred) - np.exp(k) * norm.cdf(d2_pred)
        )
        price_pred_puts = discount * forward * (
            np.exp(k) * norm.cdf(-d2_pred) - norm.cdf(-d1_pred)
        )
        price_pred = is_call * price_pred_calls + (1 - is_call) * price_pred_puts

        # data['bid']/data['ask'] are already option prices (see
        # OptionsDataset.load_data docstring), not implied volatilities —
        # do not run them through compute_d1_d2.
        bid = data['bid'].cpu().numpy()
        ask = data['ask'].cpu().numpy()

        aux_data = {
            'price_pred': price_pred,
            'price_market': price_market,
            'bid': bid,
            'ask': ask
        }

        scorer = SpreadNormalizedScore(**score_kwargs)

    elif score_type == 'vega':
        aux_data = {
            'rho': data['r'].cpu().numpy(),
            'z': data['z'].cpu().numpy(),
        }

        scorer = VegaNormalizedScore(**score_kwargs)

    else:
        raise ValueError(f"Unknown score_type: {score_type}. Must be 'spread' or 'vega'")

    scores = scorer.compute(iv_pred, iv_market, aux_data)
    return scores
