"""
Nonconformity score implementations for conformal prediction.

Score A: Spread-normalized price error
Score B: Vega-weighted IV error
"""

import numpy as np
import torch
from typing import Dict, Optional


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

    Per exposé Equation (2), calibration scores evaluate Vega at the
    observed market IV. This is distinct from Equation (6), where a new
    prediction band's weight is evaluated at the GNO estimate.
    """

    def __init__(self, min_weight: float = 1.0):
        self.min_weight = min_weight

    def compute(self,
                iv_pred: np.ndarray,
                iv_market: np.ndarray,
                aux_data: Dict) -> np.ndarray:
        # Exposé Eq. (2): w_V(x; v_t), with v_t the observed market surface.
        vega = aux_data['vega_market']

        mean_vega = vega.mean()
        weights = vega / mean_vega
        weights = np.maximum(weights, self.min_weight)

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
    from volatility_smoothing.utils.black_scholes import vega
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
        r_tensor = data['r'].cpu()
        z_tensor = data['z'].cpu()
        iv_market_tensor = data['implied_volatility'].cpu()

        vega_market_tensor = vega(r_tensor, z_tensor, iv_market_tensor)
        vega_market = vega_market_tensor.numpy()

        aux_data = {
            'vega_market': vega_market
        }

        scorer = VegaNormalizedScore(**score_kwargs)

    else:
        raise ValueError(f"Unknown score_type: {score_type}. Must be 'spread' or 'vega'")

    scores = scorer.compute(iv_pred, iv_market, aux_data)
    return scores
