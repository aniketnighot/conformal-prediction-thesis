"""
Prediction band construction for conformal prediction.

Supports price-spread mode and Vega-IV mode for uncertainty bands.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Tuple, Optional


def compute_d1_d2(z: np.ndarray,
                  iv: np.ndarray,
                  sqrt_tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Black-Scholes d1 and d2 parameters.

    Parameters:
    -----------
    z : np.ndarray
        Normalized log-moneyness
    iv : np.ndarray
        Implied volatility
    sqrt_tau : np.ndarray
        Square root of time to maturity (ρ)

    Returns:
    --------
    d1, d2 : Tuple[np.ndarray, np.ndarray]
    """
    a = -z / iv
    total_vol = iv * sqrt_tau
    d1 = a + total_vol / 2
    d2 = a - total_vol / 2
    return d1, d2


def black_scholes_price(z: np.ndarray,
                        iv: np.ndarray,
                        sqrt_tau: np.ndarray,
                        discount: np.ndarray,
                        forward: np.ndarray) -> np.ndarray:
    """
    Compute Black-Scholes option price (calls and puts), scaled by the
    discount factor and forward price: discount * forward * (N(d1) -
    e^k N(d2)) for calls (and the corresponding put formula). This is a
    dollar/currency-denominated price, not the forward-normalized,
    dimensionless BS_pm(x,v) = N(d1) - e^k N(d2) used in the exposé and the
    base GNO paper (Sec. 4.2, footnote 7) — that quantity is implemented
    separately as volatility_smoothing.utils.black_scholes.normalized_option_price.
    Callers that need forward-normalized prices should not use this function
    directly.

    Parameters:
    -----------
    z : np.ndarray
        Normalized log-moneyness
    iv : np.ndarray
        Implied volatility
    sqrt_tau : np.ndarray
        Square root of time to maturity (ρ)
    discount : np.ndarray
        Discount factor
    forward : np.ndarray
        Forward price

    Returns:
    --------
    price : np.ndarray
        Dollar-denominated option prices
    """
    d1, d2 = compute_d1_d2(z, iv, sqrt_tau)

    k = sqrt_tau * z

    is_call = (k >= 0).astype(float)

    call_price = discount * forward * (
        norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
    )

    put_price = discount * forward * (
        np.exp(k) * norm.cdf(-d2) - norm.cdf(-d1)
    )

    price = is_call * call_price + (1 - is_call) * put_price

    return price


def vega_bs(z: np.ndarray,
            iv: np.ndarray,
            sqrt_tau: np.ndarray,
            discount: np.ndarray,
            forward: np.ndarray) -> np.ndarray:
    """
    Compute Black-Scholes Vega (∂price/∂IV).

    Parameters:
    -----------
    z : np.ndarray
        Normalized log-moneyness
    iv : np.ndarray
        Implied volatility
    sqrt_tau : np.ndarray
        Square root of time to maturity (ρ)
    discount : np.ndarray
        Discount factor
    forward : np.ndarray
        Forward price

    Returns:
    --------
    vega : np.ndarray
        Vega values
    """
    d1, _ = compute_d1_d2(z, iv, sqrt_tau)
    vega = discount * forward * sqrt_tau * norm.pdf(d1)
    return vega


def implied_volatility_from_price(price: np.ndarray,
                                   z: np.ndarray,
                                   sqrt_tau: np.ndarray,
                                   discount: np.ndarray,
                                   forward: np.ndarray,
                                   iv_min: float = 1e-4,
                                   iv_max: float = 5.0,
                                   return_diagnostics: bool = False):
    """
    Invert black_scholes_price for implied volatility (Eq. 5: v = IV(x, P)).

    Root-finds per point with Brent's method rather than a closed-form
    solver, since black_scholes_price is already the pricing function used
    elsewhere for band construction and this avoids introducing a separate
    IV-inversion dependency (e.g. py_vollib) with its own (S, K, t)
    conventions.

    Parameters:
    -----------
    price : np.ndarray
        Target option price to invert (e.g. price_lower or price_upper)
    z, sqrt_tau, discount, forward : np.ndarray
        Same normalized-coordinate inputs as black_scholes_price

    Returns:
    --------
    iv : np.ndarray
        Implied volatility solving black_scholes_price(z, iv, ...) == price.
        Points where no sign change is found within [iv_min, iv_max]
        (price outside the model's attainable range) are clipped to the
        nearest bound rather than raising.
    diagnostics : dict, optional
        Returned with ``iv`` when ``return_diagnostics=True``. It records
        successful inversions, clips to either volatility bound, solver
        failures, and negative or non-finite price endpoints.
    """
    price = np.atleast_1d(np.asarray(price, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    sqrt_tau = np.atleast_1d(np.asarray(sqrt_tau, dtype=float))
    discount = np.atleast_1d(np.asarray(discount, dtype=float))
    forward = np.atleast_1d(np.asarray(forward, dtype=float))

    n = price.shape[0]
    iv = np.full(n, np.nan)
    status = np.full(n, "success", dtype=object)
    negative_price = price < 0
    nonfinite_price = ~np.isfinite(price)

    for i in range(n):
        z_i, sqrt_tau_i, discount_i, forward_i, price_i = (
            z[i], sqrt_tau[i], discount[i], forward[i], price[i]
        )

        if not np.all(np.isfinite(
            [z_i, sqrt_tau_i, discount_i, forward_i, price_i]
        )):
            iv[i] = iv_min
            status[i] = "solver_failure"
            continue

        def f(vol, z_i=z_i, sqrt_tau_i=sqrt_tau_i, discount_i=discount_i,
              forward_i=forward_i, price_i=price_i):
            return black_scholes_price(
                np.array([z_i]), np.array([vol]), np.array([sqrt_tau_i]),
                np.array([discount_i]), np.array([forward_i])
            )[0] - price_i

        f_lo, f_hi = f(iv_min), f(iv_max)
        if f_lo > 0:
            iv[i] = iv_min
            status[i] = "clipped_min"
        elif f_hi < 0:
            iv[i] = iv_max
            status[i] = "clipped_max"
        else:
            try:
                iv[i] = brentq(f, iv_min, iv_max, xtol=1e-8, maxiter=100)
            except (ValueError, RuntimeError, OverflowError, FloatingPointError):
                # Retain a finite band endpoint even if the numerical solver
                # fails, and report the event separately.
                iv[i] = iv_min if abs(f_lo) <= abs(f_hi) else iv_max
                status[i] = "solver_failure"

    if not return_diagnostics:
        return iv

    n_classified = sum(
        int(np.sum(status == label))
        for label in ("success", "clipped_min", "clipped_max", "solver_failure")
    )
    if n_classified != n:
        raise AssertionError("each IV inversion must have exactly one numerical status")

    diagnostics = {
        "n_total": int(n),
        "successful_inversions": int(np.sum(status == "success")),
        "clipped_min": int(np.sum(status == "clipped_min")),
        "clipped_max": int(np.sum(status == "clipped_max")),
        "solver_failures": int(np.sum(status == "solver_failure")),
        "negative_price_endpoints": int(np.sum(negative_price)),
        "nonfinite_price_endpoints": int(np.sum(nonfinite_price)),
        "iv_min": float(iv_min),
        "iv_max": float(iv_max),
    }
    return iv, diagnostics


class PredictionBands:
    """
    Unified interface for both prediction band modes.

    Provides methods to compute bands in either price or IV space,
    with optional conversion between the two.
    """

    def __init__(self,
                 mode: str = "vega_iv",
                 min_spread: float = 0.10,
                 min_vega: float = 1.0):
        self.mode = mode
        self.min_spread = min_spread
        self.min_vega = min_vega

    def compute(self,
                z: np.ndarray,
                iv_pred: np.ndarray,
                sqrt_tau: np.ndarray,
                discount: np.ndarray,
                forward: np.ndarray,
                quantile: np.ndarray,
                bid_price: Optional[np.ndarray] = None,
                ask_price: Optional[np.ndarray] = None) -> dict:
        """
        Compute prediction bands in both price and IV space.

        Parameters:
        -----------
        z : np.ndarray
            Normalized log-moneyness
        iv_pred : np.ndarray
            Predicted implied volatility
        sqrt_tau : np.ndarray
            Square root of time to maturity
        discount : np.ndarray
            Discount factor
        forward : np.ndarray
            Forward price
        quantile : np.ndarray
            Conformal quantiles (per-point)
        bid_price : np.ndarray, optional
            Bid prices (required for price_spread mode)
        ask_price : np.ndarray, optional
            Ask prices (required for price_spread mode)

        Returns:
        --------
        bands : dict
            Dictionary with iv_lower, iv_upper, price_lower, price_upper
        """
        if self.mode == "vega_iv":
            vega = vega_bs(z, iv_pred, sqrt_tau, discount, forward)

            # Normalize by mean vega (Equation 2 from exposé)
            # w_V(x; v) := V(x,v) / mean(V) ∨ 1
            mean_vega = vega.mean()
            weights = vega / mean_vega
            weights = np.maximum(weights, self.min_vega)

            # Equation 6: delta_IV = q / w_V
            delta_iv = quantile / weights

            iv_lower = iv_pred - delta_iv
            iv_upper = iv_pred + delta_iv

            iv_lower = np.maximum(iv_lower, 1e-4)
            iv_upper = np.maximum(iv_upper, 1e-4)

            price_lower = black_scholes_price(z, iv_lower, sqrt_tau, discount, forward)
            price_upper = black_scholes_price(z, iv_upper, sqrt_tau, discount, forward)

        else:
            if bid_price is None or ask_price is None:
                raise ValueError("price_spread mode requires bid_price and ask_price")

            price_pred = black_scholes_price(z, iv_pred, sqrt_tau, discount, forward)

            spread = ask_price - bid_price
            spread_clamped = np.maximum(spread, self.min_spread)

            delta_price = (quantile / 2.0) * spread_clamped

            price_lower = price_pred - delta_price
            price_upper = price_pred + delta_price

            # Eq. 5: v_lo(x) = IV(x, P_lo(x)), v_hi(x) = IV(x, P_hi(x))
            iv_lower, lower_diagnostics = implied_volatility_from_price(
                price_lower, z, sqrt_tau, discount, forward,
                return_diagnostics=True)
            iv_upper, upper_diagnostics = implied_volatility_from_price(
                price_upper, z, sqrt_tau, discount, forward,
                return_diagnostics=True)

            numerical_diagnostics = {
                "lower": lower_diagnostics,
                "upper": upper_diagnostics,
                "spreads_floored": int(np.sum(spread < self.min_spread)),
                "min_spread": float(self.min_spread),
            }

        result = {
            "iv_lower": iv_lower,
            "iv_upper": iv_upper,
            "price_lower": price_lower,
            "price_upper": price_upper,
            "mode": self.mode
        }
        if self.mode == "price_spread":
            result["numerical_diagnostics"] = numerical_diagnostics
        return result
