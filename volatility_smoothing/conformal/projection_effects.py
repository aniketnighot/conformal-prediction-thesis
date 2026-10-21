"""Quote-level diagnostics for interpolation and grid projection effects."""

from collections.abc import Mapping

import numpy as np


def summarize_band_stages(
    market_iv: np.ndarray,
    predicted_iv: np.ndarray,
    stages: Mapping[str, tuple[np.ndarray, np.ndarray]],
    inclusion_tolerance: float = 1e-10,
) -> dict:
    """Compare direct, interpolated-only, and projected bands at the same quotes.

    ``interpolated`` must be the grid round trip without optimization, using
    the same interpolation and fallback rules as ``projected``. All returned
    fractions are within-surface quantities; callers may average them across
    surfaces with equal weight.
    """
    names = ("direct", "interpolated", "projected")
    if set(stages) != set(names):
        raise ValueError(f"stages must contain exactly {names}")
    if not np.isfinite(inclusion_tolerance) or inclusion_tolerance < 0:
        raise ValueError("inclusion_tolerance must be finite and nonnegative")

    market = np.asarray(market_iv, dtype=float).reshape(-1)
    prediction = np.asarray(predicted_iv, dtype=float).reshape(-1)
    if market.size == 0 or prediction.size != market.size:
        raise ValueError("market and prediction must have the same nonzero length")

    vectors = [market, prediction]
    bounds = {}
    for name in names:
        lower, upper = stages[name]
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        if lower.size != market.size or upper.size != market.size:
            raise ValueError(f"{name} bounds must match the number of quotes")
        bounds[name] = (lower, upper)
        vectors.extend((lower, upper))
    if not all(np.all(np.isfinite(values)) for values in vectors):
        raise ValueError("band-stage inputs must be finite")

    result = {"n_quotes": int(market.size)}
    covered = {}
    contains_prediction = {}
    for name in names:
        lower, upper = bounds[name]
        covered[name] = (market >= lower) & (market <= upper)
        contains_prediction[name] = (
            (prediction >= lower - inclusion_tolerance)
            & (prediction <= upper + inclusion_tolerance)
        )
        result[f"{name}_coverage"] = float(covered[name].mean())
        result[f"{name}_covered_count"] = int(covered[name].sum())
        result[f"{name}_mean_width"] = float((upper - lower).mean())
        result[f"{name}_gno_inside_fraction"] = float(
            contains_prediction[name].mean()
        )
        result[f"{name}_gno_inside_count"] = int(
            contains_prediction[name].sum()
        )

    transitions = (
        ("interpolation", "direct", "interpolated"),
        ("optimization", "interpolated", "projected"),
    )
    for label, before, after in transitions:
        lower_before, upper_before = bounds[before]
        lower_after, upper_after = bounds[after]
        for endpoint, movement in (
            ("lower", lower_after - lower_before),
            ("upper", upper_after - upper_before),
        ):
            result[f"{label}_{endpoint}_signed_mean"] = float(movement.mean())
            result[f"{label}_{endpoint}_abs_mean"] = float(
                np.abs(movement).mean()
            )
            result[f"{label}_{endpoint}_abs_p95"] = float(
                np.percentile(np.abs(movement), 95)
            )
        result[f"{label}_mean_width_change"] = float(
            ((upper_after - lower_after) - (upper_before - lower_before)).mean()
        )
        result[f"{label}_coverage_lost_count"] = int(
            (covered[before] & ~covered[after]).sum()
        )
        result[f"{label}_coverage_gained_count"] = int(
            (~covered[before] & covered[after]).sum()
        )
        result[f"{label}_gno_lost_count"] = int(
            (contains_prediction[before] & ~contains_prediction[after]).sum()
        )
        result[f"{label}_gno_gained_count"] = int(
            (~contains_prediction[before] & contains_prediction[after]).sum()
        )

    return result
