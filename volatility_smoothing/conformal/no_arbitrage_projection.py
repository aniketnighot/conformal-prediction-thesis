import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class ArbitrageConstraints:
    butterfly_violations: int = 0
    calendar_violations: int = 0
    max_butterfly_violation: float = 0.0
    max_calendar_violation: float = 0.0

    def has_violations(self) -> bool:
        return self.butterfly_violations > 0 or self.calendar_violations > 0

    def __repr__(self):
        return (f"ArbitrageConstraints(butterfly={self.butterfly_violations}, "
                f"calendar={self.calendar_violations})")


def check_butterfly_arbitrage(iv_surface: np.ndarray,
                               strikes: np.ndarray,
                               maturities: np.ndarray,
                               epsilon: float = 1e-6) -> ArbitrageConstraints:
    """
    Check convexity of total variance w = IV^2 * T in strike.
    A violation at (K1, K2, K3) means w(K2) exceeds the chord from K1 to K3,
    which permits a butterfly spread to be sold for risk-free profit.
    """
    n_maturities, n_strikes = iv_surface.shape
    if n_strikes < 3:
        return ArbitrageConstraints()

    violations = 0
    max_violation = 0.0

    for t_idx in range(n_maturities):
        T = maturities[t_idx]
        w = iv_surface[t_idx, :] ** 2 * T

        for i in range(n_strikes - 2):
            dK1 = strikes[i + 1] - strikes[i]
            dK2 = strikes[i + 2] - strikes[i + 1]
            w_interp = (dK2 * w[i] + dK1 * w[i + 2]) / (dK1 + dK2)
            violation = w[i + 1] - w_interp

            if violation > epsilon:
                violations += 1
                max_violation = max(max_violation, violation)

    return ArbitrageConstraints(
        butterfly_violations=violations,
        max_butterfly_violation=max_violation
    )


def check_calendar_arbitrage(iv_surface: np.ndarray,
                              strikes: np.ndarray,
                              maturities: np.ndarray,
                              epsilon: float = 1e-6) -> ArbitrageConstraints:
    """
    Check that total variance w = IV^2 * T is non-decreasing in maturity.
    A decrease from T1 to T2 at the same strike means the longer-dated option
    is worth less than the shorter-dated one — a calendar spread arbitrage.
    """
    n_maturities, n_strikes = iv_surface.shape
    if n_maturities < 2:
        return ArbitrageConstraints()

    violations = 0
    max_violation = 0.0

    for k_idx in range(n_strikes):
        w = iv_surface[:, k_idx] ** 2 * maturities

        for t_idx in range(n_maturities - 1):
            violation = w[t_idx] - w[t_idx + 1]
            if violation > epsilon:
                violations += 1
                max_violation = max(max_violation, violation)

    return ArbitrageConstraints(
        calendar_violations=violations,
        max_calendar_violation=max_violation
    )


def project_to_arbitrage_free(iv_surface: np.ndarray,
                               strikes: np.ndarray,
                               maturities: np.ndarray,
                               max_iterations: int = 100,
                               tolerance: float = 1e-6,
                               step_size: float = 0.5) -> Tuple[np.ndarray, dict]:
    """
    Project a volatility surface onto the nearest arbitrage-free surface.

    Alternates between enforcing butterfly convexity (row by row) and calendar
    monotonicity (column by column) until the surface stops changing.
    """
    initial_butterfly = check_butterfly_arbitrage(iv_surface, strikes, maturities, tolerance)
    initial_calendar = check_calendar_arbitrage(iv_surface, strikes, maturities, tolerance)

    if not (initial_butterfly.has_violations() or initial_calendar.has_violations()):
        return iv_surface, {
            "converged": True,
            "iterations": 0,
            "initial_violations": ArbitrageConstraints(),
            "final_violations": ArbitrageConstraints(),
            "projection_distance": 0.0
        }

    n_maturities, n_strikes = iv_surface.shape
    iv_current = iv_surface.copy()

    for iteration in range(max_iterations):
        iv_before = iv_current.copy()

        # Enforce butterfly: convexity of w in strike, one maturity at a time
        for t_idx in range(n_maturities):
            T = maturities[t_idx]
            w = iv_current[t_idx, :] ** 2 * T

            for i in range(1, n_strikes - 1):
                dK1 = strikes[i] - strikes[i - 1]
                dK2 = strikes[i + 1] - strikes[i]
                w_expected = (dK2 * w[i - 1] + dK1 * w[i + 1]) / (dK1 + dK2)

                if w[i] > w_expected + tolerance:
                    w[i] += step_size * (w_expected - w[i])

            iv_current[t_idx, :] = np.sqrt(np.maximum(w / T, 1e-8))

        # Enforce calendar: w non-decreasing in maturity, one strike at a time
        for k_idx in range(n_strikes):
            w = iv_current[:, k_idx] ** 2 * maturities

            for t_idx in range(n_maturities - 1):
                if w[t_idx] > w[t_idx + 1] + tolerance:
                    w_avg = (w[t_idx] + w[t_idx + 1]) / 2
                    w[t_idx] = w_avg
                    w[t_idx + 1] = w_avg

            iv_current[:, k_idx] = np.sqrt(np.maximum(w / maturities, 1e-8))

        if np.linalg.norm(iv_current - iv_before) < tolerance:
            break

    final_butterfly = check_butterfly_arbitrage(iv_current, strikes, maturities, tolerance)
    final_calendar = check_calendar_arbitrage(iv_current, strikes, maturities, tolerance)

    return iv_current, {
        "converged": not (final_butterfly.has_violations() or final_calendar.has_violations()),
        "iterations": iteration + 1,
        "initial_violations": ArbitrageConstraints(
            butterfly_violations=initial_butterfly.butterfly_violations + initial_calendar.calendar_violations,
            max_butterfly_violation=max(initial_butterfly.max_butterfly_violation,
                                        initial_calendar.max_calendar_violation)
        ),
        "final_violations": ArbitrageConstraints(
            butterfly_violations=final_butterfly.butterfly_violations + final_calendar.calendar_violations,
            max_butterfly_violation=max(final_butterfly.max_butterfly_violation,
                                        final_calendar.max_calendar_violation)
        ),
        "projection_distance": np.linalg.norm(iv_current - iv_surface)
    }
