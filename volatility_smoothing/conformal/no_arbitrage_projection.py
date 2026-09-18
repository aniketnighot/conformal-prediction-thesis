import numpy as np
from typing import Tuple
from dataclasses import dataclass
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack


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


def _fixed_k_interpolation_indices(z_axis: np.ndarray,
                                   rho_short: float,
                                   rho_long: float):
    """Grid indices for the paper's fixed-k mapping k=rho_long*z_long."""
    z_short = (rho_long / rho_short) * z_axis
    valid = (z_short >= z_axis[0]) & (z_short <= z_axis[-1])
    long_indices = np.flatnonzero(valid)
    z_short = z_short[valid]

    right = np.searchsorted(z_axis, z_short, side="right")
    right = np.clip(right, 1, len(z_axis) - 1)
    left = right - 1
    fraction = (z_short - z_axis[left]) / (z_axis[right] - z_axis[left])
    return long_indices, left, right, fraction


def check_calendar_arbitrage_fixed_k(iv_surface: np.ndarray,
                                      z_axis: np.ndarray,
                                      rho_axis: np.ndarray,
                                      epsilon: float = 1e-6) -> ArbitrageConstraints:
    """Check calendar monotonicity at fixed log-moneyness ``k = rho * z``.

    For a point ``(rho_long, z_long)``, the matching point on the preceding
    maturity row is ``(rho_short, rho_long*z_long/rho_short)``. Total variance
    is linearly interpolated on that row and must not exceed the total variance
    at the longer maturity. For positive implied volatility this is equivalent
    to the paper's non-decreasing total-volatility condition.
    """
    iv_surface = np.asarray(iv_surface, dtype=float)
    z_axis = np.asarray(z_axis, dtype=float)
    rho_axis = np.asarray(rho_axis, dtype=float)
    if iv_surface.shape != (rho_axis.size, z_axis.size):
        raise ValueError("iv_surface shape must match rho_axis and z_axis")
    if rho_axis.size < 2:
        return ArbitrageConstraints()
    if np.any(rho_axis <= 0) or np.any(np.diff(rho_axis) <= 0):
        raise ValueError("rho_axis must be positive and strictly increasing")
    if z_axis.size < 2 or np.any(np.diff(z_axis) <= 0):
        raise ValueError("z_axis must be strictly increasing")

    total_variance = iv_surface ** 2 * rho_axis[:, None] ** 2
    violations = 0
    max_violation = 0.0
    for i in range(rho_axis.size - 1):
        long_idx, left, right, fraction = _fixed_k_interpolation_indices(
            z_axis, rho_axis[i], rho_axis[i + 1]
        )
        short_variance = (
            (1.0 - fraction) * total_variance[i, left]
            + fraction * total_variance[i, right]
        )
        gap = short_variance - total_variance[i + 1, long_idx]
        is_violation = gap > epsilon
        violations += int(is_violation.sum())
        if is_violation.any():
            max_violation = max(max_violation, float(gap[is_violation].max()))

    return ArbitrageConstraints(
        calendar_violations=violations,
        max_calendar_violation=max_violation,
    )


def _fixed_k_shape_constraint_matrices(z_axis: np.ndarray,
                                       rho_axis: np.ndarray):
    """Sparse linear constraints for butterfly and fixed-k calendar shape."""
    n_rho, n_z = rho_axis.size, z_axis.size
    n_values = n_rho * n_z

    bf_rows, bf_cols, bf_values = [], [], []
    row = 0
    for i in range(n_rho):
        for j in range(1, n_z - 1):
            dz_left = z_axis[j] - z_axis[j - 1]
            dz_right = z_axis[j + 1] - z_axis[j]
            left_weight = dz_right / (dz_left + dz_right)
            right_weight = dz_left / (dz_left + dz_right)
            bf_rows.extend((row, row, row))
            bf_cols.extend((i * n_z + j - 1, i * n_z + j, i * n_z + j + 1))
            bf_values.extend((-left_weight, 1.0, -right_weight))
            row += 1
    butterfly = coo_matrix(
        (bf_values, (bf_rows, bf_cols)), shape=(row, n_values)
    ).tocsr()

    cal_rows, cal_cols, cal_values = [], [], []
    row = 0
    for i in range(n_rho - 1):
        long_idx, left, right, fraction = _fixed_k_interpolation_indices(
            z_axis, rho_axis[i], rho_axis[i + 1]
        )
        for j_long, j_left, j_right, weight_right in zip(
            long_idx, left, right, fraction
        ):
            cal_rows.extend((row, row, row))
            cal_cols.extend((
                i * n_z + j_left,
                i * n_z + j_right,
                (i + 1) * n_z + j_long,
            ))
            cal_values.extend((1.0 - weight_right, weight_right, -1.0))
            row += 1
    calendar = coo_matrix(
        (cal_values, (cal_rows, cal_cols)), shape=(row, n_values)
    ).tocsr()
    return butterfly, calendar


def project_band_surfaces_fixed_k(iv_lower: np.ndarray,
                                  iv_upper: np.ndarray,
                                  z_axis: np.ndarray,
                                  rho_axis: np.ndarray,
                                  tolerance: float = 1e-6,
                                  min_iv: float = 1e-4):
    """Jointly adjust band surfaces under fixed-k calendar and butterfly conditions.

    The optimization uses total implied variance as its variable. It minimizes
    the L1 change from both input surfaces subject to the discrete butterfly
    chord conditions, the GNO paper's fixed-k calendar mapping, positivity, and
    lower/upper ordering. The sparse HiGHS solver makes the result deterministic
    and avoids the cycling that can occur when the two adjustments are alternated.
    """
    iv_lower = np.asarray(iv_lower, dtype=float)
    iv_upper = np.asarray(iv_upper, dtype=float)
    z_axis = np.asarray(z_axis, dtype=float)
    rho_axis = np.asarray(rho_axis, dtype=float)
    expected_shape = (rho_axis.size, z_axis.size)
    if iv_lower.shape != expected_shape or iv_upper.shape != expected_shape:
        raise ValueError("band surface shapes must match rho_axis and z_axis")
    if not np.all(np.isfinite(iv_lower)) or not np.all(np.isfinite(iv_upper)):
        raise ValueError("band surfaces must contain only finite values")
    if np.any(rho_axis <= 0) or np.any(np.diff(rho_axis) <= 0):
        raise ValueError("rho_axis must be positive and strictly increasing")
    if z_axis.size < 3 or np.any(np.diff(z_axis) <= 0):
        raise ValueError("z_axis must contain at least three increasing values")
    if tolerance <= 0 or min_iv <= 0:
        raise ValueError("tolerance and min_iv must be positive")

    n_rho, n_z = expected_shape
    n_values = n_rho * n_z
    total_variance_lower = (iv_lower * rho_axis[:, None]) ** 2
    total_variance_upper = (iv_upper * rho_axis[:, None]) ** 2
    butterfly, calendar = _fixed_k_shape_constraint_matrices(z_axis, rho_axis)
    shape_constraints = vstack((butterfly, calendar), format="csr")

    zero_shape = csr_matrix((shape_constraints.shape[0], n_values))
    zero_values = csr_matrix((n_values, n_values))
    identity = eye(n_values, format="csr")

    # Variables are [w_lower, w_upper, abs_change_lower, abs_change_upper].
    constraint_matrix = vstack((
        hstack((shape_constraints, zero_shape, zero_shape, zero_shape)),
        hstack((zero_shape, shape_constraints, zero_shape, zero_shape)),
        hstack((identity, -identity, zero_values, zero_values)),
        hstack((identity, zero_values, -identity, zero_values)),
        hstack((-identity, zero_values, -identity, zero_values)),
        hstack((zero_values, identity, zero_values, -identity)),
        hstack((zero_values, -identity, zero_values, -identity)),
    ), format="csr")
    constraint_bounds = np.concatenate((
        np.zeros(2 * shape_constraints.shape[0] + n_values),
        total_variance_lower.ravel(),
        -total_variance_lower.ravel(),
        total_variance_upper.ravel(),
        -total_variance_upper.ravel(),
    ))
    objective = np.concatenate((np.zeros(2 * n_values), np.ones(2 * n_values)))
    variance_floor = np.repeat((rho_axis * min_iv) ** 2, n_z)
    variable_bounds = (
        [(float(value), None) for value in np.tile(variance_floor, 2)]
        + [(0.0, None)] * (2 * n_values)
    )

    result = linprog(
        objective,
        A_ub=constraint_matrix,
        b_ub=constraint_bounds,
        bounds=variable_bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"fixed-k band projection failed: {result.message}")

    variance_floor_grid = variance_floor.reshape(expected_shape)
    raw_lower_variance = result.x[:n_values].reshape(expected_shape)
    raw_upper_variance = result.x[n_values:2 * n_values].reshape(expected_shape)
    lower_floor_adjustments = int(np.sum(raw_lower_variance < variance_floor_grid))
    upper_floor_adjustments = int(np.sum(raw_upper_variance < variance_floor_grid))

    # HiGHS may return values a few machine-precision units below a very small
    # positive lower bound. Enforce the configured floor before taking square
    # roots so these harmless solver tolerances cannot create NaN IV values.
    projected_lower_variance = np.maximum(
        raw_lower_variance, variance_floor_grid
    )
    projected_upper_variance = np.maximum(
        raw_upper_variance, variance_floor_grid
    )
    projected_lower = np.sqrt(projected_lower_variance) / rho_axis[:, None]
    projected_upper = np.sqrt(projected_upper_variance) / rho_axis[:, None]

    initial_lower_butterfly = check_butterfly_arbitrage(
        iv_lower, z_axis, rho_axis ** 2, tolerance
    )
    initial_upper_butterfly = check_butterfly_arbitrage(
        iv_upper, z_axis, rho_axis ** 2, tolerance
    )
    initial_lower_calendar = check_calendar_arbitrage_fixed_k(
        iv_lower, z_axis, rho_axis, tolerance
    )
    initial_upper_calendar = check_calendar_arbitrage_fixed_k(
        iv_upper, z_axis, rho_axis, tolerance
    )
    final_lower_butterfly = check_butterfly_arbitrage(
        projected_lower, z_axis, rho_axis ** 2, tolerance
    )
    final_upper_butterfly = check_butterfly_arbitrage(
        projected_upper, z_axis, rho_axis ** 2, tolerance
    )
    final_lower_calendar = check_calendar_arbitrage_fixed_k(
        projected_lower, z_axis, rho_axis, tolerance
    )
    final_upper_calendar = check_calendar_arbitrage_fixed_k(
        projected_upper, z_axis, rho_axis, tolerance
    )
    crossings = int(np.sum(projected_lower > projected_upper + tolerance))
    finite = bool(
        np.all(np.isfinite(projected_lower))
        and np.all(np.isfinite(projected_upper))
    )
    converged = finite and not any((
        final_lower_butterfly.has_violations(),
        final_upper_butterfly.has_violations(),
        final_lower_calendar.has_violations(),
        final_upper_calendar.has_violations(),
        crossings,
    ))

    return projected_lower, projected_upper, {
        "converged": converged,
        "solver_success": bool(result.success),
        "solver_status": result.message,
        "solver_iterations": int(result.nit),
        "objective_value": float(result.fun),
        "finite": finite,
        "variance_floor_adjustments_lower": lower_floor_adjustments,
        "variance_floor_adjustments_upper": upper_floor_adjustments,
        "initial_lower_butterfly": initial_lower_butterfly,
        "initial_upper_butterfly": initial_upper_butterfly,
        "initial_lower_calendar": initial_lower_calendar,
        "initial_upper_calendar": initial_upper_calendar,
        "final_lower_butterfly": final_lower_butterfly,
        "final_upper_butterfly": final_upper_butterfly,
        "final_lower_calendar": final_lower_calendar,
        "final_upper_calendar": final_upper_calendar,
        "band_crossings": crossings,
        "projection_distance_lower": float(np.linalg.norm(projected_lower - iv_lower)),
        "projection_distance_upper": float(np.linalg.norm(projected_upper - iv_upper)),
    }


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
