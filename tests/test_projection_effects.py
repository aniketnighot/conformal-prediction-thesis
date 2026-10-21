"""Verify the three-stage projection decomposition at held-out quote locations."""

import unittest

import numpy as np

from volatility_smoothing.conformal.projection_effects import (
    summarize_band_stages,
)


class ProjectionEffectsTest(unittest.TestCase):
    def test_wider_projected_bands_can_lose_coverage_after_shifting(self):
        market = np.array([0.20, 0.26, 0.31])
        prediction = np.full(3, 0.20)
        direct = (np.full(3, 0.10), np.full(3, 0.30))
        interpolated = (np.full(3, 0.15), np.full(3, 0.35))
        projected = (
            np.array([0.25, 0.10, 0.32]),
            np.full(3, 0.45),
        )

        result = summarize_band_stages(
            market, prediction, {
                "direct": direct,
                "interpolated": interpolated,
                "projected": projected,
            }
        )

        self.assertAlmostEqual(result["direct_coverage"], 2 / 3)
        self.assertAlmostEqual(result["interpolated_coverage"], 1.0)
        self.assertAlmostEqual(result["projected_coverage"], 1 / 3)
        self.assertGreater(
            result["projected_mean_width"], result["interpolated_mean_width"]
        )
        self.assertEqual(result["interpolation_coverage_gained_count"], 1)
        self.assertEqual(result["optimization_coverage_lost_count"], 2)
        self.assertEqual(result["optimization_coverage_gained_count"], 0)
        self.assertEqual(result["projected_gno_inside_count"], 1)
        self.assertAlmostEqual(
            result["optimization_lower_signed_mean"],
            np.mean(projected[0] - interpolated[0]),
        )

    def test_identity_round_trip_has_no_interpolation_effect(self):
        market = np.array([0.2, 0.3])
        prediction = np.array([0.22, 0.28])
        bounds = (np.array([0.1, 0.2]), np.array([0.3, 0.4]))
        result = summarize_band_stages(
            market, prediction,
            {"direct": bounds, "interpolated": bounds, "projected": bounds},
        )

        for label in ("interpolation", "optimization"):
            self.assertEqual(result[f"{label}_coverage_lost_count"], 0)
            self.assertEqual(result[f"{label}_coverage_gained_count"], 0)
            self.assertEqual(result[f"{label}_mean_width_change"], 0)
            self.assertEqual(result[f"{label}_lower_abs_mean"], 0)
            self.assertEqual(result[f"{label}_upper_abs_mean"], 0)


if __name__ == "__main__":
    unittest.main()
