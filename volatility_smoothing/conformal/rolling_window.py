"""
Rolling window calibration for conformal prediction.

Maintains a sliding window of K most recent calibration surfaces,
allowing quantiles to adapt to changing market conditions.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CalibrationSurface:
    quote_datetime: str
    rho: np.ndarray
    z: np.ndarray
    scores: np.ndarray

    def __len__(self):
        return len(self.scores)


class RollingWindowCalibration:
    """
    Rolling window calibration for conformal prediction.

    Maintains a sliding window of K most recent calibration surfaces.
    When a new surface is added, the oldest is dropped.
    """

    def __init__(self, window_size: int = 20):
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        self.window_size = window_size
        self.surfaces: List[CalibrationSurface] = []

    def add_surface(self,
                   quote_datetime: str,
                   rho: np.ndarray,
                   z: np.ndarray,
                   scores: np.ndarray) -> None:
        surface = CalibrationSurface(
            quote_datetime=quote_datetime,
            rho=rho,
            z=z,
            scores=scores
        )

        self.surfaces.append(surface)

        if len(self.surfaces) > self.window_size:
            self.surfaces.pop(0)

    def get_calibration_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.surfaces:
            raise ValueError("No calibration surfaces available")

        rho_all = np.concatenate([s.rho for s in self.surfaces])
        z_all = np.concatenate([s.z for s in self.surfaces])
        scores_all = np.concatenate([s.scores for s in self.surfaces])

        return rho_all, z_all, scores_all

    def get_surface_dates(self) -> List[str]:
        return [s.quote_datetime for s in self.surfaces]

    def is_ready(self) -> bool:
        return len(self.surfaces) > 0

    def is_full(self) -> bool:
        return len(self.surfaces) == self.window_size

    def summary(self) -> dict:
        if not self.surfaces:
            return {
                "status": "empty",
                "window_size": self.window_size,
                "n_surfaces": 0
            }

        n_points_per_surface = [len(s) for s in self.surfaces]

        return {
            "status": "full" if self.is_full() else "partial",
            "window_size": self.window_size,
            "n_surfaces": len(self.surfaces),
            "total_calibration_points": sum(n_points_per_surface),
            "mean_points_per_surface": np.mean(n_points_per_surface),
            "oldest_surface": self.surfaces[0].quote_datetime,
            "newest_surface": self.surfaces[-1].quote_datetime,
            "surface_dates": self.get_surface_dates()
        }

    def __repr__(self):
        status = "full" if self.is_full() else f"partial ({len(self.surfaces)}/{self.window_size})"
        return f"RollingWindowCalibration(K={self.window_size}, {status})"

    def __len__(self):
        return len(self.surfaces)
