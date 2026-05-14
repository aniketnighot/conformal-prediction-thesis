"""
Mondrian conformal prediction with spatial binning.

Partitions the domain into bins and computes separate quantiles
for each bin to capture spatial heterogeneity.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MondrianBin:
    """Represents a single bin in the Mondrian partition."""
    rho_min: float
    rho_max: float
    z_min: float
    z_max: float
    scores: List[float]
    quantile: Optional[float] = None

    def contains(self, rho: float, z: float) -> bool:
        """Check if a point (rho, z) falls within this bin."""
        return (self.rho_min <= rho < self.rho_max and
                self.z_min <= z < self.z_max)

    def __repr__(self):
        n_scores = len(self.scores) if self.scores else 0
        q_str = f"{self.quantile:.4f}" if self.quantile is not None else "None"
        return (f"MondrianBin(rho=[{self.rho_min:.3f}, {self.rho_max:.3f}), "
                f"z=[{self.z_min:.3f}, {self.z_max:.3f}), "
                f"n_scores={n_scores}, quantile={q_str})")


class MondrianConformal:
    """
    Mondrian Conformal Prediction with spatial binning.

    Partitions the (rho, z) domain into Brho x Bz bins and computes
    separate conformal quantiles for each bin. This handles spatial
    heterogeneity where different regions have different uncertainty levels.

    Parameters:
    -----------
    n_bins_rho : int
        Number of bins along rho (sqrt of time-to-maturity) axis
    n_bins_z : int
        Number of bins along z (normalized moneyness) axis
    rho_range : tuple
        (min, max) range for rho dimension
    z_range : tuple
        (min, max) range for z dimension
    alpha : float
        Miscoverage level (1 - coverage_level)
    min_samples_per_bin : int
        Minimum number of samples required per bin to compute quantile.
        If a bin has fewer samples, use global quantile instead.

    Attributes:
    -----------
    bins : List[List[MondrianBin]]
        2D array of bins [rho_idx][z_idx]
    global_quantile : float
        Fallback quantile for bins with insufficient data
    """

    def __init__(self,
                 n_bins_rho: int = 6,
                 n_bins_z: int = 6,
                 rho_range: Tuple[float, float] = (0.01, 1.0),
                 z_range: Tuple[float, float] = (-1.5, 0.5),
                 alpha: float = 0.1,
                 min_samples_per_bin: int = 10,
                 shrinkage: float = 0.3):

        self.n_bins_rho = n_bins_rho
        self.n_bins_z = n_bins_z
        self.rho_range = rho_range
        self.z_range = z_range
        self.alpha = alpha
        self.min_samples_per_bin = min_samples_per_bin
        self.shrinkage = shrinkage

        # Create bin edges
        self.rho_edges = np.linspace(rho_range[0], rho_range[1], n_bins_rho + 1)
        self.z_edges = np.linspace(z_range[0], z_range[1], n_bins_z + 1)

        # Initialize bins
        self.bins: List[List[MondrianBin]] = []
        for i in range(n_bins_rho):
            row = []
            for j in range(n_bins_z):
                bin_obj = MondrianBin(
                    rho_min=self.rho_edges[i],
                    rho_max=self.rho_edges[i+1],
                    z_min=self.z_edges[j],
                    z_max=self.z_edges[j+1],
                    scores=[]
                )
                row.append(bin_obj)
            self.bins.append(row)

        self.global_quantile: Optional[float] = None
        self._calibrated = False

    def _find_bin_indices(self, rho: float, z: float) -> Tuple[int, int]:
        """Find the bin indices for a point (rho, z)."""
        # Find rho bin
        rho_idx = np.searchsorted(self.rho_edges[1:], rho)
        rho_idx = min(rho_idx, self.n_bins_rho - 1)

        # Find z bin
        z_idx = np.searchsorted(self.z_edges[1:], z)
        z_idx = min(z_idx, self.n_bins_z - 1)

        return rho_idx, z_idx

    def calibrate(self,
                  rho: np.ndarray,
                  z: np.ndarray,
                  scores: np.ndarray) -> None:
        """
        Calibrate by computing quantiles for each bin.

        Parameters:
        -----------
        rho : np.ndarray
            rho coordinates of calibration points (N,)
        z : np.ndarray
            z coordinates of calibration points (N,)
        scores : np.ndarray
            Nonconformity scores at calibration points (N,)
        """
        # Reset bins
        for i in range(self.n_bins_rho):
            for j in range(self.n_bins_z):
                self.bins[i][j].scores = []

        # Assign scores to bins
        for rho_i, z_i, score_i in zip(rho, z, scores):
            i, j = self._find_bin_indices(rho_i, z_i)
            self.bins[i][j].scores.append(score_i)

        # Compute global quantile (fallback)
        n_total = len(scores)
        q_level = np.ceil((n_total + 1) * (1 - self.alpha)) / n_total
        self.global_quantile = np.quantile(scores, q_level)

        # Compute per-bin quantiles with regularization
        for i in range(self.n_bins_rho):
            for j in range(self.n_bins_z):
                bin_obj = self.bins[i][j]
                n_bin = len(bin_obj.scores)

                if n_bin >= self.min_samples_per_bin:
                    # Sufficient data: compute bin-specific quantile
                    q_level_bin = np.ceil((n_bin + 1) * (1 - self.alpha)) / n_bin
                    q_raw = np.quantile(bin_obj.scores, q_level_bin)

                    # Regularization: shrink toward global quantile to prevent overfitting
                    # bin_quantile = shrinkage * global + (1 - shrinkage) * bin_specific
                    bin_obj.quantile = self.shrinkage * self.global_quantile + (1 - self.shrinkage) * q_raw
                else:
                    # Insufficient data: use global quantile
                    bin_obj.quantile = self.global_quantile

        self._calibrated = True

    def get_quantile(self, rho: float, z: float) -> float:
        """
        Get the conformal quantile for a point (rho, z).

        Parameters:
        -----------
        rho : float
            rho coordinate
        z : float
            z coordinate

        Returns:
        --------
        quantile : float
            Conformal quantile for this bin
        """
        if not self._calibrated:
            raise ValueError("MondrianConformal not calibrated. Call calibrate() first.")

        # Handle out-of-bounds points
        if (rho < self.rho_range[0] or rho > self.rho_range[1] or
            z < self.z_range[0] or z > self.z_range[1]):
            return self.global_quantile

        i, j = self._find_bin_indices(rho, z)
        return self.bins[i][j].quantile

    def get_quantiles(self, rho: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Vectorized version: get quantiles for multiple points.

        Parameters:
        -----------
        rho : np.ndarray
            rho coordinates (N,)
        z : np.ndarray
            z coordinates (N,)

        Returns:
        --------
        quantiles : np.ndarray
            Conformal quantiles (N,)
        """
        quantiles = np.zeros(len(rho))
        for idx, (rho_i, z_i) in enumerate(zip(rho, z)):
            quantiles[idx] = self.get_quantile(rho_i, z_i)
        return quantiles

    def summary(self) -> Dict:
        """
        Get summary statistics about the Mondrian partition.

        Returns:
        --------
        summary : dict
            Dictionary with statistics about bins and quantiles
        """
        if not self._calibrated:
            return {"status": "not calibrated"}

        bin_counts = []
        bin_quantiles = []
        empty_bins = 0
        sparse_bins = 0

        for i in range(self.n_bins_rho):
            for j in range(self.n_bins_z):
                bin_obj = self.bins[i][j]
                n = len(bin_obj.scores)
                bin_counts.append(n)
                if bin_obj.quantile is not None:
                    bin_quantiles.append(bin_obj.quantile)

                if n == 0:
                    empty_bins += 1
                elif n < self.min_samples_per_bin:
                    sparse_bins += 1

        return {
            "status": "calibrated",
            "total_bins": self.n_bins_rho * self.n_bins_z,
            "n_bins_rho": self.n_bins_rho,
            "n_bins_z": self.n_bins_z,
            "empty_bins": empty_bins,
            "sparse_bins": sparse_bins,
            "well_populated_bins": len(bin_counts) - empty_bins - sparse_bins,
            "total_calibration_points": sum(bin_counts),
            "mean_points_per_bin": np.mean(bin_counts),
            "median_points_per_bin": np.median(bin_counts),
            "min_points_per_bin": np.min(bin_counts),
            "max_points_per_bin": np.max(bin_counts),
            "global_quantile": self.global_quantile,
            "mean_bin_quantile": np.mean(bin_quantiles),
            "std_bin_quantile": np.std(bin_quantiles),
            "min_bin_quantile": np.min(bin_quantiles),
            "max_bin_quantile": np.max(bin_quantiles),
        }

    def __repr__(self):
        status = "calibrated" if self._calibrated else "not calibrated"
        return (f"MondrianConformal({self.n_bins_rho}x{self.n_bins_z} bins, "
                f"alpha={self.alpha}, {status})")
