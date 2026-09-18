"""Evaluation helpers for held-out quote masking and surface-level summaries."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from volatility_smoothing.utils.train.edge_index import generate_edge_index


def stratified_context_target_indices(
    rho: torch.Tensor,
    z: torch.Tensor,
    *,
    target_fraction: float,
    seed: int,
    n_bins_rho: int,
    n_bins_z: int,
    rho_range: tuple[float, float],
    z_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic coordinate-stratified context/target split.

    The split uses only the quote coordinates, never implied volatility or model
    errors. Sampling the target points separately within each spatial bin keeps
    the held-out set distributed across the observed surface. A singleton bin is
    retained in the context set so that no populated region is removed entirely.
    """
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must lie strictly between zero and one")
    if n_bins_rho < 1 or n_bins_z < 1:
        raise ValueError("the number of spatial bins must be positive")

    rho_np = np.asarray(rho.detach().cpu(), dtype=float).reshape(-1)
    z_np = np.asarray(z.detach().cpu(), dtype=float).reshape(-1)
    if rho_np.size != z_np.size or rho_np.size < 2:
        raise ValueError("rho and z must contain the same number of at least two points")

    rho_edges = np.linspace(rho_range[0], rho_range[1], n_bins_rho + 1)
    z_edges = np.linspace(z_range[0], z_range[1], n_bins_z + 1)
    rho_bin = np.clip(np.digitize(rho_np, rho_edges[1:-1]), 0, n_bins_rho - 1)
    z_bin = np.clip(np.digitize(z_np, z_edges[1:-1]), 0, n_bins_z - 1)
    bin_id = rho_bin * n_bins_z + z_bin

    rng = np.random.default_rng(seed)
    target_parts: list[np.ndarray] = []
    for current_bin in np.unique(bin_id):
        members = np.flatnonzero(bin_id == current_bin)
        if members.size < 2:
            continue
        n_target = int(np.rint(target_fraction * members.size))
        n_target = min(members.size - 1, max(1, n_target))
        target_parts.append(rng.choice(members, size=n_target, replace=False))

    if not target_parts:
        raise ValueError("the spatial split produced no target points")

    target_np = np.sort(np.concatenate(target_parts).astype(np.int64, copy=False))
    is_target = np.zeros(rho_np.size, dtype=bool)
    is_target[target_np] = True
    context_np = np.flatnonzero(~is_target).astype(np.int64, copy=False)

    if context_np.size == 0 or target_np.size == 0:
        raise ValueError("both context and target sets must be non-empty")
    if np.intersect1d(context_np, target_np).size:
        raise AssertionError("context and target indices overlap")
    if context_np.size + target_np.size != rho_np.size:
        raise AssertionError("context and target indices do not partition the surface")

    return torch.from_numpy(context_np), torch.from_numpy(target_np)


def subset_node_data(data, indices: torch.Tensor):
    """Copy a PyG data object while retaining only selected node-level values."""
    subset = data.clone()
    indices = indices.to(dtype=torch.long, device=data["r"].device)
    n_nodes = int(data["r"].shape[0])
    for key in data.keys():
        value = data[key]
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n_nodes:
            subset[key] = value.index_select(0, indices)
    subset.num_nodes = int(indices.numel())
    return subset


def build_masked_gno_input(
    data,
    loss,
    *,
    surface_index: int,
    target_fraction: float,
    mask_seed: int,
    n_bins_rho: int,
    n_bins_z: int,
    rho_range: tuple[float, float],
    z_range: tuple[float, float],
):
    """Build a GNO call with context IVs as inputs and target points as queries."""
    context_idx, target_idx = stratified_context_target_indices(
        data["r"],
        data["z"],
        target_fraction=target_fraction,
        seed=mask_seed + int(surface_index),
        n_bins_rho=n_bins_rho,
        n_bins_z=n_bins_z,
        rho_range=rho_range,
        z_range=z_range,
    )
    context_data = subset_node_data(data, context_idx)
    target_data = subset_node_data(data, target_idx)

    base_input, aux = loss.load_input(context_data)
    target_pos = torch.cat((target_data["r"], target_data["z"]), dim=1)
    pos_y = torch.cat((target_pos, base_input["pos_y"]), dim=0)
    edge_index = generate_edge_index(
        base_input["pos_x"],
        pos_y,
        subsample_size=loss.subsample_size,
        radius=loss.radius,
    )
    input_dict = {
        "x": base_input["x"],
        "pos_x": base_input["pos_x"],
        "pos_y": pos_y,
        "edge_index": edge_index,
    }
    aux = dict(aux)
    aux.update(
        n_target=int(target_idx.numel()),
        context_indices=context_idx,
        target_indices=target_idx,
    )

    if input_dict["x"].shape[0] != context_idx.numel():
        raise AssertionError("the GNO input contains values outside the context set")
    if target_pos.shape[0] != target_idx.numel():
        raise AssertionError("the GNO query does not contain every target location")

    return context_data, target_data, input_dict, aux


def read_masked_gno_output(output, aux):
    """Separate held-out target predictions from the regular-grid predictions."""
    _, iv_y = output
    n_target = int(aux["n_target"])
    iv_target = iv_y[:n_target]
    iv_grid = iv_y[n_target:]
    sections = [int(np.prod(grid.size())) for grid in aux["grids"]]
    if int(iv_grid.shape[0]) != sum(sections):
        raise ValueError("unexpected number of GNO grid outputs")
    grid_outputs = tuple(
        values.view(aux["grids"][i].size())
        for i, values in enumerate(torch.split(iv_grid, sections, dim=0))
    )
    return (iv_target, *grid_outputs)


def surface_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> tuple[float, float, float]:
    """Mean and percentile interval from resampled complete daily surfaces."""
    values_np = np.asarray(values, dtype=float).reshape(-1)
    values_np = values_np[np.isfinite(values_np)]
    if values_np.size == 0:
        raise ValueError("cannot summarize an empty sequence")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    rng = np.random.default_rng(seed)
    draws = rng.choice(values_np, size=(n_resamples, values_np.size), replace=True)
    means = draws.mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return float(values_np.mean()), float(lower), float(upper)
