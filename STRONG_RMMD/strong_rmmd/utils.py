"""Utility helpers for STRONG-RMMD."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@lru_cache(maxsize=128)
def cached_profile_normalize(profile_hash: int, min_val: float, max_val: float) -> Tuple[float, float]:
    """Cached normalization constants."""
    scale = max_val - min_val
    return (min_val, scale)


def synchronize_profiles_across_machines(
    batches: Dict[str, np.ndarray],
    target_grid_size: int = 65,
) -> Dict[str, np.ndarray]:
    """
    Regrid all profiles to common radial grid for cross-machine comparison.

    Args:
        batches: {machine_name: profile array [n_shots, n_profile_vars, radial_points]}
        target_grid_size: target number of radial points

    Returns:
        Regrided batches with uniform shape across machines
    """
    regrided = {}
    for machine_name, profiles in batches.items():
        if profiles.shape[-1] == target_grid_size:
            regrided[machine_name] = profiles
        else:
            n_shots, n_vars = profiles.shape[0], profiles.shape[1]
            regrid = np.zeros((n_shots, n_vars, target_grid_size), dtype=profiles.dtype)
            for var_idx in range(n_vars):
                for shot_idx in range(n_shots):
                    old_grid = np.linspace(0, 1, profiles.shape[2])
                    new_grid = np.linspace(0, 1, target_grid_size)
                    regrid[shot_idx, var_idx, :] = np.interp(new_grid, old_grid, profiles[shot_idx, var_idx, :])
            regrided[machine_name] = regrid
    return regrided


def compute_spectral_statistics(
    latent_trajectories: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    """
    Compute spectral properties of latent dynamics per machine.

    Returns per-machine mean/std of latent mode energies.
    """
    stats = {}
    for machine_name, trajectory in latent_trajectories.items():
        mode_energy = torch.sum(trajectory ** 2, dim=0)  # [latent_dim]
        mode_energy_normalized = mode_energy / torch.sum(mode_energy)

        stats[machine_name] = {
            "mean_energy": float(torch.mean(mode_energy_normalized).item()),
            "std_energy": float(torch.std(mode_energy_normalized).item()),
            "max_mode": float(torch.max(mode_energy_normalized).item()),
            "min_mode": float(torch.min(mode_energy_normalized).item()),
            "entropy": float(-torch.sum(mode_energy_normalized * torch.log(mode_energy_normalized + 1e-8)).item()),
        }
    return stats


def validate_checkpoint_structure(checkpoint_path: Path) -> Tuple[bool, str]:
    """
    Verify checkpoint has required fields for RMMD training.

    Returns (is_valid, reason)
    """
    if not checkpoint_path.exists():
        return False, f"Checkpoint not found: {checkpoint_path}"

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        required_keys = ["model_state_dict", "optimizer_state_dict", "epoch"]
        missing = [k for k in required_keys if k not in checkpoint]
        if missing:
            return False, f"Missing keys in checkpoint: {missing}"
        return True, "OK"
    except Exception as e:
        return False, f"Error loading checkpoint: {e}"


def prepare_batch_for_inference(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Move batch to device and validate shapes."""
    prepared = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared[key] = value.to(device)
        else:
            prepared[key] = value
    return prepared


def exponential_moving_average(
    old_value: torch.Tensor,
    new_value: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Compute EMA for metrics tracking."""
    return (1.0 - alpha) * old_value + alpha * new_value


class RunningMetrics:
    """Track rolling statistics during training."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values: Dict[str, list] = {}

    def update(self, name: str, value: float) -> None:
        """Update a metric value."""
        if name not in self.values:
            self.values[name] = []
        self.values[name].append(float(value))
        if len(self.values[name]) > self.window_size:
            self.values[name].pop(0)

    def get_mean(self, name: str) -> float:
        """Get rolling mean for a metric."""
        if name not in self.values or not self.values[name]:
            return 0.0
        return float(np.mean(self.values[name]))

    def get_std(self, name: str) -> float:
        """Get rolling std for a metric."""
        if name not in self.values or len(self.values[name]) < 2:
            return 0.0
        return float(np.std(self.values[name]))

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all tracked metrics."""
        return {
            name: {"mean": self.get_mean(name), "std": self.get_std(name)}
            for name in self.values.keys()
        }
