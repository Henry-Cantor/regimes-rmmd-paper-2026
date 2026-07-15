"""Visualization utilities for STRONG-RMMD."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

os.environ["MPLBACKEND"] = "agg"
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_spectral_comparison(
    eigenvalue_dict: Dict[str, np.ndarray],
    machine_colors: Optional[Dict[str, str]] = None,
    title: str = "Spectral Comparison Across Machines",
) -> plt.Figure:
    """
    Plot eigenvalue spectra for all machines on same figure.

    Red dots indicate "universal" modes (low variance across machines).
    """
    if not eigenvalue_dict:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=(12, 6))

    for machine_name, eigs in eigenvalue_dict.items():
        eigs_sorted = np.sort(np.abs(np.real(eigs)))[::-1]
        color = machine_colors.get(machine_name, 'C0') if machine_colors else None
        ax.semilogy(range(len(eigs_sorted)), eigs_sorted, 'o-', label=machine_name, color=color, alpha=0.7)

    ax.set_xlabel('Mode Index (ranked by magnitude)')
    ax.set_ylabel('|λ| (log scale)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_residuals_vs_predictions(
    true_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    time_indices: Optional[List[int]] = None,
    title: str = "Residuals vs Ground Truth",
) -> plt.Figure:
    """
    Scatter plot: residuals (true-pred) vs ground truth values.

    Helps diagnose systematic biases.
    """
    residuals = true_trajectory - pred_trajectory
    true_flat = true_trajectory.flatten()
    residuals_flat = residuals.flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(true_flat, residuals_flat, alpha=0.3, s=10)
    ax1.axhline(0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Ground Truth')
    ax1.set_ylabel('Residual (True - Pred)')
    ax1.set_title('Residual Scatter')
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals_flat, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual Distribution')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    return fig


def plot_d_res_structure(
    d_res_batch: torch.Tensor,
    machine_name: str = '',
    n_samples: int = 3,
) -> plt.Figure:
    """
    Visualize off-diagonal dissipation matrices for a few samples.

    Shows that D_res is symmetric, zero-diagonal, and captures coupling structure.
    """
    if d_res_batch.ndim != 3:
        raise ValueError("Expected batch of matrices: shape (batch, latent_dim, latent_dim)")

    n_show = min(n_samples, d_res_batch.shape[0])
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 4))

    if n_show == 1:
        axes = [axes]

    for idx in range(n_show):
        d_res = d_res_batch[idx].detach().cpu().numpy()
        im = axes[idx].imshow(d_res, cmap='RdBu_r', aspect='auto')
        axes[idx].set_title(f'Sample {idx+1}')
        axes[idx].set_xlabel('Latent Dim')
        axes[idx].set_ylabel('Latent Dim')
        plt.colorbar(im, ax=axes[idx])

    fig.suptitle(f'D_res Matrices ({machine_name})', fontsize=14, fontweight='bold')
    return fig


def plot_loss_curves(
    loss_history: Dict[str, List[float]],
    log_scale: bool = False,
) -> plt.Figure:
    """Plot all loss components vs epoch."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for loss_name, values in loss_history.items():
        ax.plot(range(len(values)), values, label=loss_name, linewidth=2, alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if log_scale:
        ax.set_yscale('log')
    ax.set_title('Training Loss Curves')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    return fig


def plot_nrmse_improvement(
    machine_names: List[str],
    dgknet_nrmse: List[float],
    rmmd_nrmse: List[float],
    title: str = "NRMSE: DGKNet vs RMMD",
) -> plt.Figure:
    """Bar plot comparing NRMSE across machines."""
    x = np.arange(len(machine_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, dgknet_nrmse, width, label='DGKNet', alpha=0.8)
    ax.bar(x + width / 2, rmmd_nrmse, width, label='RMMD', alpha=0.8)

    ax.set_ylabel('NRMSE')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(machine_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    return fig
