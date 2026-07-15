"""Limiter geometry encoding utilities for STRONG-RMMD."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn

from dgknet_baseline.phases.phase0_data_pipeline import CDFVariableExtractor


def fourier_harmonics_from_contour(rlim: np.ndarray, ylim: np.ndarray, n_harmonics: int = 8) -> np.ndarray:
    """Return [Rcos, Rsin, Zcos, Zsin] harmonics for a closed contour."""
    r = np.asarray(rlim, dtype=np.float32).reshape(-1)
    z = np.asarray(ylim, dtype=np.float32).reshape(-1)
    if r.shape != z.shape:
        raise ValueError('RLIM/YLIM length mismatch')
    theta = np.linspace(0.0, 2.0 * np.pi, r.size, endpoint=False)
    coeffs = []
    for k in range(1, n_harmonics + 1):
        c = np.cos(k * theta)
        s = np.sin(k * theta)
        coeffs.extend([
            float(np.dot(r, c) / r.size),
            float(np.dot(r, s) / r.size),
            float(np.dot(z, c) / z.size),
            float(np.dot(z, s) / z.size),
        ])
    return np.asarray(coeffs, dtype=np.float32)


def geometry_frobenius_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


class LimiterGeometryEncoder(nn.Module):
    def __init__(self, n_harmonics: int = 8, embedding_dim: int = 128):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.encoder = nn.Sequential(
            nn.Linear(4 * n_harmonics, 256),
            nn.SiLU(),
            nn.Linear(256, embedding_dim),
            nn.Tanh(),
        )

    def extract_harmonics_from_cdf(self, cdf_path, machine_name: str) -> np.ndarray:
        extractor = CDFVariableExtractor(str(cdf_path), verbose=False)
        try:
            contour = extractor._extract_limiter_contour(extractor.cdf, time_index=0)
            if contour is None:
                contour = extractor._load_reference_limiter_contour()
            if contour is None:
                geom = extractor.extract_limiter_geometry_tensor_t0()
                return fourier_harmonics_from_contour(geom[:, 0], geom[:, 1], self.n_harmonics)
            rlim, ylim = contour
            return fourier_harmonics_from_contour(rlim, ylim, self.n_harmonics)
        finally:
            extractor.close()

    def forward(self, harmonics_normalized: torch.Tensor) -> torch.Tensor:
        return self.encoder(harmonics_normalized)
