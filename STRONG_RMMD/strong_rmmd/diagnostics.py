"""Diagnostics for STRONG-RMMD.

The core phase-1 diagnostic is a data-driven estimate of off-diagonal
dissipation strength from residual trajectories. The implementation here is
designed to be usable directly on saved rollouts from a DGKNet baseline or any
other predictor that can provide true/predicted trajectories.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


@dataclass
class DResEstimate:
    machine_name: str
    frobenius_norm: float
    diagonal_norm: float
    off_diagonal_norm: float
    off_diagonal_ratio: float
    mean_residual: float
    std_residual: float

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "machine_name": self.machine_name,
            "frobenius_norm": float(self.frobenius_norm),
            "diagonal_norm": float(self.diagonal_norm),
            "off_diagonal_norm": float(self.off_diagonal_norm),
            "off_diagonal_ratio": float(self.off_diagonal_ratio),
            "mean_residual": float(self.mean_residual),
            "std_residual": float(self.std_residual),
        }


def _as_2d_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim > 2:
        leading = int(np.prod(array.shape[:-1]))
        return array.reshape(leading, array.shape[-1])
    return array


def _safe_pearsonr(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size < 2 or y_arr.size < 2:
        return float("nan")
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")
    x_centered = x_arr - x_arr.mean()
    y_centered = y_arr - y_arr.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denom)


def estimate_d_res_from_residuals(
    true_trajectory: np.ndarray,
    predicted_trajectory: np.ndarray,
    machine_name: str,
) -> DResEstimate:
    """Estimate an off-diagonal dissipation proxy from residual trajectories.

    Parameters
    ----------
    true_trajectory:
        Array shaped like ``(time, features)`` or ``(shots, time, features)``.
    predicted_trajectory:
        Same shape as ``true_trajectory``.
    machine_name:
        Machine label attached to the estimate.
    """

    true_array = np.asarray(true_trajectory, dtype=float)
    pred_array = np.asarray(predicted_trajectory, dtype=float)
    if true_array.shape != pred_array.shape:
        raise ValueError(f"Shape mismatch: true={true_array.shape} pred={pred_array.shape}")

    residuals = true_array - pred_array
    flat_residuals = _as_2d_array(residuals)

    covariance = np.cov(flat_residuals, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.asarray([[float(covariance)]], dtype=float)
    diagonal = np.diag(np.diag(covariance))
    off_diagonal = covariance - diagonal

    frobenius_norm = float(np.linalg.norm(covariance, ord="fro"))
    diagonal_norm = float(np.linalg.norm(diagonal, ord="fro"))
    off_diagonal_norm = float(np.linalg.norm(off_diagonal, ord="fro"))
    off_diagonal_ratio = float(off_diagonal_norm / frobenius_norm) if frobenius_norm > 0 else 0.0

    return DResEstimate(
        machine_name=machine_name,
        frobenius_norm=frobenius_norm,
        diagonal_norm=diagonal_norm,
        off_diagonal_norm=off_diagonal_norm,
        off_diagonal_ratio=off_diagonal_ratio,
        mean_residual=float(np.mean(residuals)),
        std_residual=float(np.std(residuals)),
    )


def measure_D_res_from_residuals(
    model_dgknet: Any,
    dataset: Sequence[Mapping[str, Any]],
    machine_name: str,
) -> Dict[str, Any]:
    """Estimate D_res norms from a dataset and a predictor.

    The predictor may either be a callable ``model_dgknet(sample) -> pred`` or a
    model object exposing ``predict(sample)``. The dataset must yield mappings
    with ``true``/``target`` and optionally ``features`` entries.
    """

    estimates = []
    residual_norms = []
    nrmse_values = []

    for sample in dataset:
        target = sample.get("target", sample.get("true"))
        if target is None:
            raise KeyError("Dataset sample must include 'target' or 'true'")

        if callable(model_dgknet):
            prediction = model_dgknet(sample)
        elif hasattr(model_dgknet, "predict"):
            prediction = model_dgknet.predict(sample)
        else:
            raise TypeError("model_dgknet must be callable or expose a predict() method")

        estimate = estimate_d_res_from_residuals(np.asarray(target), np.asarray(prediction), machine_name)
        estimates.append(estimate)

        residual = np.asarray(target, dtype=float) - np.asarray(prediction, dtype=float)
        residual_norms.append(float(np.linalg.norm(residual.reshape(-1), ord=2)))
        target_norm = float(np.linalg.norm(np.asarray(target, dtype=float).reshape(-1), ord=2))
        nrmse_values.append(float(np.sqrt(np.mean(np.square(residual))) / target_norm) if target_norm > 0 else float("nan"))

    pearson_r = _safe_pearsonr(residual_norms, nrmse_values)
    return {
        "machine_name": machine_name,
        "n_samples": len(estimates),
        "d_res_estimates": [estimate.to_dict() for estimate in estimates],
        "residual_norms": residual_norms,
        "nrmse_values": nrmse_values,
        "pearson_r_residual_nrmse": pearson_r,
        "mean_frobenius_norm": float(np.mean([estimate.frobenius_norm for estimate in estimates])) if estimates else float("nan"),
        "mean_off_diagonal_ratio": float(np.mean([estimate.off_diagonal_ratio for estimate in estimates])) if estimates else float("nan"),
    }


def build_gate_report(machine_reports: Sequence[Mapping[str, Any]], threshold: float = 0.3) -> Dict[str, Any]:
    """Build a simple phase-1 gate summary from per-machine diagnostics."""

    report = {
        "gate": "GATE_1_D_RES_CORRELATION",
        "threshold": threshold,
        "machines": [],
        "pass_count": 0,
        "fail_count": 0,
    }
    for machine_report in machine_reports:
        machine_name = str(machine_report.get("machine_name", machine_report.get("machine", "unknown")))
        r_value = machine_report.get("pearson_r_residual_nrmse")
        passed = bool(np.isfinite(r_value) and r_value > threshold)
        report["machines"].append({
            "machine_name": machine_name,
            "pearson_r_residual_nrmse": None if r_value is None or not np.isfinite(r_value) else float(r_value),
            "passed": passed,
        })
        report["pass_count"] += int(passed)
        report["fail_count"] += int(not passed)
    report["overall_pass"] = report["fail_count"] == 0 and len(report["machines"]) > 0
    return report


def _load_np_arrays(path: Path) -> Dict[str, np.ndarray]:
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        return {key: data[key] for key in data.files}
    if path.suffix == ".npy":
        return {"array": np.load(path, allow_pickle=False)}
    raise ValueError(f"Unsupported array file: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate phase-1 D_res diagnostics")
    parser.add_argument("--true", type=Path, required=True, help="Path to true trajectory array (.npy/.npz)")
    parser.add_argument("--pred", type=Path, required=True, help="Path to predicted trajectory array (.npy/.npz)")
    parser.add_argument("--machine", required=True, help="Machine name")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    true_arrays = _load_np_arrays(args.true)
    pred_arrays = _load_np_arrays(args.pred)
    true_value = true_arrays.get("array", true_arrays.get("true"))
    pred_value = pred_arrays.get("array", pred_arrays.get("pred"))
    if true_value is None or pred_value is None:
        raise KeyError("Expected arrays named 'array', 'true', or 'pred'")

    estimate = estimate_d_res_from_residuals(true_value, pred_value, args.machine)
    payload = estimate.to_dict()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote diagnostics to {args.out}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
