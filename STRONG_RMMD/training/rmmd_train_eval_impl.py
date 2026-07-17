#!/usr/bin/env python3
"""STRONG-RMMD phase-3 train/eval — compact NI + geometry model.

Compact model design (initial-value surrogate):
  Inputs:  pre-shot context (1280-dim)
           + NI(t=0) profile (40-dim, normalized from state_t0)
           + t=0 geometry (40×66, normalized from state_t0)
           + limiter/vessel geometry (40×66, fixed per machine)
           + machine id
  Output:  autoregressive NI(t) and geometry(t) via unit-step rollout (dt=1).
           Report/eval horizons: 1, 20, 100, 200, 500, 1000.

The t=0 NI and geometry are KNOWN INPUTS (not leakage).  Cross-machine
generalization is encoded via limiter geometry + machine embedding.
Physics constraints (energy conservation, dissipation) are ramped in after
the model first learns to fit the data signal.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
STRONG_RMMD_ROOT = REPO_ROOT / "STRONG_RMMD"
sys.path.insert(0, str(STRONG_RMMD_ROOT))
sys.path.insert(0, str(STRONG_RMMD_ROOT / "data_io"))

from strong_rmmd.losses import RMMDLossFunction
from strong_rmmd.resonance_frequencies import compute_resonance_frequencies
from strong_rmmd.multi_machine_rmmd import MultiMachineRMMD
from data_io.dataset_loader import Phase0DatasetView, is_sharded_payload, load_phase0_payload

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rmmd_train_eval")

# Only NI is the trained profile.
PROFILE_ORDER = ["NI"]
# Report horizons for eval JSON + sparse y_seq checkpoints in the dataset.
# Training uses UNIT-STEP rollout (dt=1) with loss at every step 1..T_frontier,
# comparing against dense ni_traj / geom_traj (not sparse checkpoint jumps).
DIRECT_COMPACT_HORIZONS: tuple[int, ...] = (1, 20, 100, 200, 500, 1000)
# Extra diagnostic horizons recorded for near-term visibility, unioned with the report horizons.
DIAGNOSTIC_HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 20, 32, 50, 75, 100, 150, 200, 300, 500, 750, 1000)
_REPORT_HORIZONS: tuple[int, ...] = tuple(sorted(set(DIRECT_COMPACT_HORIZONS) | set(DIAGNOSTIC_HORIZONS)))
COMPACT_PRE_SHOT_CONTEXT_DIM = 1280
# Number of exogenous actuator channels (must match the builder and the model's n_drivers). Drivers are
# known controls (NBI/ohmic power, plasma current, gas), not state diagnostics.
N_DRIVERS = 8


# ============================================================================
# CURRICULUM SCHEDULER
# ============================================================================

# Training curriculum frontier (decoupled from the report horizons): the autoregressive rollout grows its
# supervised horizon gradually along a geometric ladder so the model learns to consume its own predictions.
TRAINING_FRONTIERS: tuple[int, ...] = (
    1, 2, 3, 5, 8, 12, 20, 32, 50, 75, 100, 150, 200, 300, 500, 750, 1000,
)


class CurriculumScheduler:
    """Curriculum over the farthest supervised TIME step (unit steps, dt=1).

    Walks TRAINING_FRONTIERS gradually. `current_frontier` is a physical time
    value (1, 2, 3, …). Rollout runs steps 1..frontier with loss at every step.
    REPORT horizons (DIRECT_COMPACT_HORIZONS) stay fixed for eval/logging.
    """

    def __init__(self, frontiers: Sequence[int] = TRAINING_FRONTIERS, gamma_init: float = 0.98):
        self.frontiers = [int(f) for f in frontiers]
        self.gamma_init = gamma_init
        self.idx = 0
        self.epochs_at_frontier = 0

    def get_gamma(self, epoch: int, total_epochs: int) -> float:
        if total_epochs <= 1:
            return self.gamma_init
        return self.gamma_init + (1.0 - self.gamma_init) * (epoch / total_epochs)

    @property
    def current_frontier(self) -> int:
        return self.frontiers[self.idx]

    @property
    def at_last(self) -> bool:
        return self.idx >= len(self.frontiers) - 1

    def advance(self) -> bool:
        """Move to the next (larger) frontier. Returns True if it actually advanced."""
        if not self.at_last:
            self.idx += 1
            self.epochs_at_frontier = 0
            return True
        return False


# ============================================================================
# NORMALIZATION UTILITIES
# ============================================================================

def _iter_sample_states(payload: Dict):
    """Yield raw state dicts (state_t0 and state_trajectory entries) from a payload."""
    view = Phase0DatasetView(payload)
    for idx in range(len(view)):
        sample = view.get_sample(idx)
        state_t0 = sample.get("state_t0", {})
        if state_t0:
            yield state_t0
        for state in sample.get("state_trajectory", []) or []:
            yield state


def _collect_normalization_stats(payload: Dict) -> Dict[str, Dict[str, float]]:
    """Compute mean/std normalization stats from raw state dicts in a payload."""
    stats: Dict[str, List[np.ndarray]] = defaultdict(list)

    def collect(prefix: str, value):
        if value is None:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                collect(f"{prefix}.{key}", item)
            return
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size:
            stats[prefix].append(array)

    for state in _iter_sample_states(payload):
        collect("kinetic_profiles", state.get("kinetic_profiles", {}))
        collect("geometry_tensor", state.get("geometry_tensor"))

    normalization_stats: Dict[str, Dict[str, float]] = {}
    for key, values in stats.items():
        flat = np.concatenate(values)
        flat = np.nan_to_num(flat, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        if flat.size == 0:
            continue
        mean_val = float(np.mean(flat))
        std_val = float(np.std(flat))
        if not np.isfinite(std_val) or std_val < 1e-8:
            std_val = 1.0
        normalization_stats[key] = {"mean": mean_val, "std": std_val}
    return normalization_stats


def _normalize_array(value, stats: Dict[str, Dict[str, float]], key: str):
    if value is None or key not in stats:
        return value
    arr = np.asarray(value, dtype=np.float32)
    entry = stats[key]
    return (arr - entry["mean"]) / entry["std"]


def _extract_per_element_stats_from_metadata(payload: Dict) -> Dict[str, Dict[str, Any]]:
    """Extract per-element (per-radial-point) normalization stats from the payload metadata.

    The data_build per-machine compact payloads store y_stats and geom_stats in metadata.
    These are the SAME stats used to normalize y_seq and geom_seq during dataset build,
    so using them for ni_t0/geom_t0 normalization gives a consistent representation.
    Returns empty dict if the payload has no usable per-element stats.
    """
    meta = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        return {}

    result: Dict[str, Dict[str, Any]] = {}

    y_stats = meta.get("y_stats")
    if isinstance(y_stats, dict):
        mean_v = y_stats.get("mean")
        std_v = y_stats.get("std")
        if mean_v is not None and std_v is not None:
            mean_arr = np.asarray(mean_v, dtype=np.float32).reshape(-1)
            std_arr = np.asarray(std_v, dtype=np.float32).reshape(-1)
            # Guard: only use if shape matches expected 40 radial points
            if mean_arr.shape == (40,) and std_arr.shape == (40,):
                result["kinetic_profiles.NI"] = {
                    "mean": float(np.mean(mean_arr)),
                    "std": float(np.mean(std_arr)),
                    "mean_per_element": mean_arr.tolist(),
                    "std_per_element": std_arr.tolist(),
                }
                logger.info(
                    "Extracted per-element NI stats from metadata: mean[0]=%.4g std[0]=%.4g",
                    mean_arr[0], std_arr[0],
                )
            else:
                # Scalar-form stats stored in metadata
                result["kinetic_profiles.NI"] = {
                    "mean": float(np.mean(mean_arr)),
                    "std": float(np.mean(std_arr)),
                }

    geom_stats = meta.get("geom_stats")
    if isinstance(geom_stats, dict):
        mean_v = geom_stats.get("mean")
        std_v = geom_stats.get("std")
        if mean_v is not None and std_v is not None:
            mean_arr = np.asarray(mean_v, dtype=np.float32).reshape(-1)
            std_arr = np.asarray(std_v, dtype=np.float32).reshape(-1)
            result["geometry_tensor"] = {
                "mean": float(np.mean(mean_arr)),
                "std": float(np.mean(std_arr)),
            }
            if mean_arr.size == 40 * 66:
                result["geometry_tensor"]["mean_per_element"] = mean_arr.tolist()
                result["geometry_tensor"]["std_per_element"] = std_arr.tolist()

    return result


def _ensure_normalization_stats(
    payload_path: Path,
    checkpoint_dir: Path | None = None,
    require: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Return normalization stats, checking sources in priority order:
       1. existing checkpoint (so eval matches training exactly)
       2. pre-computed sidecar JSON
       3. per-element stats from payload metadata (data_build per-machine payloads)
       4. stats embedded directly in the payload dict (older compact datasets)
       5. computed on the fly from state_t0 frames (scalar, fallback only)
    """
    # 1. Checkpoint
    if checkpoint_dir is not None:
        ckpt_best = checkpoint_dir / "checkpoint_best.pt"
        plain_best, compressed_best = _checkpoint_paths(ckpt_best)
        if plain_best.exists() or compressed_best.exists():
            try:
                ck = _torch_load_checkpoint_any(ckpt_best, map_location="cpu")
                if isinstance(ck, dict) and "normalization_stats" in ck and ck["normalization_stats"]:
                    logger.info("Normalization stats from checkpoint: %s", compressed_best if compressed_best.exists() else plain_best)
                    return ck["normalization_stats"]
            except Exception as exc:
                logger.warning("Could not read normalization from checkpoint: %s", exc)

    # 2. Sidecar JSON
    sidecar = payload_path.with_suffix(payload_path.suffix + ".norm.json")
    if sidecar.exists():
        try:
            with sidecar.open("r", encoding="utf-8") as f:
                s = json.load(f)
            if isinstance(s, dict) and s and "kinetic_profiles.NI" in s:
                logger.info("Normalization stats from sidecar: %s", sidecar)
                return s
        except Exception as exc:
            logger.warning("Could not read normalization sidecar: %s", exc)

    if require:
        raise ValueError(f"Normalization stats required but not found at checkpoint_dir={checkpoint_dir} or sidecar={sidecar}")

    # 3-5. Load payload once for all remaining checks
    payload = _load_phase0_dataset(payload_path)

    # 3. Per-element stats from metadata (data_build compact per-machine payloads)
    meta_stats = _extract_per_element_stats_from_metadata(payload)
    if meta_stats:
        logger.info("Using per-element normalization stats from payload metadata")
        stats = meta_stats
    # 4. Embedded normalization_stats key (older compact datasets)
    elif isinstance(payload.get("normalization_stats"), dict) and payload["normalization_stats"].get("kinetic_profiles.NI"):
        logger.info("Normalization stats embedded in payload: %s", payload_path)
        stats = payload["normalization_stats"]
    # 5. Compute from state frames (scalar fallback — NOTE: may not match y_seq normalization
    #    for combined multi-machine datasets. Use per-machine payloads for best results.)
    else:
        logger.warning(
            "No per-element normalization stats in payload metadata. "
            "Falling back to scalar stats from state_t0 frames. "
            "For best accuracy, use per-machine compact payloads (which store y_stats in metadata)."
        )
        stats = _collect_normalization_stats(payload)

    sidecar.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sidecar.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved normalization stats to sidecar: %s", sidecar)
    except Exception as exc:
        logger.warning("Could not save normalization sidecar: %s", exc)
    return stats


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def _load_compatible_state_dict(model: nn.Module, checkpoint_state: Dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            skipped.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped.append(key)
            continue
        compatible_state[key] = value
    missing = [key for key in model_state.keys() if key not in compatible_state]
    if compatible_state:
        model.load_state_dict(compatible_state, strict=False)
    logger.info(
        "Loaded checkpoint tensors: %d matched, %d skipped, %d missing",
        len(compatible_state), len(skipped), len(missing),
    )
    if skipped:
        logger.info("  skipped (first 8): %s", skipped[:8])


def _checkpoint_safe_value(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.is_floating_point() and tensor.dtype != torch.float16:
            tensor = tensor.to(torch.float16)
        return tensor
    if isinstance(value, dict):
        return {k: _checkpoint_safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_checkpoint_safe_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_safe_value(v) for v in value)
    return value


def _checkpoint_paths(base_path: Path) -> tuple[Path, Path]:
    if str(base_path).endswith(".pt.gz"):
        compressed = base_path
        plain = Path(str(base_path)[:-3])
    elif base_path.suffix == ".pt":
        plain = base_path
        compressed = Path(str(base_path) + ".gz")
    else:
        plain = base_path
        compressed = Path(str(base_path) + ".gz")
    return plain, compressed


def _torch_save_checkpoint(checkpoint_obj: Dict[str, object], base_path: Path) -> Path:
    _, compressed_path = _checkpoint_paths(base_path)
    compressed_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(compressed_path, "wb") as handle:
        torch.save(checkpoint_obj, handle)
    return compressed_path


def _torch_load_checkpoint_any(path: Path, map_location: str = "cpu"):
    plain_path, compressed_path = _checkpoint_paths(path)
    if plain_path.exists():
        return torch.load(str(plain_path), map_location=map_location, weights_only=True)
    if compressed_path.exists():
        with gzip.open(compressed_path, "rb") as handle:
            return torch.load(handle, map_location=map_location, weights_only=True)
    raise FileNotFoundError(f"Checkpoint not found at {plain_path} or {compressed_path}")


# ============================================================================
# MODEL / DATA UTILITIES
# ============================================================================

def _set_rmmd_trainable_stage(model: nn.Module, train_all: bool) -> None:
    rmmd_keywords = ("rmmd.", "rmmd_geom.", "s_universal", "delta_s_machines")
    for name, param in model.named_parameters():
        if train_all:
            param.requires_grad = True
        else:
            param.requires_grad = any(kw in name for kw in rmmd_keywords)


def _extract_profiles_from_state(state: Dict) -> torch.Tensor:
    """Extract NI profile from a state dict, padded to 40 points."""
    kinetic = state.get("kinetic_profiles", {})
    val = kinetic.get("NI") if isinstance(kinetic, dict) else None
    if val is None:
        return torch.zeros(40, dtype=torch.float32)
    arr = np.asarray(val, dtype=np.float32).reshape(-1)
    padded = np.zeros(40, dtype=np.float32)
    padded[: min(len(arr), 40)] = arr[:40]
    return torch.tensor(padded, dtype=torch.float32)


def _extract_geometry_matrix(state: Dict, key: str, n_psi: int = 40, n_fourier: int = 66) -> np.ndarray:
    """Extract a (n_psi, n_fourier) geometry matrix from a state dict."""
    g = state.get(key)
    if g is None:
        return np.zeros((n_psi, n_fourier), dtype=np.float32)
    arr = np.asarray(g, dtype=np.float32)
    if arr.ndim == 1:
        flat = np.zeros(n_psi * n_fourier, dtype=np.float32)
        flat[: min(arr.size, flat.size)] = arr[: flat.size]
        arr = flat.reshape(n_psi, n_fourier)
    out = np.zeros((n_psi, n_fourier), dtype=np.float32)
    rows = min(arr.shape[0], n_psi)
    cols = min(arr.shape[1], n_fourier)
    out[:rows, :cols] = arr[:rows, :cols]
    return out


def _denormalize_ni_batch(batch_40: torch.Tensor, normalization_stats: Dict[str, Dict[str, float]]) -> torch.Tensor:
    """Denormalize a (B, 40) normalized NI batch to physical space."""
    if batch_40.ndim == 1:
        batch_40 = batch_40.unsqueeze(0)
    out = batch_40.clone()
    entry = normalization_stats.get("kinetic_profiles.NI") if normalization_stats else None
    if entry is not None:
        mean_val = float(entry.get("mean", 0.0))
        std_val = float(entry.get("std", 1.0))
        out = out * std_val + mean_val
    return out


def _denormalize_geometry_batch(batch_geom: torch.Tensor, normalization_stats: Dict[str, Dict[str, float]]) -> torch.Tensor:
    """Denormalize a (B, 40, 66) normalized geometry batch to physical space."""
    if batch_geom.ndim == 2:
        batch_geom = batch_geom.unsqueeze(0)
    out = batch_geom.clone()
    entry = normalization_stats.get("geometry_tensor") if normalization_stats else None
    if entry is not None:
        mean_val = float(entry.get("mean", 0.0))
        std_val = float(entry.get("std", 1.0))
        out = out * std_val + mean_val
    return out


def _compute_omegas_for_compact_batch(
    ni_t0: torch.Tensor,                # (N, 40) normalized NI, on CPU
    pre_shot_scalars_list: List[Dict],  # N dicts with raw physical scalars (PINJ, BTDIA, ...)
    machine_names: Sequence[str],
    device: str,
    normalization_stats: Dict[str, Dict[str, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute omega_t and omega_d from t=0 NI and pre-shot physical scalars."""
    ni_denorm = _denormalize_ni_batch(ni_t0.detach().cpu(), normalization_stats)  # (N, 40) physical
    t0 = time.time()
    omega_t_vals: List[float] = []
    omega_d_vals: List[float] = []
    for b, machine in enumerate(machine_names):
        scalars = pre_shot_scalars_list[b] if b < len(pre_shot_scalars_list) else {}
        profile_map = {"NI": ni_denorm[b].numpy()}
        omega = compute_resonance_frequencies(profile_map, str(machine), scalars)
        omega_t_vals.append(float(omega.get("omega_t", 0.0)))
        omega_d_vals.append(float(omega.get("omega_d", 1.0)))
    logger.debug("_compute_omegas_for_compact_batch: N=%d in %.3fs", len(machine_names), time.time() - t0)
    return (
        torch.tensor(omega_t_vals, dtype=torch.float32, device=device),
        torch.tensor(omega_d_vals, dtype=torch.float32, device=device),
    )


def _normalized_rmse_mae(pred: np.ndarray, target: np.ndarray):
    pred_flat = pred.astype(np.float64).reshape(-1)
    target_flat = target.astype(np.float64).reshape(-1)
    diff = pred_flat - target_flat
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    # In normalized space profiles are ~unit std; floor avoids divide-by-near-zero blowups.
    target_rms = float(np.sqrt(np.mean(target_flat ** 2)))
    target_rms = max(target_rms, 0.05)
    return rmse / target_rms, mae / target_rms


def _ni_nrmse_tensor(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batch NRMSE in normalized space: rmse / target_rms per sample, then mean."""
    diff = (pred - target).reshape(pred.shape[0], -1)
    rmse_i = torch.sqrt(torch.mean(diff ** 2, dim=1) + 1e-12)
    target_rms = torch.sqrt(torch.mean(target.reshape(target.shape[0], -1) ** 2, dim=1) + 1e-12)
    return rmse_i / target_rms.clamp_min(0.05)


def _horizon_time_for_curriculum(horizon_idx: int) -> int:
    idx = min(int(horizon_idx), len(DIRECT_COMPACT_HORIZONS)) - 1
    return int(DIRECT_COMPACT_HORIZONS[max(idx, 0)])


def _coerce_ni_geom_traj(
    ni_traj: torch.Tensor,
    geom_traj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure dense trajectories are (T, 40) and (T, 40, 66). Fixes legacy/wrong layouts."""
    ni = torch.as_tensor(ni_traj, dtype=torch.float32)
    geom = torch.as_tensor(geom_traj, dtype=torch.float32)
    if ni.ndim == 1:
        ni = ni.reshape(1, -1) if ni.numel() == 40 else ni.unsqueeze(0)
    elif ni.ndim == 2 and ni.shape[0] == 40 and ni.shape[1] != 40:
        ni = ni.transpose(0, 1).contiguous()
    if ni.ndim == 2 and ni.shape[1] != 40:
        if ni.shape[0] == 40:
            ni = ni.transpose(0, 1).contiguous()
        else:
            ni = ni.reshape(ni.shape[0], -1)[:, :40]

    if geom.ndim == 2 and geom.shape == (40, 66):
        geom = geom.unsqueeze(0)
    elif geom.ndim == 3 and geom.shape[1] == 40 and geom.shape[2] == 66:
        pass
    elif geom.numel() > 0:
        geom = geom.reshape(-1, 40, 66)

    tlen = int(ni.shape[0]) if ni.ndim == 2 and ni.shape[1] == 40 else 0
    if tlen <= 0:
        return torch.zeros(0, 40, dtype=torch.float32), torch.zeros(0, 40, 66, dtype=torch.float32)

    if geom.ndim != 3 or int(geom.shape[0]) != tlen:
        geom = torch.zeros(tlen, 40, 66, dtype=torch.float32)
    return ni[:tlen].contiguous(), geom[:tlen].contiguous()


def _dense_traj_from_state_trajectory(
    state_trajectory: Sequence[Dict],
    normalize_ni,
    normalize_geom,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized dense trajectories when legacy payloads lack ni_traj."""
    ni_rows: List[np.ndarray] = []
    geom_rows: List[np.ndarray] = []
    for state in state_trajectory or []:
        if not isinstance(state, dict):
            continue
        kinetic = state.get("kinetic_profiles", {}) or {}
        ni = kinetic.get("NI")
        geom = state.get("geometry_tensor")
        if ni is not None:
            arr = np.asarray(ni, dtype=np.float32).reshape(-1)
            padded = np.zeros(40, dtype=np.float32)
            padded[: min(len(arr), 40)] = arr[:40]
            ni_rows.append(normalize_ni(padded))
        else:
            ni_rows.append(np.zeros(40, dtype=np.float32))
        if geom is not None:
            g = np.asarray(geom, dtype=np.float32)
            if g.ndim == 1:
                g = g.reshape(40, 66) if g.size >= 40 * 66 else np.zeros((40, 66), dtype=np.float32)
            out = np.zeros((40, 66), dtype=np.float32)
            r, c = min(g.shape[0], 40), min(g.shape[1], 66)
            out[:r, :c] = g[:r, :c]
            geom_rows.append(normalize_geom(out))
        else:
            geom_rows.append(np.zeros((40, 66), dtype=np.float32))
    if not ni_rows:
        return torch.zeros(0, 40), torch.zeros(0, 40, 66)
    return torch.from_numpy(np.stack(ni_rows)), torch.from_numpy(np.stack(geom_rows))


# ============================================================================
# DATASET
# ============================================================================

class CompactRolloutDataset(Dataset):
    """One sample per shot: t=0 IC + pre-shot context + NI/geom targets at checkpoints.

    y_seq[c] and geom_seq[c] align with DIRECT_COMPACT_HORIZONS[c] (T=1, 10, 20, …).
    `horizon` = how many leading checkpoints are exposed (curriculum frontier).
    """

    def __init__(
        self,
        payload: Dict,
        max_time: int,
        normalization_stats: Dict[str, Dict[str, float]] | None = None,
    ):
        self.view = Phase0DatasetView(payload)
        self.max_time = int(max_time)
        self.normalization_stats = normalization_stats or {}
        self.shot_indices: List[int] = []

        skipped = 0
        for idx in range(len(self.view)):
            sample = self.view.get_sample(idx)
            ni_seq = sample.get("y_seq")
            geom_seq = sample.get("geom_seq")
            target_mask = sample.get("target_mask")
            if ni_seq is None or geom_seq is None:
                skipped += 1
                continue
            if len(ni_seq) < 1 or len(geom_seq) < 1:
                skipped += 1
                continue
            if target_mask is not None and not np.asarray(target_mask, dtype=np.bool_).any():
                skipped += 1
                continue
            self.shot_indices.append(idx)

        if not self.shot_indices:
            raise ValueError(f"No valid compact shots found for max_time={max_time}")
        logger.info(
            "CompactRolloutDataset: max_time=%d shots=%d skipped=%d",
            self.max_time, len(self.shot_indices), skipped,
        )

    def _normalize_ni_arr(self, arr: np.ndarray) -> np.ndarray:
        """Normalize NI profile.

        Uses per-element (per-radial-point) stats when available in the payload
        metadata — these are the SAME stats used to normalize y_seq targets during
        dataset build, giving consistent input/target scaling. Falls back to scalar
        stats for combined multi-machine payloads (acceptable approximation).
        """
        entry = self.normalization_stats.get("kinetic_profiles.NI")
        if entry is None:
            return arr.astype(np.float32)
        mean_pe = entry.get("mean_per_element")
        std_pe = entry.get("std_per_element")
        if mean_pe is not None and std_pe is not None:
            mean_arr = np.asarray(mean_pe, dtype=np.float64)
            std_arr = np.clip(np.abs(np.asarray(std_pe, dtype=np.float64)), 1e-8, None)
            if mean_arr.shape == arr.shape and std_arr.shape == arr.shape:
                return ((arr.astype(np.float64) - mean_arr) / std_arr).astype(np.float32)
        # Fallback: scalar normalization
        mean_val = float(entry.get("mean", 0.0))
        std_val = max(abs(float(entry.get("std", 1.0))), 1e-8)
        return ((arr.astype(np.float64) - mean_val) / std_val).astype(np.float32)

    def _normalize_geom_arr(self, arr: np.ndarray) -> np.ndarray:
        """Normalize geometry tensor. Uses per-element stats when available."""
        entry = self.normalization_stats.get("geometry_tensor")
        if entry is None:
            return arr.astype(np.float32)
        mean_pe = entry.get("mean_per_element")
        std_pe = entry.get("std_per_element")
        if mean_pe is not None and std_pe is not None:
            mean_arr = np.asarray(mean_pe, dtype=np.float64).reshape(40, 66)
            std_arr = np.clip(np.abs(np.asarray(std_pe, dtype=np.float64).reshape(40, 66)), 1e-8, None)
            if mean_arr.shape == arr.shape and std_arr.shape == arr.shape:
                return ((arr.astype(np.float64) - mean_arr) / std_arr).astype(np.float32)
        # Fallback: scalar normalization
        mean_val = float(entry.get("mean", 0.0))
        std_val = max(abs(float(entry.get("std", 1.0))), 1e-8)
        return ((arr.astype(np.float64) - mean_val) / std_val).astype(np.float32)

    def __len__(self) -> int:
        return len(self.shot_indices)

    def __getitem__(self, idx: int) -> Dict:
        shot_idx = self.shot_indices[idx]
        sample = self.view.get_sample(shot_idx)

        ni_seq = torch.as_tensor(sample.get("y_seq"), dtype=torch.float32)
        geom_seq = torch.as_tensor(sample.get("geom_seq"), dtype=torch.float32)
        target_steps = torch.as_tensor(sample.get("target_steps"), dtype=torch.int64)
        target_mask_arr = sample.get("target_mask")

        if target_mask_arr is not None:
            target_mask = torch.as_tensor(target_mask_arr, dtype=torch.bool)
        else:
            target_mask = torch.ones(ni_seq.shape[0], dtype=torch.bool)

        max_time = self.max_time

        # Sparse report targets (for eval alignment); keep only steps <= curriculum frontier.
        keep = target_steps <= max_time
        if keep.any():
            ni_seq = ni_seq[keep].contiguous()
            geom_seq = geom_seq[keep].contiguous()
            target_steps = target_steps[keep].contiguous()
            target_mask = target_mask[keep].contiguous()
        else:
            ni_seq = ni_seq[:0]
            geom_seq = geom_seq[:0]
            target_steps = target_steps[:0]
            target_mask = target_mask[:0]

        machine = sample.get("machine", "UNKNOWN")
        if isinstance(machine, bytes):
            machine = machine.decode("utf-8")
        elif isinstance(machine, torch.Tensor):
            machine = str(machine.item())
        else:
            machine = str(machine)

        pre_shot_context = torch.as_tensor(
            sample.get("pre_shot_context", np.zeros(COMPACT_PRE_SHOT_CONTEXT_DIM, dtype=np.float32)),
            dtype=torch.float32,
        )
        limiter_geometry = torch.as_tensor(
            sample.get("limiter_geometry_tensor", np.zeros((40, 66), dtype=np.float32)),
            dtype=torch.float32,
        )

        # --- ni_t0: prefer pre-stored normalized value (from updated dataset builder) ---
        # Fall back to extracting from state_t0 and normalizing for old payloads.
        ni_t0_stored = sample.get("ni_t0")
        if ni_t0_stored is not None:
            ni_t0 = torch.as_tensor(ni_t0_stored, dtype=torch.float32).reshape(40)
        else:
            state_t0 = sample.get("state_t0") or {}
            ni_t0_raw = None
            if isinstance(state_t0, dict):
                kinetic = state_t0.get("kinetic_profiles", {})
                if isinstance(kinetic, dict):
                    ni_t0_raw = kinetic.get("NI")
            if ni_t0_raw is not None:
                arr = np.asarray(ni_t0_raw, dtype=np.float32).reshape(-1)
                padded = np.zeros(40, dtype=np.float32)
                padded[: min(len(arr), 40)] = arr[:40]
                ni_t0 = torch.from_numpy(self._normalize_ni_arr(padded))
            else:
                ni_t0 = torch.zeros(40, dtype=torch.float32)

        # --- geom_t0: prefer pre-stored normalized value ---
        geom_t0_stored = sample.get("geom_t0")
        if geom_t0_stored is not None:
            geom_t0 = torch.as_tensor(geom_t0_stored, dtype=torch.float32).reshape(40, 66)
        else:
            state_t0 = sample.get("state_t0") or {}
            geom_t0_raw = state_t0.get("geometry_tensor") if isinstance(state_t0, dict) else None
            if geom_t0_raw is not None:
                arr = np.asarray(geom_t0_raw, dtype=np.float32)
                if arr.ndim == 1:
                    flat_arr = np.zeros(40 * 66, dtype=np.float32)
                    flat_arr[: min(arr.size, flat_arr.size)] = arr[: flat_arr.size]
                    arr = flat_arr.reshape(40, 66)
                out_arr = np.zeros((40, 66), dtype=np.float32)
                r, c = min(arr.shape[0], 40), min(arr.shape[1], 66)
                out_arr[:r, :c] = arr[:r, :c]
                geom_t0 = torch.from_numpy(self._normalize_geom_arr(out_arr))
            else:
                geom_t0 = torch.zeros((40, 66), dtype=torch.float32)

        # --- Dense per-timestep targets for unit-step rollout (required) ---
        ni_traj_stored = sample.get("ni_traj")
        geom_traj_stored = sample.get("geom_traj")
        if ni_traj_stored is not None and np.asarray(ni_traj_stored).size > 0:
            ni_traj, geom_traj = _coerce_ni_geom_traj(
                torch.as_tensor(ni_traj_stored, dtype=torch.float32),
                torch.as_tensor(
                    geom_traj_stored if geom_traj_stored is not None else np.zeros((0, 40, 66)),
                    dtype=torch.float32,
                ),
            )
        else:
            state_traj = sample.get("state_trajectory") or []
            ni_traj, geom_traj = _dense_traj_from_state_trajectory(
                state_traj,
                self._normalize_ni_arr,
                self._normalize_geom_arr,
            )

        raw_traj_len = int(ni_traj.shape[0])
        if raw_traj_len > 0:
            use_t = min(max_time, raw_traj_len)
            ni_traj = ni_traj[:use_t].contiguous()
            geom_traj = geom_traj[:use_t].contiguous()
        else:
            ni_traj = torch.zeros(0, 40, dtype=torch.float32)
            geom_traj = torch.zeros(0, 40, 66, dtype=torch.float32)
        # Effective length AFTER curriculum slice (must match collated ni_traj rows).
        traj_len = int(ni_traj.shape[0])

        # Time-resolved exogenous drivers [T, N_DRIVERS], sliced to the ni_traj horizon; zero-filled when absent.
        drivers_stored = sample.get("drivers_traj")
        if drivers_stored is not None and np.asarray(drivers_stored).size > 0:
            drivers_traj = torch.as_tensor(np.asarray(drivers_stored), dtype=torch.float32)
            if drivers_traj.ndim == 1:
                drivers_traj = drivers_traj.unsqueeze(-1)
            # match channel count to N_DRIVERS (pad/trim last dim)
            if drivers_traj.shape[-1] < N_DRIVERS:
                pad = N_DRIVERS - drivers_traj.shape[-1]
                drivers_traj = torch.cat([drivers_traj, torch.zeros(drivers_traj.shape[0], pad)], dim=-1)
            elif drivers_traj.shape[-1] > N_DRIVERS:
                drivers_traj = drivers_traj[:, :N_DRIVERS]
            drivers_traj = drivers_traj[:traj_len].contiguous() if traj_len > 0 else torch.zeros(0, N_DRIVERS)
            # pad rows up to traj_len if the stored series is short
            if drivers_traj.shape[0] < traj_len:
                last = drivers_traj[-1:] if drivers_traj.shape[0] > 0 else torch.zeros(1, N_DRIVERS)
                drivers_traj = torch.cat([drivers_traj, last.repeat(traj_len - drivers_traj.shape[0], 1)], dim=0)
        else:
            drivers_traj = torch.zeros(traj_len, N_DRIVERS, dtype=torch.float32)

        # --- Raw pre-shot physical scalars for omega computation ---
        pre_shot_scalars: Dict = {}
        state_t0 = sample.get("state_t0") or {}
        if isinstance(state_t0, dict):
            psi = state_t0.get("pre_shot_inputs", {})
            if isinstance(psi, dict):
                pre_shot_scalars = dict(psi.get("scalars", {}) or {})

        return {
            "y_seq": ni_seq,
            "target_steps": target_steps,
            "target_mask": target_mask,
            "machine": machine,
            "shot_idx": shot_idx,
            "geom_seq": geom_seq,
            "limiter_geometry_tensor": limiter_geometry,
            "pre_shot_context": pre_shot_context,
            "compact_mode": True,
            "ni_t0": ni_t0,
            "geom_t0": geom_t0,
            "ni_traj": ni_traj,
            "geom_traj": geom_traj,
            "drivers_traj": drivers_traj,
            "traj_len": traj_len,
            "max_time": max_time,
            "pre_shot_scalars": pre_shot_scalars,
        }

    def get_sample(self, idx: int) -> Dict:
        return self[idx]


def _compact_rollout_collate(batch: List[Dict]) -> Dict:
    if not batch:
        return {}

    max_targets = max(int(item["y_seq"].shape[0]) for item in batch)
    max_traj = max(int(item["ni_traj"].shape[0]) for item in batch)
    batch_size = len(batch)
    device_dtype = batch[0]["y_seq"].dtype

    pre_shot_context = torch.stack([item["pre_shot_context"] for item in batch], dim=0)
    limiter_geometry_tensor = torch.stack([item["limiter_geometry_tensor"] for item in batch], dim=0)
    ni_t0 = torch.stack([item["ni_t0"] for item in batch], dim=0)
    geom_t0 = torch.stack([item["geom_t0"] for item in batch], dim=0)

    y_seq = torch.zeros(batch_size, max_targets, batch[0]["y_seq"].shape[-1], dtype=device_dtype)
    geom_seq = torch.zeros(
        batch_size, max_targets,
        batch[0]["geom_seq"].shape[-2], batch[0]["geom_seq"].shape[-1],
        dtype=batch[0]["geom_seq"].dtype,
    )
    target_steps = torch.zeros(batch_size, max_targets, dtype=batch[0]["target_steps"].dtype)
    target_mask = torch.zeros(batch_size, max_targets, dtype=torch.bool)

    ni_traj = torch.zeros(batch_size, max_traj, 40, dtype=device_dtype)
    geom_traj = torch.zeros(
        batch_size, max_traj, batch[0]["geom_traj"].shape[-2], batch[0]["geom_traj"].shape[-1],
        dtype=batch[0]["geom_traj"].dtype,
    )
    drivers_traj = torch.zeros(batch_size, max_traj, N_DRIVERS, dtype=device_dtype)
    traj_len = torch.zeros(batch_size, dtype=torch.int64)
    max_time = torch.zeros(batch_size, dtype=torch.int64)

    machine: List[str] = []
    shot_idx: List[int] = []
    pre_shot_scalars_list: List[Dict] = []
    for row, item in enumerate(batch):
        length = int(item["y_seq"].shape[0])
        if length > 0:
            y_seq[row, :length] = item["y_seq"]
            geom_seq[row, :length] = item["geom_seq"]
            target_steps[row, :length] = item["target_steps"]
            target_mask[row, :length] = item["target_mask"]
        tlen = int(item["ni_traj"].shape[0])
        if tlen > 0:
            ni_traj[row, :tlen] = item["ni_traj"]
            geom_traj[row, :tlen] = item["geom_traj"]
            dtraj = item.get("drivers_traj")
            if isinstance(dtraj, torch.Tensor) and dtraj.shape[0] >= tlen and dtraj.shape[-1] == N_DRIVERS:
                drivers_traj[row, :tlen] = dtraj[:tlen]
        traj_len[row] = tlen
        max_time[row] = int(item.get("max_time", tlen))
        machine.append(item["machine"])
        shot_idx.append(int(item["shot_idx"]))
        pre_shot_scalars_list.append(item.get("pre_shot_scalars", {}))

    return {
        "y_seq": y_seq,
        "target_steps": target_steps,
        "target_mask": target_mask,
        "machine": machine,
        "shot_idx": torch.as_tensor(shot_idx, dtype=torch.int64),
        "geom_seq": geom_seq,
        "limiter_geometry_tensor": limiter_geometry_tensor,
        "pre_shot_context": pre_shot_context,
        "compact_mode": True,
        "ni_t0": ni_t0,
        "geom_t0": geom_t0,
        "ni_traj": ni_traj,
        "geom_traj": geom_traj,
        "drivers_traj": drivers_traj,
        "traj_len": traj_len,
        "max_time": max_time,
        "pre_shot_scalars": pre_shot_scalars_list,
    }


def _validate_unit_step_dataset(payload: Dict, normalization_stats: Dict) -> None:
    """Fail fast if compact payloads lack dense ni_traj (causes all-zero loss/NRMSE)."""
    probe = CompactRolloutDataset(payload, max_time=1, normalization_stats=normalization_stats)
    if len(probe) < 1:
        raise ValueError("CompactRolloutDataset has zero shots — check dataset path and filters.")

    ok = 0
    checked = min(32, len(probe))
    max_t = 0
    for i in range(checked):
        sample = probe[i]
        tlen = int(sample["ni_traj"].shape[0])
        max_t = max(max_t, tlen)
        if tlen >= 1:
            ok += 1

    if ok == 0:
        raw_lens = []
        for i in range(checked):
            s = probe.view.get_sample(probe.shot_indices[i])
            raw = s.get("ni_traj")
            raw_lens.append(int(np.asarray(raw).shape[0]) if raw is not None else -1)
        raise RuntimeError(
            "Unit-step training requires ni_traj in every compact sample, but none of the "
            f"first {checked} shots have ni_traj length >= 1 after loading. "
            f"Raw ni_traj lengths in payload (first {checked}): {raw_lens}. "
            "Rebuild with STRONG_RMMD/data_build/build_phase0new_from_cdfs.py and use the NEW "
            f"dataset_*_compact.pt (per-machine or combined/) with horizons {DIRECT_COMPACT_HORIZONS}. "
            "If you rebuilt per-machine only, point --compact-train-data at "
            "combined/dataset_train_compact.pt or a single machine's dataset_train_compact.pt."
        )
    logger.info(
        "Dataset OK: %d/%d probe shots have ni_traj (max traj length in probe=%d)",
        ok, checked, max_t,
    )


def _normalized_step_dt(dt: float) -> float:
    """Normalize a checkpoint interval (in time-steps) to ~[0, 1] for dt-conditioning."""
    return float(np.log1p(max(dt, 0.0)) / np.log1p(1000.0))


def _select_batch_rows(
    tensor: torch.Tensor | None,
    idx: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor | None:
    """Index leading batch dimension when a subset of shots is valid at this time step."""
    if tensor is None or idx is None:
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.dim() > 0 and int(tensor.shape[0]) == batch_size:
        return tensor.index_select(0, idx)
    return tensor


def _autoregressive_rollout_checkpoint_batch(
    model: nn.Module,
    batch: Dict,
    loss_fn: "RMMDLossFunction",
    device: str,
    n_checkpoints: int,
    epoch: int,
    max_epochs: int,
    normalization_stats: Dict,
    compute_loss: bool = True,
) -> tuple["torch.Tensor", float, int, Dict[int, float]]:
    """Checkpoint-to-checkpoint rollout using sparse y_seq (proven at T=1).

    One model step per DIRECT_COMPACT_HORIZONS checkpoint with dt = delta between
    checkpoints.  Supervision uses y_seq / geom_seq + target_mask (same as the
    script that reached ~0.001 NRMSE at T=1 on rebuilt datasets).
    """
    ni_curr = batch["ni_t0"].to(device)
    geom_curr = batch["geom_t0"].to(device)
    pre_shot = batch["pre_shot_context"].to(device)
    limiter = batch["limiter_geometry_tensor"].to(device)
    machine_names = [str(m) for m in batch["machine"]]
    y_seq = batch["y_seq"].to(device)
    geom_seq_tgt = batch["geom_seq"].to(device)
    target_mask_t = batch.get("target_mask")

    pre_shot = torch.nan_to_num(pre_shot, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.any(pre_shot.abs().amax(dim=1, keepdim=True) > 100.0):
        pre_shot = torch.sign(pre_shot) * torch.log1p(pre_shot.abs())
    pre_shot = torch.clamp(pre_shot, min=-12.0, max=12.0)

    n_steps = int(min(n_checkpoints, y_seq.shape[1], len(DIRECT_COMPACT_HORIZONS)))
    pre_shot_scalars_list = batch.get("pre_shot_scalars", [{}] * len(machine_names))
    omega_t, omega_d = _compute_omegas_for_compact_batch(
        ni_curr.detach().cpu(), pre_shot_scalars_list, machine_names, device, normalization_stats,
    )
    z_true = model.state_legacy_encoder(ni_curr)

    total_loss = torch.zeros(1, device=device).squeeze() if compute_loss else torch.tensor(0.0, device=device)
    n_scored = 0
    ni_nrmse_final = float("nan")
    batch_size = ni_curr.shape[0]
    report_nrmse: Dict[int, List[float]] = {int(h): [] for h in _REPORT_HORIZONS}

    prev_h = 0
    for c in range(n_steps):
        h = int(DIRECT_COMPACT_HORIZONS[c])
        dt = h - prev_h
        prev_h = h
        step_dt = torch.full((batch_size, 1), _normalized_step_dt(dt), device=device)

        out = model(
            x_t=ni_curr,
            machine_names=machine_names,
            omega_t=omega_t,
            omega_d=omega_d,
            batch_data={
                "compact_mode": True,
                "pre_shot_context": pre_shot,
                "limiter_geometry_tensor": limiter,
                "ni_t0": ni_curr,
                "geometry_tensor": geom_curr,
                "step_dt": step_dt,
            },
        )
        ni_curr = out.x_next
        geom_curr = out.geometry_pred

        x_true_step = y_seq[:, c, :]
        geom_true_step = geom_seq_tgt[:, c, :, :]
        valid = (
            target_mask_t[:, c].to(device)
            if target_mask_t is not None
            else torch.ones(batch_size, dtype=torch.bool, device=device)
        )
        if not valid.any():
            continue

        if compute_loss:
            z_pred = out.latent_next if getattr(out, "latent_next", None) is not None else out.rmmd.z_next
            step_losses = loss_fn(
                x_true=x_true_step,
                x_pred=ni_curr,
                z_true=z_true,
                z_pred=z_pred,
                d_total=out.rmmd.d_total,
                d_res=out.rmmd.d_res,
                epoch=epoch,
                max_epochs=max_epochs,
                geom_pred=geom_curr,
                geom_target=geom_true_step,
                s_matrix=out.rmmd.k_sym,
                d_psd=out.rmmd.d_psd,
                shared_private_penalty=(
                    model.shared_private_penalty() if hasattr(model, "shared_private_penalty") else None
                ),
            )
            total_loss = total_loss + step_losses["total"]
            n_scored += 1

        nrmse_step = float(_ni_nrmse_tensor(ni_curr[valid], x_true_step[valid]).mean().item())
        ni_nrmse_final = nrmse_step
        if h in report_nrmse:
            report_nrmse[h].append(nrmse_step)

    if compute_loss and n_scored > 0:
        total_loss = total_loss / n_scored

    report_means = {
        h: float(np.mean(vals)) if vals else float("nan")
        for h, vals in report_nrmse.items()
    }
    return total_loss, ni_nrmse_final, n_scored, report_means


def _advance_rollout_state(
    ni_pred: torch.Tensor,
    geom_pred: torch.Tensor,
    step: int,
    tbptt_steps: int,
    state_noise_std: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce the state fed to the next rollout step.

    - Truncated BPTT: detach every `tbptt_steps` steps. tbptt_steps==1 => detach
      every step (pure per-step flow-map / pushforward training): each gradient is a
      clean 1-step update from a realistic (possibly drifted) state, so near-term
      accuracy (T1) is NOT diluted by long-horizon gradients, yet the map is trained
      on its own rollout distribution (kills exposure bias / identity collapse).
    - Optional small Gaussian drift noise on the detached state hardens the map
      against its own error so compounding stays small.
    """
    ni_curr = ni_pred
    geom_curr = geom_pred
    if training and tbptt_steps and tbptt_steps > 0 and (step % tbptt_steps == 0):
        ni_curr = ni_curr.detach()
        geom_curr = geom_curr.detach()
        if state_noise_std and state_noise_std > 0.0:
            ni_curr = ni_curr + state_noise_std * torch.randn_like(ni_curr)
    return ni_curr, geom_curr


# Diagnostic: mean one-step Jacobian amplification ||J^T v||/||v||. <=1 means the flow map is non-expansive
# (rollout errors cannot grow); >1 means it amplifies them.
_LAST_JAC_RATIO: float = float("nan")
# Diagnostics for the (A) tendency + conservation losses (updated per rollout).
_LAST_TENDENCY: float = float("nan")   # relative increment error  ||dNI_pred - dNI_true||^2 / ||dNI_true||^2
_LAST_CONS: float = float("nan")       # relative drift of the (volume-weighted) particle integral
# DgknetHybrid gate supervision (0 = off): the gate is supervised on per-step true dynamism so it opens on
# q4 and stays at 0 on quiet shots; skip-competence trains the dgknet skip everywhere.
_GATE_SUP_WEIGHT: float = 0.0
_SKIP_COMPETENCE_WEIGHT: float = 0.0
_GATE_TARGET_SCALE: float = 0.1        # relative |dNI| at which the gate target ~= 0.76 (tanh(1))
_LAST_GATE_MEAN: float = float("nan")  # diagnostic: mean APPLIED gate over the last rollout
# Persistence baseline: NRMSE of NI(t)=NI(0) over the validation pass. The model's NRMSE should sit well
# below this.
_PERSIST_ACC: Dict[int, List[float]] = {}


def _conservation_volume_weights(mode: str, n: int, device: str) -> torch.Tensor:
    """Per-radial-point weights approximating the flux-surface volume element V'(rho).

    'radial' uses w_i ∝ (i + 0.5) (cylindrical V' ∝ rho proxy); 'uniform' uses ones.
    NOTE: this is a PROXY. The exact V'(rho) from the geometry tensor is a C deliverable;
    until then the conservation term pins a volume-weighted integral, not exact particles.
    """
    if str(mode) == "uniform":
        w = torch.ones(n, device=device)
    else:  # 'radial'
        w = (torch.arange(n, device=device, dtype=torch.float32) + 0.5)
    return w / w.mean().clamp_min(1e-8)


def _contractivity_penalty(
    model: nn.Module,
    ni_curr: torch.Tensor,
    geom_curr: torch.Tensor,
    pre_shot: torch.Tensor,
    limiter: torch.Tensor,
    machine_names: List[str],
    omega_t: torch.Tensor,
    omega_d: torch.Tensor,
    step_dt: torch.Tensor,
    device: str,
    target_gain: float = 1.0,
    n_probes: int = 1,
) -> torch.Tensor:
    """JAWS-style non-expansiveness penalty on the one-step NI flow map.

    Autoregressive rollout error propagates as  e_{t+1} ≈ J(x_t) · e_t  with
    J = ∂x_next/∂x_t.  If the largest singular value of J exceeds 1 the map is
    expansive and rollout error compounds (the observed T1→T20 blow-up); if J is
    non-expansive the error cannot grow.  We push the *decoded NI flow map* (not
    just the latent operator) toward contractivity by penalising expansive
    directions, estimated cheaply with a Hutchinson/VJP probe: for a Rademacher
    vector v, ||J^T v|| / ||v|| is the amplification along v.  Because the model
    predicts NI as a residual (x_next = ni + delta), J = I + ∂delta/∂ni, so a
    perfect identity step gives ratio == 1 and incurs ZERO penalty — the term only
    fights *expansion*, it never suppresses legitimate dynamics.  This is the
    observable-space realisation of the RMMD D_res dissipation.

    Refs: JAWS — Spatially-Adaptive Jacobian Regularization (arXiv:2603.05538);
    Message-Passing Neural PDE Solvers / pushforward trick (arXiv:2202.03376).
    """
    global _LAST_JAC_RATIO
    if not torch.is_grad_enabled():
        return torch.zeros((), device=device)
    ni_in = ni_curr.detach().clone().requires_grad_(True)
    batch_data_step = {
        "compact_mode": True,
        "pre_shot_context": pre_shot.detach(),
        "limiter_geometry_tensor": limiter.detach(),
        "ni_t0": ni_in,
        "geometry_tensor": geom_curr.detach(),
        "step_dt": step_dt.detach(),
    }
    out = model(
        x_t=ni_in,
        machine_names=machine_names,
        omega_t=omega_t.detach(),
        omega_d=omega_d.detach(),
        batch_data=batch_data_step,
    )
    x_next = out.x_next
    pen = torch.zeros((), device=device)
    ratio_accum = 0.0
    n_probes = max(1, int(n_probes))
    for _ in range(n_probes):
        v = torch.randint(0, 2, x_next.shape, device=device).to(x_next.dtype).mul_(2).sub_(1)
        g = torch.autograd.grad(
            outputs=x_next, inputs=ni_in, grad_outputs=v,
            create_graph=True, retain_graph=True,
        )[0]
        num = torch.sqrt(torch.sum(g.reshape(g.shape[0], -1) ** 2, dim=1) + 1e-12)
        den = torch.sqrt(torch.sum(v.reshape(v.shape[0], -1) ** 2, dim=1) + 1e-12)
        ratio = num / den
        pen = pen + torch.mean(F.relu(ratio - float(target_gain)) ** 2)
        ratio_accum += float(ratio.mean().detach().item())
    _LAST_JAC_RATIO = ratio_accum / n_probes
    return pen / n_probes


def _autoregressive_rollout_unit_step_batch(
    model: nn.Module,
    batch: Dict,
    loss_fn: "RMMDLossFunction",
    device: str,
    max_time_step: int,
    epoch: int,
    max_epochs: int,
    normalization_stats: Dict,
    compute_loss: bool = True,
    tbptt_steps: int = 1,
    step_anchor_weight: float = 4.0,
    step_anchor_tau: float = 2.0,
    state_noise_std: float = 0.0,
    jacobian_weight: float = 0.0,
    jacobian_target: float = 1.0,
    jac_max_steps: int = 6,
    jac_probes: int = 1,
    tendency_weight: float = 0.0,
    conservation_weight: float = 0.0,
    cons_volume_mode: str = "radial",
    direct_prediction: bool = False,
) -> tuple["torch.Tensor", float, int, Dict[int, float]]:
    """Unit-step autoregressive rollout: dt=1 every step, loss at EVERY step 1..T.

    If direct_prediction: every horizon t is predicted DIRECTLY from the true initial
    condition (ni_curr := ni_t0, step_dt := absolute time t), never from a previous
    prediction. This eliminates error accumulation/drift BY CONSTRUCTION (no recurrence),
    so long-horizon error = the direct t-step prediction error, not 1000 compounding ones.

    Targets come from dense ni_traj[:, step-1] (physical time step), NOT sparse y_seq
    indexed by DIRECT_COMPACT_HORIZONS[c] (which breaks when checkpoints are skipped).

    EARLY-STEP ANCHORING: a plain mean over steps lets the model minimize the average
    by retreating toward identity (tiny deltas everywhere) — which destroys the sharp
    1-step accuracy (T1 0.001 -> 0.02) and stalls long horizons at the identity-drift
    baseline. We instead weight step k by  w_k = 1 + a*exp(-(k-1)/tau)  so the first
    few steps (which the model CAN nail) dominate and pin real dynamics, while every
    step keeps a baseline weight of 1 (long horizons still train; no exponential kill).

    Truncated BPTT: every `tbptt_steps` the recurrent state (ni_curr/geom_curr) is
    detached so the gradient chain through the rollout stays bounded. This prevents
    exploding/vanishing gradients and runaway memory on long (T>100) rollouts while
    still supervising every step. tbptt_steps<=0 disables truncation (full BPTT).

    Returns (total_loss, ni_nrmse_at_T_frontier, n_scored, per_report_horizon_nrmse).
    """
    ni_curr = batch["ni_t0"].to(device)
    geom_curr = batch["geom_t0"].to(device)
    pre_shot = batch["pre_shot_context"].to(device)
    limiter = batch["limiter_geometry_tensor"].to(device)
    machine_names = [str(m) for m in batch["machine"]]
    ni_traj = batch["ni_traj"].to(device)
    geom_traj = batch["geom_traj"].to(device)
    traj_len = batch["traj_len"].to(device)
    # Time-resolved exogenous drivers (actuators) per step, if the dataset provides them.
    # Shape [B, T, K]. KNOWN control inputs (not leakage). None -> model uses zeros.
    drivers_traj = batch.get("drivers_traj")
    if isinstance(drivers_traj, torch.Tensor):
        drivers_traj = drivers_traj.to(device)
    else:
        drivers_traj = None

    pre_shot = torch.nan_to_num(pre_shot, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.any(pre_shot.abs().amax(dim=1, keepdim=True) > 100.0):
        pre_shot = torch.sign(pre_shot) * torch.log1p(pre_shot.abs())
    pre_shot = torch.clamp(pre_shot, min=-12.0, max=12.0)

    batch_size = ni_curr.shape[0]
    max_step = int(max_time_step)
    step_dt = torch.full((batch_size, 1), _normalized_step_dt(1.0), device=device)

    pre_shot_scalars_list = batch.get("pre_shot_scalars", [{}] * len(machine_names))
    omega_t, omega_d = _compute_omegas_for_compact_batch(
        ni_curr.detach().cpu(), pre_shot_scalars_list, machine_names, device, normalization_stats,
    )
    z_true = model.state_legacy_encoder(ni_curr)

    total_loss = torch.zeros(1, device=device).squeeze() if compute_loss else torch.tensor(0.0, device=device)
    n_scored = 0
    weight_sum = 0.0
    ni_nrmse_final = 0.0
    jac_pen_sum = torch.zeros((), device=device)
    jac_count = 0
    # (A) tendency + conservation accumulators. ni_t0_ref is the TRUE initial NI, used as
    # the "previous true state" for the step-1 increment (ni_curr is overwritten in-loop).
    ni_t0_ref = ni_curr
    tend_sum = torch.zeros((), device=device)
    tend_count = 0
    cons_sum = torch.zeros((), device=device)
    cons_count = 0
    cons_w = _conservation_volume_weights(cons_volume_mode, ni_curr.shape[-1], device)
    gate_sup_sum = torch.zeros((), device=device)   # DgknetHybrid gate supervision + skip competence
    skip_comp_sum = torch.zeros((), device=device)
    gate_aux_count = 0
    gate_mean_acc: List[float] = []
    report_nrmse: Dict[int, List[float]] = {int(h): [] for h in _REPORT_HORIZONS}

    for step in range(1, max_step + 1):
        if direct_prediction:
            # DIRECT-from-IC: predict horizon `step` from the TRUE initial condition with
            # absolute-time conditioning (step_dt = normalized t). No feedback => no drift.
            ni_curr = ni_t0_ref
            step_dt = torch.full((batch_size, 1), _normalized_step_dt(float(step)), device=device)
        batch_data_step = {
            "compact_mode": True,
            "pre_shot_context": pre_shot,
            "limiter_geometry_tensor": limiter,
            "ni_t0": ni_curr,
            "geometry_tensor": geom_curr,
            "step_dt": step_dt,
        }
        # Drive at the step being predicted (index step-1, clamped to available length).
        if drivers_traj is not None and drivers_traj.shape[1] > 0:
            d_idx = min(step - 1, drivers_traj.shape[1] - 1)
            batch_data_step["drivers"] = drivers_traj[:, d_idx, :]
        out = model(
            x_t=ni_curr,
            machine_names=machine_names,
            omega_t=omega_t,
            omega_d=omega_d,
            batch_data=batch_data_step,
        )
        # Grad-ful prediction used for loss + metrics at THIS step.
        ni_pred = out.x_next
        geom_pred = out.geometry_pred

        tidx = step - 1
        if tidx >= ni_traj.shape[1]:
            break

        # Per-sample validity from traj_len (post-curriculum slice), not padded batch width.
        valid = traj_len > tidx
        if not valid.any():
            # Still advance the fed-forward state so the rollout continues (autoregressive only).
            if not direct_prediction:
                ni_curr, geom_curr = _advance_rollout_state(
                    ni_pred, geom_pred, step, tbptt_steps, state_noise_std, compute_loss
                )
            continue

        x_true = ni_traj[:, tidx, :]
        geom_true = geom_traj[:, tidx, :, :]

        if compute_loss:
            z_pred = out.latent_next if getattr(out, "latent_next", None) is not None else out.rmmd.z_next
            idx = None if valid.all() else valid.nonzero(as_tuple=True)[0]
            step_losses = loss_fn(
                x_true=_select_batch_rows(x_true, idx, batch_size),
                x_pred=_select_batch_rows(ni_pred, idx, batch_size),
                z_true=_select_batch_rows(z_true, idx, batch_size),
                z_pred=_select_batch_rows(z_pred, idx, batch_size),
                d_total=_select_batch_rows(out.rmmd.d_total, idx, batch_size),
                d_res=_select_batch_rows(out.rmmd.d_res, idx, batch_size),
                epoch=epoch,
                max_epochs=max_epochs,
                geom_pred=_select_batch_rows(geom_pred, idx, batch_size),
                geom_target=_select_batch_rows(geom_true, idx, batch_size),
                s_matrix=_select_batch_rows(out.rmmd.k_sym, idx, batch_size),
                d_psd=_select_batch_rows(out.rmmd.d_psd, idx, batch_size),
                shared_private_penalty=(
                    model.shared_private_penalty() if hasattr(model, "shared_private_penalty") else None
                ),
            )
            step_w = 1.0 + float(step_anchor_weight) * math.exp(-(step - 1) / max(1e-6, float(step_anchor_tau)))
            total_loss = total_loss + step_w * step_losses["total"]
            weight_sum += step_w
            n_scored += 1

            # DgknetHybrid: supervise the gate on per-step local dynamism (relative |dNI|) so it opens on q4 and stays
            # at 0 on quiet shots; train the dgknet skip everywhere.
            if _GATE_SUP_WEIGHT > 0.0 and getattr(model, "last_gate_logit", None) is not None:
                prev_true = ni_traj[:, tidx - 1, :] if tidx >= 1 else ni_t0_ref
                d_mag = torch.sqrt(torch.sum((x_true - prev_true) ** 2, dim=1) + 1e-12)
                rel = d_mag / (torch.norm(x_true, dim=1) + 1e-6)
                gate_target = torch.tanh(rel / max(_GATE_TARGET_SCALE, 1e-6)).detach()   # (B,) in [0,1)
                logit_v = model.last_gate_logit.view(-1)[valid]
                gate_sup_sum = gate_sup_sum + torch.nn.functional.binary_cross_entropy_with_logits(
                    logit_v, gate_target[valid])
                ndgk = getattr(model, "last_ni_dgk", None)
                if _SKIP_COMPETENCE_WEIGHT > 0.0 and isinstance(ndgk, torch.Tensor):
                    skip_comp_sum = skip_comp_sum + _ni_nrmse_tensor(ndgk[valid], x_true[valid]).mean()
                gate_aux_count += 1
                gate_mean_acc.append(float(getattr(model, "last_gate_mean", 0.0)))

        nrmse_step = float(_ni_nrmse_tensor(ni_pred[valid], x_true[valid]).mean().item())
        ni_nrmse_final = nrmse_step
        if step in report_nrmse:
            report_nrmse[step].append(nrmse_step)
        # Persistence baseline NI(t)=NI(0): accumulate during VALIDATION only (no grad).
        if not torch.is_grad_enabled() and step in report_nrmse:
            pers_step = float(_ni_nrmse_tensor(ni_t0_ref[valid], x_true[valid]).mean().item())
            _PERSIST_ACC.setdefault(int(step), []).append(pers_step)

        # Contractivity (non-expansiveness) penalty on the one-step flow map at the
        # CURRENT operating point.  Applied to the near-term steps (where error is
        # introduced and where it must not be amplified).  Only meaningful with grad.
        if (
            compute_loss
            and not direct_prediction
            and jacobian_weight > 0.0
            and step <= int(jac_max_steps)
            and torch.is_grad_enabled()
        ):
            jac_pen_sum = jac_pen_sum + _contractivity_penalty(
                model, ni_curr, geom_curr, pre_shot, limiter, machine_names,
                omega_t, omega_d, step_dt, device,
                target_gain=jacobian_target, n_probes=jac_probes,
            )
            jac_count += 1

        # (A) TENDENCY + CONSERVATION losses (supplement; attack the additive per-step
        # bias that drives the ~linear error growth). Flat-averaged over steps (the bias
        # is uniform across the horizon), separate from the anchor-weighted state loss.
        if compute_loss and not direct_prediction and (tendency_weight > 0.0 or conservation_weight > 0.0):
            prev_true = ni_traj[:, tidx - 1, :] if tidx >= 1 else ni_t0_ref
            if tendency_weight > 0.0:
                # Relative increment error: ||dNI_pred - dNI_true||^2 / ||dNI_true||^2.
                # Normalising by the SMALL true increment (not the big state) gives the
                # under-supervised tendency a real gradient and directly penalises the
                # per-step bias (dNI_pred - dNI_true == e_k - e_{k-1}).
                d_pred = (ni_pred - ni_curr)[valid]
                d_true = (x_true - prev_true)[valid]
                t_num = torch.mean(torch.sum((d_pred - d_true) ** 2, dim=1))
                # Floor the denominator at a MEANINGFUL scale. Flat-top steps have
                # ||dNI_true||^2 ~ 0; a 1e-8 floor made the relative error blow up to ~1e7.
                # Floor at tend_eps (abs) so flat-top -> tend ~ 0 (model correctly predicts
                # ~no change); clamp the ratio as a final guard against pathological batches.
                t_den = torch.mean(torch.sum(d_true ** 2, dim=1)).clamp_min(1e-3)
                tend_sum = tend_sum + torch.clamp(t_num / t_den, max=20.0)
                tend_count += 1
            if conservation_weight > 0.0:
                # Relative drift of the volume-weighted integral sum_i NI_i V'_i.
                # NOTE: in NORMALISED space NI is ~zero-mean so this integral is ~0 and
                # is a (clamped) global-drift anchor, NOT true particle number — real
                # particle conservation needs denormalised NI + true V'(rho) (a C task).
                int_pred = torch.sum(ni_pred[valid] * cons_w, dim=1)
                int_true = torch.sum(x_true[valid] * cons_w, dim=1)
                cons_rel = torch.abs(int_pred - int_true) / torch.abs(int_true).clamp_min(0.1)
                cons_sum = cons_sum + torch.mean(cons_rel).clamp(max=10.0)
                cons_count += 1

        # Build the state fed to the NEXT step (autoregressive only): detach per truncation
        # window + optional drift noise. In direct mode ni_curr is reset to the IC each step.
        if not direct_prediction:
            ni_curr, geom_curr = _advance_rollout_state(
                ni_pred, geom_pred, step, tbptt_steps, state_noise_std, compute_loss
            )

    global _LAST_TENDENCY, _LAST_CONS, _LAST_GATE_MEAN
    if compute_loss and weight_sum > 0:
        total_loss = total_loss / weight_sum
        # Add the contractivity regulariser as an ABSOLUTE term (not diluted by the
        # per-step weight sum): it directly caps how fast rollout error can grow.
        if jac_count > 0 and jacobian_weight > 0.0:
            total_loss = total_loss + float(jacobian_weight) * (jac_pen_sum / jac_count)
        if tend_count > 0 and tendency_weight > 0.0:
            tend_mean = tend_sum / tend_count
            total_loss = total_loss + float(tendency_weight) * tend_mean
            _LAST_TENDENCY = float(tend_mean.detach().item())
        if cons_count > 0 and conservation_weight > 0.0:
            cons_mean = cons_sum / cons_count
            total_loss = total_loss + float(conservation_weight) * cons_mean
            _LAST_CONS = float(cons_mean.detach().item())
        # DgknetHybrid gate supervision + skip competence (ABSOLUTE terms, like the aux above).
        if gate_aux_count > 0 and _GATE_SUP_WEIGHT > 0.0:
            total_loss = total_loss + float(_GATE_SUP_WEIGHT) * (gate_sup_sum / gate_aux_count)
            if _SKIP_COMPETENCE_WEIGHT > 0.0:
                total_loss = total_loss + float(_SKIP_COMPETENCE_WEIGHT) * (skip_comp_sum / gate_aux_count)
            _LAST_GATE_MEAN = float(np.mean(gate_mean_acc)) if gate_mean_acc else float("nan")
    elif n_scored == 0:
        ni_nrmse_final = float("nan")

    report_means = {
        h: float(np.mean(vals)) if vals else float("nan")
        for h, vals in report_nrmse.items()
    }
    return total_loss, ni_nrmse_final, n_scored, report_means


def _autoregressive_rollout_batch(
    model: nn.Module,
    batch: Dict,
    loss_fn: "RMMDLossFunction",
    device: str,
    max_time_step: int,
    epoch: int,
    max_epochs: int,
    normalization_stats: Dict,
    compute_loss: bool = True,
    curriculum_frontier: int = 1,
    step_anchor_weight: float = 4.0,
    step_anchor_tau: float = 2.0,
    tbptt_steps: int = 1,
    state_noise_std: float = 0.0,
    jacobian_weight: float = 0.0,
    jacobian_target: float = 1.0,
    jac_max_steps: int = 6,
    jac_probes: int = 1,
    tendency_weight: float = 0.0,
    conservation_weight: float = 0.0,
    cons_volume_mode: str = "radial",
    direct_prediction: bool = False,
) -> tuple["torch.Tensor", float, int, Dict[int, float]]:
    """Dispatch: frontier==1 uses y_seq checkpoint step; frontier>1 (or direct mode) uses unit-step."""
    if int(curriculum_frontier) <= 1 and not direct_prediction:
        return _autoregressive_rollout_checkpoint_batch(
            model,
            batch,
            loss_fn,
            device,
            n_checkpoints=1,
            epoch=epoch,
            max_epochs=max_epochs,
            normalization_stats=normalization_stats,
            compute_loss=compute_loss,
        )
    return _autoregressive_rollout_unit_step_batch(
        model,
        batch,
        loss_fn,
        device,
        max_time_step,
        epoch,
        max_epochs,
        normalization_stats,
        compute_loss=compute_loss,
        tbptt_steps=tbptt_steps,
        step_anchor_weight=step_anchor_weight,
        step_anchor_tau=step_anchor_tau,
        state_noise_std=state_noise_std,
        jacobian_weight=jacobian_weight,
        jacobian_target=jacobian_target,
        jac_max_steps=jac_max_steps,
        jac_probes=jac_probes,
        tendency_weight=tendency_weight,
        conservation_weight=conservation_weight,
        cons_volume_mode=cons_volume_mode,
        direct_prediction=direct_prediction,
    )


@torch.no_grad()
def _rollout_compact_shot_to_checkpoints(
    model: nn.Module,
    ni_t0: torch.Tensor,
    geom_t0: torch.Tensor,
    pre_shot_context: torch.Tensor,
    limiter_geometry: torch.Tensor,
    ni_traj: torch.Tensor,
    geom_traj: torch.Tensor,
    machine: str,
    pre_shot_scalars: Dict,
    device: str,
    normalization_stats: Dict,
    max_time_step: int | None = None,
    drivers_traj: torch.Tensor | None = None,
    report_horizons: Sequence[int] | None = None,
) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Unit-step rollout for eval; record predictions only at DIRECT_COMPACT_HORIZONS.

    LEAKAGE-CLEAN: feeds the model its OWN predicted geometry/NI each step (never the true
    future ni_traj/geom_traj — those are TARGETS only). drivers_traj is the KNOWN exogenous
    actuator program (not leakage); fed per step so eval matches the driver-trained model."""
    ni_curr = ni_t0.unsqueeze(0).to(device)
    geom_curr = geom_t0.unsqueeze(0).to(device)
    pre_shot = pre_shot_context.unsqueeze(0).to(device)
    pre_shot = torch.nan_to_num(pre_shot, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.any(pre_shot.abs().amax(dim=1, keepdim=True) > 100.0):
        pre_shot = torch.sign(pre_shot) * torch.log1p(pre_shot.abs())
    pre_shot = torch.clamp(pre_shot, min=-12.0, max=12.0)
    limiter = limiter_geometry.unsqueeze(0).to(device)
    step_dt = torch.full((1, 1), _normalized_step_dt(1.0), device=device)

    omega_t, omega_d = _compute_omegas_for_compact_batch(
        ni_t0.unsqueeze(0), [pre_shot_scalars], [machine], device, normalization_stats,
    )

    ni_preds: Dict[int, torch.Tensor] = {}
    geom_preds: Dict[int, torch.Tensor] = {}
    report_set = set(report_horizons) if report_horizons is not None else set(DIRECT_COMPACT_HORIZONS)
    max_step = int(max_time_step if max_time_step is not None else ni_traj.shape[0])
    max_step = min(max_step, int(ni_traj.shape[0]))

    drivers_t = drivers_traj.to(device) if isinstance(drivers_traj, torch.Tensor) else None
    for step in range(1, max_step + 1):
        batch_data = {
            "compact_mode": True,
            "pre_shot_context": pre_shot,
            "limiter_geometry_tensor": limiter,
            "ni_t0": ni_curr,
            "geometry_tensor": geom_curr,
            "step_dt": step_dt,
        }
        if drivers_t is not None and drivers_t.shape[0] > 0:
            d_idx = min(step - 1, drivers_t.shape[0] - 1)
            batch_data["drivers"] = drivers_t[d_idx].unsqueeze(0)
        out = model(
            x_t=ni_curr,
            machine_names=[machine],
            omega_t=omega_t,
            omega_d=omega_d,
            batch_data=batch_data,
        )
        ni_curr = out.x_next
        geom_curr = out.geometry_pred
        if step in report_set:
            ni_preds[step] = ni_curr.detach().cpu().squeeze(0)
            if geom_curr is not None:
                geom_preds[step] = geom_curr.detach().cpu().squeeze(0)

    return ni_preds, geom_preds


def _batch_bool_flag(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.to(dtype=torch.bool).flatten().all().item())
    if isinstance(value, (list, tuple)):
        return all(bool(item) for item in value)
    return bool(value)


# ---------------------------------------------------------------------------
# LEGACY COMPATIBILITY SHIM — kept for eval backward-compat but not used in
# the autoregressive training path.
# ---------------------------------------------------------------------------
def _compact_long_horizon_aux_loss(
    out,
    flat_batch: Dict,
    long_horizon_weight: float,
    short_horizon_weight: float,
    short_horizon_scale: float,
    geom_weight: float,
) -> "torch.Tensor":
    """No-op shim kept for backward compatibility.  No longer used."""
    device = out.x_next.device if hasattr(out, "x_next") else torch.device("cpu")
    return torch.tensor(0.0, device=device)



# ============================================================================
# DATA LOADING & MODEL CONSTRUCTION
# ============================================================================

def _load_phase0_dataset(path: Path) -> Dict:
    payload = load_phase0_payload(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid payload format at {path}: expected dict")
    if "data" not in payload and "samples" not in payload and not is_sharded_payload(payload):
        raise ValueError(f"Invalid payload format at {path}: missing 'data', 'samples', or sharded format")
    return payload


def _make_model(
    machine_names: Sequence[str],
    state_dim: int = 40,
    latent_dim: int = 256,
    latent_profile: int = 96,
    latent_geom: int = 96,
    machine_embedding_dim: int = 24,
    n_harmonics: int = 4,
    use_transport_step: bool = True,
    n_drivers: int = N_DRIVERS,
    ablate_drivers: bool = False,
    ablate_geometry: bool = False,
    ablate_dres: bool = False,
    drivergate: bool = False,
    model_type: str = "rmmd",
    baseline_latent_dim: int = 128,
    ablate_skip: bool = False,
    hybrid_skip_hidden: int = 512,
):
    # Baseline models (mlp/lstm/node) share the RMMD forward/harness contract. baseline_latent_dim=128 default;
    _mt = str(model_type).lower()
    _rmmd_kwargs = dict(
        state_dim=state_dim, latent_dim=latent_dim,
        machine_names=list(sorted(set(machine_names))),
        machine_embedding_dim=machine_embedding_dim, n_harmonics=n_harmonics,
        latent_profile=latent_profile, latent_geom=latent_geom,
        use_transport_step=use_transport_step, n_drivers=n_drivers,
        ablate_drivers=ablate_drivers, ablate_geometry=ablate_geometry, ablate_dres=ablate_dres,
        drivergate=drivergate,
    )
    if _mt == "hybrid":
        # RMMD operator (SUT/D_res/transport/drivers/geometry intact) + gated flexible MLP skip.
        from strong_rmmd.hybrid_rmmd import HybridRMMD
        return HybridRMMD(ablate_skip=ablate_skip, hybrid_skip_hidden=int(hybrid_skip_hidden), **_rmmd_kwargs)
    if _mt in ("dgknet-hybrid", "dgknet_hybrid", "dgkhybrid"):
        # RMMD on quiet shots + a genuine dgknet (MetriplecticKoopman) operator on q4, dead-zone
        # per-shot gate. SUT/D_res/transport of the RMMD side all intact; ablate_skip recovers RMMD.
        from strong_rmmd.dgknet_hybrid_rmmd import DgknetHybridRMMD
        return DgknetHybridRMMD(ablate_skip=ablate_skip, **_rmmd_kwargs)
    if _mt in ("fused", "fused-rmmd", "fused_rmmd"):
        # SOTA build: RMMD blended with a DGKNet skip by a PER-RADIUS learned gate (blend, not switch).
        # env FUSED_RMMD_CKPT loads+freezes the trained RMMD parent so only gate+skip train (cheap).
        from strong_rmmd.fused_rmmd import FusedRMMD
        return FusedRMMD(ablate_skip=ablate_skip, **_rmmd_kwargs)
    if _mt not in ("rmmd", "", "none"):
        from strong_rmmd.baselines import make_baseline
        return make_baseline(
            _mt, machine_names=list(sorted(set(machine_names))),
            n_drivers=n_drivers, latent_dim=int(baseline_latent_dim),
            machine_embedding_dim=machine_embedding_dim,
        )
    return MultiMachineRMMD(**_rmmd_kwargs)


def _infer_model_dimensions(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    dims = {
        "state_dim": 40,
        "latent_dim": 256,
        "latent_profile": 96,
        "latent_geom": 96,
        "machine_embedding_dim": 24,
        "n_harmonics": 4,
    }
    if "s_universal" in state_dict:
        dims["latent_dim"] = int(state_dict["s_universal"].shape[0])
    if "machine_embedding.weight" in state_dict:
        dims["machine_embedding_dim"] = int(state_dict["machine_embedding.weight"].shape[1])
    if "profile_encoder.mlp.3.weight" in state_dict:
        dims["latent_profile"] = int(state_dict["profile_encoder.mlp.3.weight"].shape[0])
    elif "rmmd.kernel.mode_vectors" in state_dict:
        dims["latent_profile"] = int(state_dict["rmmd.kernel.mode_vectors"].shape[1])
    if "geometry_encoder.net.0.weight" in state_dict:
        dims["latent_geom"] = int(state_dict["geometry_encoder.net.0.weight"].shape[1]) // (40 * 66) * state_dict["geometry_encoder.net.0.weight"].shape[0]
    elif "rmmd_geom.kernel.mode_vectors" in state_dict:
        dims["latent_geom"] = int(state_dict["rmmd_geom.kernel.mode_vectors"].shape[1])
    if "rmmd.kernel.mode_vectors" in state_dict:
        dims["n_harmonics"] = int(state_dict["rmmd.kernel.mode_vectors"].shape[0])
    if "state_legacy_encoder.net.0.weight" in state_dict:
        dims["state_dim"] = int(state_dict["state_legacy_encoder.net.0.weight"].shape[1])
    return dims


def _collect_machine_names(payload: Dict) -> List[str]:
    sample_index = payload.get("sample_index") if isinstance(payload, dict) else None
    if isinstance(sample_index, list):
        machines: List[str] = []
        for entry in sample_index:
            machine = entry.get("machine", "UNKNOWN") if isinstance(entry, dict) else "UNKNOWN"
            machines.append(str(machine) if not isinstance(machine, bytes) else machine.decode())
        return machines

    data = payload.get("data") if isinstance(payload, dict) else None
    machines: List[str] = []
    if isinstance(data, list):
        for sample in data:
            machine = sample.get("machine", "UNKNOWN") if isinstance(sample, dict) else "UNKNOWN"
            machines.append(str(machine) if not isinstance(machine, bytes) else machine.decode())
        return machines

    view = Phase0DatasetView(payload)
    for idx in range(len(view)):
        sample = view.get_sample(idx)
        machine = sample.get("machine", "UNKNOWN")
        machines.append(str(machine) if not isinstance(machine, bytes) else machine.decode())
    return machines


# ============================================================================
# TRAINING LOOP  (autoregressive rollout — true 1-step-at-a-time dynamics)
# ============================================================================

def _train_epoch_with_curriculum(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    loss_fn: "RMMDLossFunction",
    device: str,
    epoch: int,
    max_epochs: int,
    gamma: float,
    normalization_stats: Dict[str, Any],
    max_time_step: int = 1,
    curriculum_frontier: int = 1,
    step_anchor_weight: float = 4.0,
    step_anchor_tau: float = 2.0,
    tbptt_steps: int = 1,
    state_noise_std: float = 0.0,
    jacobian_weight: float = 0.0,
    jacobian_target: float = 1.0,
    jac_max_steps: int = 6,
    jac_probes: int = 1,
    tendency_weight: float = 0.0,
    conservation_weight: float = 0.0,
    cons_volume_mode: str = "radial",
    direct_prediction: bool = False,
) -> float:
    """Train one epoch: checkpoint rollout at frontier=1, else unit-step 1..max_time_step."""
    model.train()
    total_loss = 0.0
    count = 0

    for batch_idx, batch in enumerate(loader):
        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            _rmmd_blk = getattr(model, "rmmd", None)
            dissip_frac = getattr(_rmmd_blk, "last_dissip_frac", float("nan"))
            offdiag_frac = getattr(_rmmd_blk, "last_offdiag_frac", float("nan"))
            logger.info(
                "  batch %d/%d  T_frontier=%d  jac=%.4f  D_res_dissip=%.4f  offdiag=%.3f  "
                "tend=%.4f (rel increment err)  cons=%.4f (particle drift)",
                batch_idx + 1, len(loader), max_time_step, _LAST_JAC_RATIO, dissip_frac, offdiag_frac,
                _LAST_TENDENCY, _LAST_CONS,
            )

        if not batch or "ni_t0" not in batch:
            continue
        if curriculum_frontier <= 1:
            if int(batch["y_seq"].shape[1]) < 1:
                continue
        else:
            if batch.get("ni_traj") is None or int(batch["ni_traj"].shape[1]) < 1:
                continue

        batch_loss, _, n_scored, _ = _autoregressive_rollout_batch(
            model, batch, loss_fn, device, max_time_step, epoch, max_epochs,
            normalization_stats, compute_loss=True, curriculum_frontier=curriculum_frontier,
            step_anchor_weight=step_anchor_weight, step_anchor_tau=step_anchor_tau,
            tbptt_steps=tbptt_steps, state_noise_std=state_noise_std,
            jacobian_weight=jacobian_weight, jacobian_target=jacobian_target,
            jac_max_steps=jac_max_steps, jac_probes=jac_probes,
            tendency_weight=tendency_weight, conservation_weight=conservation_weight,
            cons_volume_mode=cons_volume_mode, direct_prediction=direct_prediction,
        )
        if n_scored == 0:
            if batch_idx == 0:
                logger.warning(
                    "  batch 0: n_scored=0 (frontier=%d traj_len min=%s max=%s ni_traj W=%s y_seq W=%s)",
                    curriculum_frontier,
                    int(batch["traj_len"].min().item()),
                    int(batch["traj_len"].max().item()),
                    int(batch.get("ni_traj", torch.zeros(1, 0, 40)).shape[1]),
                    int(batch["y_seq"].shape[1]),
                )
            continue

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(batch_loss.item())
        count += 1

    if count == 0:
        logger.error(
            "Train epoch produced NO batches (ni_traj empty on all samples?). "
            "Rebuild data_build compact datasets."
        )
        return float("nan")
    return total_loss / count


@torch.no_grad()
def _validate_epoch_with_curriculum(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: "RMMDLossFunction",
    device: str,
    epoch: int,
    max_epochs: int,
    gamma: float,
    normalization_stats: Dict[str, Any],
    max_time_step: int = 1,
    curriculum_frontier: int = 1,
    step_anchor_weight: float = 4.0,
    step_anchor_tau: float = 2.0,
    tbptt_steps: int = 1,
    state_noise_std: float = 0.0,
    tendency_weight: float = 0.0,
    conservation_weight: float = 0.0,
    cons_volume_mode: str = "radial",
    direct_prediction: bool = False,
) -> tuple[float, float, Dict[int, float]]:
    """Validate: checkpoint rollout at frontier=1, else unit-step; report per-horizon NRMSE."""
    model.eval()
    total_loss = 0.0
    total_ni_nrmse = 0.0
    count = 0
    report_accum: Dict[int, List[float]] = {int(h): [] for h in _REPORT_HORIZONS}

    for batch in loader:
        if not batch or "ni_t0" not in batch:
            continue
        if curriculum_frontier <= 1:
            if int(batch["y_seq"].shape[1]) < 1:
                continue
        elif batch.get("ni_traj") is None or int(batch["ni_traj"].shape[1]) < 1:
            continue

        batch_loss, ni_nrmse, n_scored, report_means = _autoregressive_rollout_batch(
            model, batch, loss_fn, device, max_time_step, epoch, max_epochs,
            normalization_stats, compute_loss=True, curriculum_frontier=curriculum_frontier,
            step_anchor_weight=step_anchor_weight, step_anchor_tau=step_anchor_tau,
            tbptt_steps=tbptt_steps, state_noise_std=0.0,
            tendency_weight=tendency_weight, conservation_weight=conservation_weight,
            cons_volume_mode=cons_volume_mode, direct_prediction=direct_prediction,
        )
        if n_scored == 0:
            continue
        if ni_nrmse == ni_nrmse:
            total_ni_nrmse += ni_nrmse

        total_loss += float(batch_loss.item())
        count += 1
        for h, val in report_means.items():
            if val == val and val < 1e6:
                report_accum.setdefault(int(h), []).append(val)

    if count == 0:
        logger.error("Val epoch produced NO scored batches — check ni_traj in dataset.")
        avg_report = {int(h): float("nan") for h in _REPORT_HORIZONS}
        return float("nan"), float("nan"), avg_report

    avg_report = {
        h: float(np.mean(vals)) if vals else float("nan")
        for h, vals in report_accum.items()
    }
    return total_loss / count, total_ni_nrmse / count, avg_report


# ============================================================================
# TRAIN COMMAND
# ============================================================================

def train_command(args) -> None:
    logger.info("=" * 80)
    logger.info("STRONG-RMMD: Compact NI + Geometry Training (curriculum + physics ramp)")
    logger.info("=" * 80)

    # Reproducible seed replicates (decisive EXP-1 rule 5). Default None = current nondeterministic behavior.
    _seed = getattr(args, "seed", None)
    if _seed is not None:
        import random as _random
        _random.seed(int(_seed)); np.random.seed(int(_seed)); torch.manual_seed(int(_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(_seed))
        logger.info("SEED set to %d (reproducible replicate)", int(_seed))

    # DgknetHybrid gate supervision -> module globals (read inside the rollout loss, no threading).
    global _GATE_SUP_WEIGHT, _SKIP_COMPETENCE_WEIGHT, _GATE_TARGET_SCALE
    _GATE_SUP_WEIGHT = float(getattr(args, "gate_sup_weight", 0.0) or 0.0)
    _SKIP_COMPETENCE_WEIGHT = float(getattr(args, "skip_competence_weight", 0.0) or 0.0)
    _GATE_TARGET_SCALE = float(getattr(args, "gate_target_scale", 0.1) or 0.1)
    if str(getattr(args, "model", "rmmd")).lower() in ("dgknet-hybrid", "dgknet_hybrid", "dgkhybrid") \
            and _GATE_SUP_WEIGHT <= 0.0:
        _GATE_SUP_WEIGHT, _SKIP_COMPETENCE_WEIGHT = 1.0, 1.0   # sensible defaults so the gate IS trained
        logger.info("dgknet-hybrid: gate-sup/skip-competence default to 1.0 (set --gate-sup-weight to override)")

    # Fast/ablation protocol: one flag for a fast, uniform run (use the same for every model in the table).
    # Caps the frontier, loosens gating, and trims epochs. Explicit flags override.
    if bool(getattr(args, "fast_protocol", False)):
        if not getattr(args, "max_frontier", 0):
            args.max_frontier = 100
        args.epochs = min(int(getattr(args, "epochs", 120)), 70)   # --epochs 50 still gives 50
        args.curriculum_advance_threshold = max(float(getattr(args, "curriculum_advance_threshold", 0.05)), 0.15)
        # Curriculum min-hold: minimum epochs per frontier before it can advance; higher walks the curriculum slower.
        args.curriculum_min_hold_epochs = int(getattr(args, "curriculum_min_hold_epochs", 2))
        args.curriculum_max_hold_epochs = min(int(getattr(args, "curriculum_max_hold_epochs", 25)), 6)
        if args.curriculum_max_hold_epochs < args.curriculum_min_hold_epochs:
            args.curriculum_max_hold_epochs = args.curriculum_min_hold_epochs
        # Ramp the SUT alignment in proportional to the epoch budget so it reaches full weight by mid-training and
        # is enforced through the back half for any --epochs.
        _e = int(args.epochs)
        args.loss_sut_ramp_start = min(int(getattr(args, "loss_sut_ramp_start", 60)), max(3, _e // 6))
        args.loss_sut_ramp_end = min(int(getattr(args, "loss_sut_ramp_end", 220)), max(10, _e // 2))
        logger.info("FAST PROTOCOL: SUT ramp %d->%d (full for ~%d epochs)",
                    args.loss_sut_ramp_start, args.loss_sut_ramp_end, max(0, _e - args.loss_sut_ramp_end))
        logger.info(
            "FAST PROTOCOL: epochs=%d max_frontier=%d advance_thresh=%.2f hold=[%d,%d] "
            "(use identically for every ablation/baseline)",
            args.epochs, args.max_frontier, args.curriculum_advance_threshold,
            args.curriculum_min_hold_epochs, args.curriculum_max_hold_epochs,
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_data_path = Path(getattr(args, "compact_train_data", args.train_data))
    val_data_path = Path(getattr(args, "compact_val_data", args.val_data))

    train_payload = _load_phase0_dataset(train_data_path)
    val_payload = _load_phase0_dataset(val_data_path)

    # Load normalization stats: embedded payload stats take priority over recomputed
    normalization_stats = _ensure_normalization_stats(train_data_path, checkpoint_dir=checkpoint_dir)
    logger.info("Normalization stats keys: %s", list(normalization_stats.keys()))
    ni_stats = normalization_stats.get("kinetic_profiles.NI", {})
    geom_stats = normalization_stats.get("geometry_tensor", {})
    logger.info(
        "NI  stats: mean=%.4g std=%.4g",
        ni_stats.get("mean", float("nan")), ni_stats.get("std", float("nan")),
    )
    logger.info(
        "Geom stats: mean=%.4g std=%.4g",
        geom_stats.get("mean", float("nan")), geom_stats.get("std", float("nan")),
    )

    train_machines = _collect_machine_names(train_payload)

    _validate_unit_step_dataset(train_payload, normalization_stats)

    # Model dimensions (compact: smaller, faster)
    model_state_dim = len(PROFILE_ORDER) * 40   # 1 * 40 = 40
    model_latent_dim = min(int(args.latent_dim), 256)
    model_latent_profile = min(int(args.latent_profile), 96)
    model_latent_geom = min(int(args.latent_geom), 96)
    model_machine_embedding_dim = min(int(args.machine_embedding_dim), 24)
    model_n_harmonics = 4

    # Direct-prediction mode predicts each horizon from the IC via absolute-time (step_dt)
    # conditioning, so it needs the free-residual decode (the transport step uses a fixed dt=1
    # and would give the same x_1 for every horizon). Disable transport when direct.
    _direct = bool(getattr(args, "direct_prediction", False))
    # Ablation flags (each removes ONE novel component for the ablation table).
    _abl_drivers = bool(getattr(args, "ablate_drivers", False))
    _abl_geometry = bool(getattr(args, "ablate_geometry", False))
    _abl_dres = bool(getattr(args, "ablate_dres", False))
    _abl_transport = bool(getattr(args, "ablate_transport", False))
    _abl_skip = bool(getattr(args, "ablate_skip", False))
    _drivergate = bool(getattr(args, "drivergate", False))
    _use_transport = bool(getattr(args, "use_transport_step", True)) and not _direct and not _abl_transport
    _model_type = str(getattr(args, "model", "rmmd") or "rmmd").lower()
    # hybrid is an RMMD-FAMILY model (operator + SUT loss stay ON) -> NOT a baseline.
    _is_baseline = _model_type not in ("rmmd", "hybrid", "dgknet-hybrid", "dgknet_hybrid", "dgkhybrid", "", "none")
    model = _make_model(
        sorted(set(train_machines)),
        state_dim=model_state_dim,
        latent_dim=model_latent_dim,
        latent_profile=model_latent_profile,
        latent_geom=model_latent_geom,
        machine_embedding_dim=model_machine_embedding_dim,
        n_harmonics=model_n_harmonics,
        use_transport_step=_use_transport,
        ablate_drivers=_abl_drivers,
        ablate_geometry=_abl_geometry,
        ablate_dres=_abl_dres,
        drivergate=_drivergate,
        model_type=_model_type,
        baseline_latent_dim=int(getattr(args, "baseline_latent_dim", 128)),
        ablate_skip=_abl_skip,
        hybrid_skip_hidden=int(getattr(args, "hybrid_skip_hidden", 512)),
    ).to(args.device)
    _n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: type=%s params=%.2fM use_transport_step=%s direct=%s ablate[drivers=%s geometry=%s dres=%s transport=%s]",
        _model_type, _n_params / 1e6, _use_transport, _direct, _abl_drivers, _abl_geometry, _abl_dres, _abl_transport,
    )

    # Physics losses are ramped in after the data fit is established. latent_align_weight = 0.0 (future NI in the
    # latent signal would be training-time leakage). Heavy physics terms ramp from epoch 40+.
    loss_config = {
        "kinetics_weight": float(getattr(args, "loss_kinetics_weight", 0.75)),
        "geometry_weight": float(getattr(args, "loss_geometry_weight", 0.20)),
        "kinetics_cons_weight": float(getattr(args, "loss_kinetics_cons_weight", 0.05)),
        # Profile bounds enforce NON-NEGATIVITY, which is only meaningful in physical
        # space. We train in NORMALIZED space (zero-mean, unit-std) where NI is
        # legitimately negative, so this penalty is disabled — otherwise it biases all
        # predictions upward and corrupts the NI fit.
        "profile_bounds_weight": float(getattr(args, "loss_profile_bounds_weight", 0.0)),
        "hard_kinetics_weight": float(getattr(args, "loss_hard_kinetics_weight", 0.08)),
        "enable_dres_hard_kinetics": not bool(getattr(args, "disable_dres_hard_kinetics", False)),
        "nmae_weight": 0.0,
        "nrmse_weight": 0.0,
        # Latent alignment DISABLED (0.0): using future NI as a training signal for the
        # latent constitutes training-time leakage. The primary NRMSE loss provides all
        # the gradient the dynamics need; alignment is unnecessary.
        "latent_align_weight": float(getattr(args, "loss_latent_align_weight", 0.0)),
        "norm_align_weight": 0.0,
        # Physics terms ramp start LATER (see losses.py linear_ramp start epochs)
        "energy_weight_base": float(getattr(args, "loss_energy_weight_base", 0.04)),
        "dissip_weight_base": float(getattr(args, "loss_dissip_weight_base", 0.04)),
        # Ramp the metriplectic energy/dissipation physics in early (full by ~epoch 30) so they constrain the
        # long-horizon rollout; the latent complement to the observable-space contractivity penalty.
        "energy_ramp_start": int(getattr(args, "loss_energy_ramp_start", 3)),
        "energy_ramp_end": int(getattr(args, "loss_energy_ramp_end", 30)),
        "dissip_ramp_start": int(getattr(args, "loss_dissip_ramp_start", 3)),
        "dissip_ramp_end": int(getattr(args, "loss_dissip_ramp_end", 30)),
        # Off-diagonal dissipation guardrail: keep cross-mode resonant coupling as a real share of the dissipation
        # operator rather than collapsing to diagonal damping.
        "offdiag_dissip_weight": float(getattr(args, "loss_offdiag_dissip_weight", 0.02)),
        "offdiag_target_frac": float(getattr(args, "offdiag_target_frac", 0.3)),
        "d_res_time_weight": float(getattr(args, "loss_d_res_time_weight", 0.005)),
        "time_freq_weight": float(getattr(args, "loss_time_freq_weight", 0.005)),
        "physics_weight_base": float(getattr(args, "loss_physics_weight_base", 0.03)),
        "sut_weight_base": float(getattr(args, "loss_sut_weight_base", 0.01)),
        "sut_ramp_start": int(getattr(args, "loss_sut_ramp_start", 60)),
        "sut_ramp_end": int(getattr(args, "loss_sut_ramp_end", 220)),
        "snt_weight_base": float(getattr(args, "loss_snt_weight_base", 0.01)),
        "d_res_sparse_weight": float(getattr(args, "loss_d_res_sparse_weight", 0.005)),
        "delta_s_weight": float(getattr(args, "loss_delta_s_weight", 0.005)),
        "jarzy_weight_base": float(getattr(args, "loss_jarzy_weight_base", 0.01)),
        # Train and eval both operate in normalized space (zero-mean, unit-std). In physical space the NI magnitude
        # (~1e19) makes the scale-invariant term contribute negligible gradient; normalized space balances gradients
        # and makes NRMSE a bounded relative error. Eval scores in normalized space so the metrics match.
        "use_denormalized_data_loss": False,
        "normalization_stats": normalization_stats,
        "profile_order": PROFILE_ORDER,
    }
    if _is_baseline:
        # Baselines have no RMMD physics structure -> judge them on the DATA loss only (fair).
        for _k in ("energy_weight_base", "dissip_weight_base", "offdiag_dissip_weight",
                   "d_res_time_weight", "time_freq_weight", "physics_weight_base", "sut_weight_base",
                   "snt_weight_base", "d_res_sparse_weight", "delta_s_weight", "jarzy_weight_base",
                   "hard_kinetics_weight"):
            loss_config[_k] = 0.0
        loss_config["enable_dres_hard_kinetics"] = False
        logger.info("BASELINE (%s): physics loss weights zeroed -> data loss only", _model_type)
    logger.info("Loss config (key weights): %s", {
        k: v for k, v in loss_config.items()
        if k not in ("normalization_stats", "profile_order")
    })
    loss_fn = RMMDLossFunction(model, config=loss_config)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Curriculum: start at frontier=1, advance gradually through TRAINING_FRONTIERS as
    # the model masters each horizon (decoupled from the sparse REPORT horizons).
    # Optional frontier cap (fast/ablation protocol): walk the curriculum only up to --max-frontier.
    _max_frontier = int(getattr(args, "max_frontier", 0) or 0)
    _frontiers = TRAINING_FRONTIERS
    if _max_frontier > 0:
        _frontiers = tuple(f for f in TRAINING_FRONTIERS if f <= _max_frontier) or (1,)
        logger.info("Curriculum capped at T_frontier<=%d -> %s", _max_frontier, _frontiers)
    scheduler = CurriculumScheduler(frontiers=_frontiers)
    logger.info(
        "Curriculum: gradual frontier ladder %s (report horizons=%s)",
        list(TRAINING_FRONTIERS), list(DIRECT_COMPACT_HORIZONS),
    )

    warmup_epochs = int(getattr(args, "rmmd_warmup_epochs", 0))
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ni_nrmse": [],
        "best_val_loss": float("inf"),
        "best_ni_nrmse": float("inf"),
        "best_epoch": 0,
        "horizons": [],
    }
    patience_counter = 0
    frontier_best_nrmse = float("inf")

    for epoch in range(args.epochs):
        max_time_step = scheduler.current_frontier
        rollout_mode = "checkpoint/y_seq" if max_time_step <= 1 else "unit-step/ni_traj"
        logger.info(
            "\nEpoch %d/%d [frontier_idx=%d/%d, T_frontier=%d, rollout=%s, held=%d]",
            epoch + 1,
            args.epochs,
            scheduler.idx,
            len(scheduler.frontiers) - 1,
            max_time_step,
            rollout_mode,
            scheduler.epochs_at_frontier,
        )

        _set_rmmd_trainable_stage(model, train_all=(epoch >= warmup_epochs))

        train_ds = CompactRolloutDataset(
            train_payload, max_time_step, normalization_stats=normalization_stats
        )
        val_ds = CompactRolloutDataset(
            val_payload, max_time_step, normalization_stats=normalization_stats
        )
        if getattr(args, "max_train_shots", None):
            train_ds = Subset(train_ds, list(range(min(args.max_train_shots, len(train_ds)))))
        if getattr(args, "max_val_shots", None):
            val_ds = Subset(val_ds, list(range(min(args.max_val_shots, len(val_ds)))))

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
            collate_fn=_compact_rollout_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
            collate_fn=_compact_rollout_collate,
        )

        gamma = scheduler.get_gamma(epoch, args.epochs)
        logger.info("  gamma=%.3f train_batches=%d val_batches=%d", gamma, len(train_loader), len(val_loader))

        # Rollout-stability knobs (defaults are the strong-stability config): early-step anchoring keeps the one-step
        # map sharp, and a short BPTT window + drift noise + the contractivity penalty stop near-term error compounding.
        anchor_w = float(getattr(args, "rollout_anchor_weight", 12.0))
        anchor_tau = float(getattr(args, "rollout_anchor_tau", 1.5))
        tbptt = int(getattr(args, "rollout_tbptt_steps", 4))
        noise_std = float(getattr(args, "rollout_state_noise", 0.01))

        # Contractivity (Jacobian) penalty weight, ramped in over early epochs so the
        # model first locks onto the data, then is driven non-expansive.
        jac_base = float(getattr(args, "loss_jacobian_weight", 0.5))
        jac_start = int(getattr(args, "jacobian_ramp_start", 2))
        jac_end = int(getattr(args, "jacobian_ramp_end", 20))
        if epoch <= jac_start:
            jac_ramp = 0.0
        elif epoch >= jac_end:
            jac_ramp = 1.0
        else:
            jac_ramp = (epoch - jac_start) / max(jac_end - jac_start, 1)
        jac_weight = jac_base * jac_ramp
        jac_target = float(getattr(args, "jacobian_target_gain", 1.0))
        jac_max_steps = int(getattr(args, "jacobian_max_steps", 6))
        jac_probes = int(getattr(args, "jacobian_probes", 1))

        # (A) tendency + conservation weights. Tendency ramps in early so the increment
        # gets supervised once the state fit is established; conservation is constant.
        tend_base = float(getattr(args, "loss_tendency_weight", 0.05))
        tend_start = int(getattr(args, "tendency_ramp_start", 2))
        tend_end = int(getattr(args, "tendency_ramp_end", 12))
        if epoch <= tend_start:
            tend_ramp = 0.0
        elif epoch >= tend_end:
            tend_ramp = 1.0
        else:
            tend_ramp = (epoch - tend_start) / max(tend_end - tend_start, 1)
        tendency_weight = tend_base * tend_ramp
        conservation_weight = float(getattr(args, "loss_conservation_weight", 0.05))
        cons_volume_mode = str(getattr(args, "conservation_volume_mode", "radial"))
        direct_pred = bool(getattr(args, "direct_prediction", False))
        if _is_baseline:
            # Pure data-driven baselines: no Jacobian/tendency/conservation physics regularizers.
            jac_weight = 0.0
            tendency_weight = 0.0
            conservation_weight = 0.0

        train_loss = _train_epoch_with_curriculum(
            model, train_loader, optimizer, loss_fn,
            args.device, epoch, args.epochs, gamma, normalization_stats,
            max_time_step=max_time_step,
            curriculum_frontier=max_time_step,
            step_anchor_weight=anchor_w, step_anchor_tau=anchor_tau,
            tbptt_steps=tbptt, state_noise_std=noise_std,
            jacobian_weight=jac_weight, jacobian_target=jac_target,
            jac_max_steps=jac_max_steps, jac_probes=jac_probes,
            tendency_weight=tendency_weight, conservation_weight=conservation_weight,
            cons_volume_mode=cons_volume_mode, direct_prediction=direct_pred,
        )
        val_loss, val_ni_nrmse, val_report_nrmse = _validate_epoch_with_curriculum(
            model, val_loader, loss_fn,
            args.device, epoch, args.epochs, gamma, normalization_stats,
            max_time_step=max_time_step,
            curriculum_frontier=max_time_step,
            step_anchor_weight=anchor_w, step_anchor_tau=anchor_tau,
            tbptt_steps=tbptt, state_noise_std=0.0,
            tendency_weight=tendency_weight, conservation_weight=conservation_weight,
            cons_volume_mode=cons_volume_mode, direct_prediction=direct_pred,
        )

        report_str = " ".join(
            f"T{h}={val_report_nrmse.get(h, float('nan')):.4f}"
            for h in _REPORT_HORIZONS
            if h <= max_time_step and val_report_nrmse.get(h, float("nan")) == val_report_nrmse.get(h, float("nan"))
        )
        # Persistence baseline NI(t)=NI(0) over this validation pass, then reset accumulator.
        persist_str = " ".join(
            f"T{h}={sum(_PERSIST_ACC[h]) / len(_PERSIST_ACC[h]):.4f}"
            for h in _REPORT_HORIZONS
            if h <= max_time_step and _PERSIST_ACC.get(h)
        )
        _PERSIST_ACC.clear()
        logger.info(
            "  train_loss=%.6f  val_loss=%.6f  val_NI_NRMSE@T%d=%.4f  best=%.4f  [%s]",
            train_loss, val_loss, max_time_step, val_ni_nrmse, history["best_ni_nrmse"], report_str,
        )
        if persist_str:
            logger.info("  persistence NI(t)=NI(0) baseline:  [%s]   (model must be << this)", persist_str)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ni_nrmse"].append(val_ni_nrmse)
        history["horizons"].append(scheduler.current_frontier)

        checkpoint_obj = {
            "model_state": _checkpoint_safe_value(model.state_dict()),
            "normalization_stats": normalization_stats,
            "config": {
                "state_dim": model_state_dim,
                "latent_dim": model_latent_dim,
                "latent_profile": model_latent_profile,
                "latent_geom": model_latent_geom,
                "machine_embedding_dim": model_machine_embedding_dim,
                "n_harmonics": model_n_harmonics,
                "machine_names": sorted(set(train_machines)),
                "compact_ni_geom": True,
                "epoch": epoch,
                # so eval can rebuild the EXACT model (baseline type + which ablation)
                "model_type": _model_type,
                "use_transport_step": _use_transport,
                "ablate_drivers": _abl_drivers,
                "ablate_geometry": _abl_geometry,
                "ablate_dres": _abl_dres,
                "drivergate": _drivergate,
                "ablate_skip": _abl_skip,
                "hybrid_skip_hidden": int(getattr(args, "hybrid_skip_hidden", 512)),
                "n_drivers": N_DRIVERS,
                "baseline_latent_dim": int(getattr(args, "baseline_latent_dim", 128)),
                "n_params": _n_params,
            },
        }
        _torch_save_checkpoint(checkpoint_obj, checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt")

        # With a gradual curriculum, NRMSE rises when the frontier grows, so 'best' is tracked per frontier (otherwise
        # the frontier=1 best would never be beaten and training would early-stop).
        scheduler.epochs_at_frontier += 1
        improved = val_ni_nrmse < frontier_best_nrmse - 1e-4
        if improved:
            frontier_best_nrmse = val_ni_nrmse
            patience_counter = 0
            # checkpoint_best.pt tracks the best model at the LARGEST frontier reached
            # (the most capable long-horizon model — what production/eval should load).
            _torch_save_checkpoint(checkpoint_obj, checkpoint_dir / "checkpoint_best.pt")
            if val_ni_nrmse < history["best_ni_nrmse"]:
                history["best_ni_nrmse"] = val_ni_nrmse
                history["best_val_loss"] = val_loss
                history["best_epoch"] = epoch
            logger.info(
                "  -> new best @ T_frontier=%d (NRMSE=%.4f)", max_time_step, val_ni_nrmse
            )
        else:
            patience_counter += 1

        # Curriculum advancement: master the current frontier (NRMSE below threshold,
        # after a minimum hold) OR force-advance after max_hold epochs so we never get
        # permanently stuck on one horizon.
        # Advance only on MASTERY (NRMSE < threshold).  Default threshold == the T20
        # target (0.05): do not move to a longer, harder horizon until the current one
        # is actually solved, otherwise error explodes and never recovers.  max_hold is
        # generous so we hold long enough for the contractivity term to take effect.
        advance_threshold = float(getattr(args, "curriculum_advance_threshold", 0.05))
        min_hold = int(getattr(args, "curriculum_min_hold_epochs", 2))
        max_hold = int(getattr(args, "curriculum_max_hold_epochs", 25))
        mastered = (val_ni_nrmse < advance_threshold) and (scheduler.epochs_at_frontier >= min_hold)
        forced = scheduler.epochs_at_frontier >= max_hold
        if (mastered or forced) and not scheduler.at_last:
            old_frontier = scheduler.current_frontier
            scheduler.advance()
            frontier_best_nrmse = float("inf")  # reset per-frontier best for the new horizon
            patience_counter = 0
            logger.info(
                "  Curriculum advanced: T_frontier %d -> %d (%s, NRMSE=%.4f, held=%d epochs)",
                old_frontier, scheduler.current_frontier,
                "mastered" if mastered else "forced (max_hold)",
                val_ni_nrmse, scheduler.epochs_at_frontier,
            )
        elif not scheduler.at_last:
            logger.info(
                "  Curriculum HELD at T_frontier=%d (NRMSE=%.4f >= %.3f, held=%d/%d)",
                scheduler.current_frontier, val_ni_nrmse, advance_threshold,
                scheduler.epochs_at_frontier, max_hold,
            )

        # Only early-stop once we are at the FINAL frontier (otherwise we'd quit before
        # ever training the long horizons).
        if scheduler.at_last and patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d (final frontier, patience exhausted)", epoch + 1)
            break

    with open(checkpoint_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(
        "Training complete. Best NI_NRMSE=%.4f  val_loss=%.6f  at epoch %d",
        history["best_ni_nrmse"], history["best_val_loss"], history["best_epoch"],
    )


# ============================================================================
# EVAL COMMAND
# ============================================================================

@torch.no_grad()
def eval_command(args) -> None:
    logger.info("=" * 80)
    logger.info("STRONG-RMMD: Compact NI + Geometry Evaluation")
    logger.info("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data_path = Path(getattr(args, "compact_test_data", args.test_data))
    payload = _load_phase0_dataset(test_data_path)

    # Load checkpoint
    ck = _torch_load_checkpoint_any(Path(args.checkpoint), map_location="cpu")
    if isinstance(ck, dict) and "model_state" in ck:
        state_dict = ck["model_state"]
        normalization_stats = ck.get("normalization_stats", {})
        config = ck.get("config", {})
        model_machine_names = config.get("machine_names")
        inferred_dims = _infer_model_dimensions(state_dict)
        state_dim = int(config.get("state_dim", inferred_dims["state_dim"]))
        latent_dim = int(config.get("latent_dim", inferred_dims["latent_dim"]))
        latent_profile = int(config.get("latent_profile", inferred_dims["latent_profile"]))
        latent_geom = int(config.get("latent_geom", inferred_dims["latent_geom"]))
        machine_embedding_dim = int(config.get("machine_embedding_dim", inferred_dims["machine_embedding_dim"]))
        n_harmonics = int(config.get("n_harmonics", inferred_dims["n_harmonics"]))
    else:
        state_dict = ck
        normalization_stats = {}
        model_machine_names = None
        inferred_dims = _infer_model_dimensions(state_dict)
        state_dim = inferred_dims["state_dim"]
        latent_dim = inferred_dims["latent_dim"]
        latent_profile = inferred_dims["latent_profile"]
        latent_geom = inferred_dims["latent_geom"]
        machine_embedding_dim = inferred_dims["machine_embedding_dim"]
        n_harmonics = inferred_dims["n_harmonics"]

    if not normalization_stats:
        normalization_stats = _ensure_normalization_stats(test_data_path, checkpoint_dir=None, require=False)

    if model_machine_names is None:
        model_machine_names = sorted(set(_collect_machine_names(payload)))

    _cfg = config if isinstance(ck, dict) else {}
    model = _make_model(
        model_machine_names,
        state_dim=int(state_dim),
        latent_dim=int(latent_dim),
        latent_profile=int(latent_profile),
        latent_geom=int(latent_geom),
        machine_embedding_dim=int(machine_embedding_dim),
        n_harmonics=int(n_harmonics),
        use_transport_step=bool(_cfg.get("use_transport_step", True)),
        ablate_drivers=bool(_cfg.get("ablate_drivers", False)),
        ablate_geometry=bool(_cfg.get("ablate_geometry", False)),
        ablate_dres=bool(_cfg.get("ablate_dres", False)),
        drivergate=bool(_cfg.get("drivergate", False)),
        model_type=str(_cfg.get("model_type", "rmmd")),
        baseline_latent_dim=int(_cfg.get("baseline_latent_dim", 128)),
        ablate_skip=bool(_cfg.get("ablate_skip", False)),
        hybrid_skip_hidden=int(_cfg.get("hybrid_skip_hidden", 512)),
    ).to(args.device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        logger.warning("Strict checkpoint load failed (%s); loading compatible tensors only", exc)
        _load_compatible_state_dict(model, state_dict)
    model.eval()

    horizons = list(DIRECT_COMPACT_HORIZONS)
    max_horizon = max(horizons) if horizons else 0

    # Build eval dataset with full normalization stats and the full report horizon span
    view = CompactRolloutDataset(payload, max(DIRECT_COMPACT_HORIZONS), normalization_stats=normalization_stats)
    limit = min(len(view), args.max_shots) if args.max_shots is not None else len(view)
    logger.info("Eval: device=%s shots=%d/%d horizons=%s", args.device, limit, len(view), horizons[:8])

    profile_metrics: Dict[int, Dict[str, List[float]]] = {h: {"nrmse": [], "nmae": []} for h in horizons}
    geometry_metrics: Dict[int, Dict[str, List[float]]] = {h: {"nrmse": [], "nmae": []} for h in horizons}

    for idx in range(limit):
        if args.log_every > 0 and (idx == 0 or (idx + 1) % args.log_every == 0 or (idx + 1) == limit):
            logger.info("Evaluating shot %d/%d (autoregressive rollout)", idx + 1, limit)

        sample = view.get_sample(idx)
        ni_seq = sample.get("y_seq")
        geom_seq_sample = sample.get("geom_seq")
        target_steps = sample.get("target_steps")
        target_mask = sample.get("target_mask")
        if not isinstance(ni_seq, torch.Tensor) or not isinstance(geom_seq_sample, torch.Tensor):
            continue
        if ni_seq.shape[0] < 1:
            continue

        machine = str(sample.get("machine", "UNKNOWN"))

        limiter_geometry = sample.get("limiter_geometry_tensor", torch.zeros(40, 66))
        if not isinstance(limiter_geometry, torch.Tensor):
            limiter_geometry = torch.as_tensor(limiter_geometry, dtype=torch.float32)

        pre_shot_context = sample.get("pre_shot_context", torch.zeros(COMPACT_PRE_SHOT_CONTEXT_DIM))
        if not isinstance(pre_shot_context, torch.Tensor):
            pre_shot_context = torch.as_tensor(pre_shot_context, dtype=torch.float32)

        ni_t0 = sample.get("ni_t0", torch.zeros(40))
        if not isinstance(ni_t0, torch.Tensor):
            ni_t0 = torch.zeros(40, dtype=torch.float32)

        geom_t0 = sample.get("geom_t0", torch.zeros(40, 66))
        if not isinstance(geom_t0, torch.Tensor):
            geom_t0 = torch.zeros(40, 66, dtype=torch.float32)

        target_steps_t = torch.as_tensor(target_steps, dtype=torch.int64) if not isinstance(target_steps, torch.Tensor) else target_steps
        target_mask_t = (
            torch.as_tensor(target_mask, dtype=torch.bool) if target_mask is not None
            else torch.ones(ni_seq.shape[0], dtype=torch.bool)
        )
        if not target_mask_t.any():
            continue

        ni_traj = sample.get("ni_traj")
        geom_traj = sample.get("geom_traj")
        if not isinstance(ni_traj, torch.Tensor) or int(ni_traj.shape[0]) < 1:
            continue
        if not isinstance(geom_traj, torch.Tensor):
            geom_traj = torch.zeros(ni_traj.shape[0], 40, 66)

        # Drivers (known actuator program) for this shot, if present — fed per step so eval
        # matches the driver-trained model (NOT leakage; actuators are planned ahead).
        drivers_traj_eval = sample.get("drivers_traj")
        if drivers_traj_eval is not None and not isinstance(drivers_traj_eval, torch.Tensor):
            drivers_traj_eval = torch.as_tensor(np.asarray(drivers_traj_eval), dtype=torch.float32)

        # Unit-step rollout (dt=1); record preds only at report horizons 1,20,100,...
        ni_preds, geom_preds = _rollout_compact_shot_to_checkpoints(
            model,
            ni_t0,
            geom_t0,
            pre_shot_context,
            limiter_geometry,
            ni_traj,
            geom_traj,
            machine,
            sample.get("pre_shot_scalars", {}),
            args.device,
            normalization_stats,
            max_time_step=max_horizon,
            drivers_traj=drivers_traj_eval,
        )

        # Score at report horizons using dense traj ground truth (ni_traj[h-1] = state at time h).
        for horizon_value in horizons:
            if horizon_value not in ni_preds or horizon_value > int(ni_traj.shape[0]):
                continue
            pred_ni = ni_preds[horizon_value]
            target_ni = ni_traj[horizon_value - 1]
            nrmse, nmae = _normalized_rmse_mae(pred_ni.numpy(), target_ni.numpy())
            profile_metrics[horizon_value]["nrmse"].append(nrmse)
            profile_metrics[horizon_value]["nmae"].append(nmae)

            if horizon_value in geom_preds and horizon_value <= int(geom_traj.shape[0]):
                pred_geom = geom_preds[horizon_value]
                target_geom = geom_traj[horizon_value - 1]
                geom_nrmse, geom_nmae = _normalized_rmse_mae(
                    pred_geom.reshape(-1).numpy(),
                    target_geom.reshape(-1).numpy(),
                )
                geometry_metrics[horizon_value]["nrmse"].append(geom_nrmse)
                geometry_metrics[horizon_value]["nmae"].append(geom_nmae)

    results: Dict = {"profiles": {}, "geometry": {}}
    for h in horizons:
        nrmse_arr = np.array(profile_metrics[h]["nrmse"], dtype=np.float64)
        nmae_arr = np.array(profile_metrics[h]["nmae"], dtype=np.float64)
        results["profiles"][str(h)] = {
            "nrmse_mean": float(np.mean(nrmse_arr)) if len(nrmse_arr) else None,
            "nrmse_median": float(np.median(nrmse_arr)) if len(nrmse_arr) else None,
            "nmae_mean": float(np.mean(nmae_arr)) if len(nmae_arr) else None,
            "nmae_median": float(np.median(nmae_arr)) if len(nmae_arr) else None,
            "n_shots": int(len(nrmse_arr)),
        }
        geom_nrmse_arr = np.array(geometry_metrics[h]["nrmse"], dtype=np.float64)
        geom_nmae_arr = np.array(geometry_metrics[h]["nmae"], dtype=np.float64)
        results["geometry"][str(h)] = {
            "nrmse_mean": float(np.mean(geom_nrmse_arr)) if len(geom_nrmse_arr) else None,
            "nrmse_median": float(np.median(geom_nrmse_arr)) if len(geom_nrmse_arr) else None,
            "nmae_mean": float(np.mean(geom_nmae_arr)) if len(geom_nmae_arr) else None,
            "nmae_median": float(np.median(geom_nmae_arr)) if len(geom_nmae_arr) else None,
            "n_shots": int(len(geom_nrmse_arr)),
        }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "test_data": str(test_data_path),
        "max_shots": args.max_shots,
        "horizons": horizons,
        "results": results,
    }
    report_path = output_dir / "eval_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation complete: %s", report_path)
    print(json.dumps(results, indent=2))


# ============================================================================
# TEST COMMAND (quick smoke-test)
# ============================================================================

def test_command(args) -> None:
    logger.info("=" * 80)
    logger.info("STRONG-RMMD: Quick Integration Test")
    logger.info("=" * 80)

    train_payload = _load_phase0_dataset(Path(args.train_data))
    view = Phase0DatasetView(train_payload)
    logger.info("Loaded payload with %d shots", len(view))

    normalization_stats = _ensure_normalization_stats(Path(args.train_data), checkpoint_dir=None, require=False)
    ds = CompactRolloutDataset(train_payload, max_time=max(DIRECT_COMPACT_HORIZONS), normalization_stats=normalization_stats)
    sample = ds[0]
    assert sample["y_seq"].shape[0] >= 1, f"y_seq too short: {sample['y_seq'].shape}"
    assert sample["ni_t0"].shape == (40,), f"ni_t0 shape mismatch: {sample['ni_t0'].shape}"
    assert sample["geom_t0"].shape == (40, 66), f"geom_t0 shape mismatch: {sample['geom_t0'].shape}"
    assert sample["ni_traj"].shape[0] >= 1, f"ni_traj missing or empty: {sample.get('ni_traj')}"
    assert "pre_shot_context" in sample, "pre_shot_context missing from sample"
    assert "limiter_geometry_tensor" in sample, "limiter_geometry_tensor missing from sample"
    logger.info(
        "Quick test sample shapes: y_seq=%s ni_t0=%s geom_t0=%s pre_shot=%s",
        tuple(sample["y_seq"].shape),
        tuple(sample["ni_t0"].shape),
        tuple(sample["geom_t0"].shape),
        tuple(sample["pre_shot_context"].shape),
    )
    ni_nonzero = bool(sample["ni_t0"].abs().max().item() > 1e-6)
    logger.info("ni_t0 non-zero: %s (False means state_t0 missing from dataset)", ni_nonzero)
    logger.info("QUICK TEST PASSED")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="STRONG-RMMD compact NI + geometry training and evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    train_parser = subparsers.add_parser("train", help="Train compact RMMD with curriculum")
    train_parser.add_argument("--train-data", default="/scratch/gpfs/USER/strong_rmmd/phase0/dataset_train.pt")
    train_parser.add_argument("--val-data", default="/scratch/gpfs/USER/strong_rmmd/phase0/dataset_val.pt")
    train_parser.add_argument("--compact-train-data", default="/scratch/gpfs/USER/strong_rmmd/data_build/dataset_train_compact.pt")
    train_parser.add_argument("--compact-val-data", default="/scratch/gpfs/USER/strong_rmmd/data_build/dataset_val_compact.pt")
    train_parser.add_argument("--epochs", type=int, default=120)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=5e-5)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--checkpoint-dir", default="/scratch/gpfs/USER/models/rmmd_final")
    train_parser.add_argument("--latent-dim", type=int, default=256)
    train_parser.add_argument("--latent-profile", type=int, default=96)
    train_parser.add_argument("--latent-geom", type=int, default=96)
    train_parser.add_argument("--machine-embedding-dim", type=int, default=24)
    train_parser.add_argument("--rmmd-warmup-epochs", type=int, default=0)
    train_parser.add_argument("--patience", type=int, default=20)
    # Gradual curriculum controls (frontier ladder TRAINING_FRONTIERS).
    train_parser.add_argument("--curriculum-advance-threshold", type=float, default=0.05,
                              help="Advance to next frontier when frontier NRMSE < this.")
    train_parser.add_argument("--curriculum-min-hold-epochs", type=int, default=1,
                              help="Minimum epochs to spend at a frontier before advancing.")
    train_parser.add_argument("--curriculum-max-hold-epochs", type=int, default=12,
                              help="Force-advance after this many epochs at one frontier.")
    # Early-step anchoring for the unit-step rollout loss (prevents identity collapse).
    train_parser.add_argument("--rollout-anchor-weight", type=float, default=4.0,
                              help="Extra loss weight a on step 1: w_k = 1 + a*exp(-(k-1)/tau).")
    train_parser.add_argument("--rollout-anchor-tau", type=float, default=2.0,
                              help="Decay constant tau for early-step anchoring.")
    train_parser.add_argument("--rollout-tbptt-steps", type=int, default=1,
                              help="Detach the fed-forward state every N steps. 1 = pure "
                                   "per-step flow-map training (decouples T1 accuracy from "
                                   "long-horizon gradients; recommended). 0 = full BPTT.")
    train_parser.add_argument("--rollout-state-noise", type=float, default=0.0,
                              help="Std of Gaussian drift noise added to the detached rollout "
                                   "state during training (e.g. 0.01-0.02 hardens against "
                                   "compounding). 0 disables.")
    train_parser.add_argument("--gamma-init", type=float, default=0.98)
    train_parser.add_argument("--max-train-shots", type=int, default=None)
    train_parser.add_argument("--max-val-shots", type=int, default=None)
    # Loss weights
    train_parser.add_argument("--loss-kinetics-weight", type=float, default=0.75)
    train_parser.add_argument("--loss-geometry-weight", type=float, default=0.20)
    train_parser.add_argument("--loss-kinetics-cons-weight", type=float, default=0.05)
    train_parser.add_argument("--loss-profile-bounds-weight", type=float, default=0.01)
    train_parser.add_argument("--loss-hard-kinetics-weight", type=float, default=0.08)
    train_parser.add_argument("--disable-dres-hard-kinetics", action="store_true")
    train_parser.add_argument("--loss-latent-align-weight", type=float, default=0.05)
    train_parser.add_argument("--loss-energy-weight-base", type=float, default=0.04)
    train_parser.add_argument("--loss-dissip-weight-base", type=float, default=0.04)
    train_parser.add_argument("--loss-d-res-time-weight", type=float, default=0.005)
    train_parser.add_argument("--loss-time-freq-weight", type=float, default=0.005)
    train_parser.add_argument("--loss-physics-weight-base", type=float, default=0.03)
    train_parser.add_argument("--loss-sut-weight-base", type=float, default=0.01)
    train_parser.add_argument("--loss-snt-weight-base", type=float, default=0.01)
    train_parser.add_argument("--loss-d-res-sparse-weight", type=float, default=0.005)
    train_parser.add_argument("--loss-delta-s-weight", type=float, default=0.005)
    train_parser.add_argument("--loss-jarzy-weight-base", type=float, default=0.01)
    # Compact aux loss
    train_parser.add_argument("--compact-long-aux-weight", type=float, default=0.10)
    train_parser.add_argument("--compact-long-horizon-weight", type=float, default=0.5)
    train_parser.add_argument("--compact-short-horizon-weight", type=float, default=0.5)
    train_parser.add_argument("--compact-short-horizon-scale", type=float, default=30.0)
    train_parser.add_argument("--compact-long-geom-weight", type=float, default=0.5)
    train_parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # ---- eval ----
    eval_parser = subparsers.add_parser("eval", help="Evaluate compact RMMD")
    eval_parser.add_argument("--checkpoint", default="/scratch/gpfs/USER/models/rmmd_final/checkpoint_best.pt")
    eval_parser.add_argument("--test-data", default="/scratch/gpfs/USER/strong_rmmd/phase0/dataset_test.pt")
    eval_parser.add_argument("--compact-test-data", default="/scratch/gpfs/USER/strong_rmmd/data_build/dataset_test_compact.pt")
    eval_parser.add_argument("--output-dir", default="/scratch/gpfs/USER/strong_rmmd/eval_results_rmmd")
    eval_parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), choices=["cpu", "cuda"])
    eval_parser.add_argument("--max-shots", type=int, default=1000)
    eval_parser.add_argument("--log-every", type=int, default=25)

    # ---- test ----
    test_parser = subparsers.add_parser("test", help="Quick smoke test")
    test_parser.add_argument("--train-data", default="/scratch/gpfs/USER/strong_rmmd/data_build/dataset_train_compact.pt")

    args = parser.parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "eval":
        eval_command(args)
    elif args.command == "test":
        test_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
