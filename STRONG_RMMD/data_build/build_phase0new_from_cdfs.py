#!/usr/bin/env python3
"""Build CDF-backed data_build compact datasets for the five main machines.

This script reads raw TRANSP CDFs, extracts the richer pre-shot context now
provided by the canonical Phase-0 extractor, builds the direct NI + geometry
targets, and writes per-machine as well as combined train/val/test payloads.
Shots with shorter trajectories only keep the horizons they can actually reach.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dgknet_baseline.phases.phase0_multicdf import build_sample
from dgknet_baseline.phases.phase0_data_pipeline import PlasmaVariables

# Sparse report/score horizons. Training rollouts advance 1 unit-step at a time (dt=1)
# and supervise every step using the full ni_traj / geom_traj series.
DIRECT_COMPACT_HORIZONS: Tuple[int, ...] = (1, 20, 100, 200, 500, 1000)
# EAST is the holdout for zero-shot geometry extrapolation: its heating mix resembles the training
# machines, so a zero-shot result is attributable to geometry rather than a heating-mix confound.
# Build EAST separately (--machines EAST --output-root ..._east) for evaluation.
DEFAULT_MACHINES: Tuple[str, ...] = ("HL2A", "NSTX", "D3D", "KSTR", "CMOD")
DEFAULT_CDF_ROOT = Path("/scratch/gpfs/USER/cdf")
DEFAULT_OUTPUT_ROOT = Path("/scratch/gpfs/USER/strong_rmmd/data_build")
DEFAULT_SEED = 1729


@dataclass
class MachineSplit:
    train: List[Dict]
    val: List[Dict]
    test: List[Dict]


def _normalize_with_stats(array: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    mean = stats["mean"]
    std = stats["std"]
    std = np.where(std < 1e-6, 1.0, std)
    return (array - mean) / std


def _signed_log1p_context(array: np.ndarray, size: int = 256) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    output = np.zeros(size, dtype=np.float32)
    count = min(values.size, size)
    if count > 0:
        output[:count] = values[:count]
    output = np.sign(output) * np.log1p(np.abs(output))
    output = np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
    return output


def _safe_signed_log1p(value: float) -> float:
    return float(np.sign(value) * np.log1p(np.abs(value)))


def _resample_1d(values: np.ndarray, n_points: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(n_points, dtype=np.float32)
    if values.size == 1:
        return np.full(n_points, float(values[0]), dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, num=values.size, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, num=n_points, dtype=np.float32)
    return np.interp(target_x, source_x, values).astype(np.float32)


def _append_section(features: List[float], values: np.ndarray, *, resample_to: int | None = None) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if resample_to is None:
        for value in flat:
            features.append(_safe_signed_log1p(float(value)))
        return

    padded = np.zeros(resample_to, dtype=np.float32)
    count = min(flat.size, resample_to)
    padded[:count] = flat[:count]
    for value in padded:
        features.append(_safe_signed_log1p(float(value)))


def _build_structured_pre_shot_context(pre_shot_inputs: Dict, size: int = 1280) -> Tuple[np.ndarray, Dict[str, List[str]]]:
    scalars = pre_shot_inputs.get("scalars", {}) if isinstance(pre_shot_inputs, dict) else {}
    profiles = pre_shot_inputs.get("profiles", {}) if isinstance(pre_shot_inputs, dict) else {}
    shape_params = pre_shot_inputs.get("shape_params", {}) if isinstance(pre_shot_inputs, dict) else {}
    control_arrays = pre_shot_inputs.get("control_arrays", {}) if isinstance(pre_shot_inputs, dict) else {}

    features: List[float] = []
    layout: Dict[str, List[str]] = {"scalars": [], "profiles": [], "shapes": [], "controls": []}

    for key in PlasmaVariables.SAFE_PRE_SHOT_SCALAR_KEYS:
        value = scalars.get(key)
        features.append(_safe_signed_log1p(float(value)) if value is not None else 0.0)
        layout["scalars"].append(key)

    for key in list(PlasmaVariables.PRE_SHOT_PROFILE_INPUTS.keys()):
        profile = np.asarray(profiles.get(key), dtype=np.float32).reshape(-1) if key in profiles else np.zeros(0, dtype=np.float32)
        _append_section(features, profile, resample_to=40)
        _append_section(features, _resample_1d(profile, 20), resample_to=20)
        if profile.size == 0:
            features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            stats = [float(np.mean(profile)), float(np.std(profile)), float(profile[0]), float(profile[-1])]
            features.extend([_safe_signed_log1p(stat) for stat in stats])
        layout["profiles"].append(key)

    for key in list(PlasmaVariables.SHAPE_PARAMS):
        shape = np.asarray(shape_params.get(key), dtype=np.float32).reshape(-1) if key in shape_params else np.zeros(0, dtype=np.float32)
        _append_section(features, shape, resample_to=40)
        _append_section(features, _resample_1d(shape, 20), resample_to=20)
        if shape.size == 0:
            features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            stats = [float(np.mean(shape)), float(np.std(shape)), float(shape[0]), float(shape[-1])]
            features.extend([_safe_signed_log1p(stat) for stat in stats])
        layout["shapes"].append(key)

    for key in ("LPED", "CPED"):
        control = np.asarray(control_arrays.get(key), dtype=np.float32).reshape(-1) if key in control_arrays else np.zeros(0, dtype=np.float32)
        _append_section(features, control, resample_to=32)
        _append_section(features, _resample_1d(control, 16), resample_to=16)
        if control.size == 0:
            features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            stats = [float(np.mean(control)), float(np.std(control)), float(control[0]), float(control[-1])]
            features.extend([_safe_signed_log1p(stat) for stat in stats])
        layout["controls"].append(key)

    context = np.asarray(features, dtype=np.float32)
    if context.shape[0] < size:
        padded = np.zeros(size, dtype=np.float32)
        padded[: context.shape[0]] = context
        context = padded
    return context[:size], layout


def _pad_or_trim_1d(array: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.size >= size:
        return array[:size]
    output = np.zeros(size, dtype=np.float32)
    output[: array.size] = array
    return output


def _pad_or_trim_2d(array: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    array = np.asarray(array, dtype=np.float32)
    output = np.zeros((rows, cols), dtype=np.float32)
    if array.size == 0:
        return output
    if array.ndim == 1:
        flat = _pad_or_trim_1d(array, rows * cols)
        return flat.reshape(rows, cols)
    copy_rows = min(rows, array.shape[0])
    copy_cols = min(cols, array.shape[1])
    output[:copy_rows, :copy_cols] = array[:copy_rows, :copy_cols]
    return output


def _compute_stats(samples: Sequence[Dict], key: str) -> Dict[str, np.ndarray]:
    arrays = [np.asarray(sample[key], dtype=np.float32) for sample in samples if key in sample]
    if not arrays:
        raise ValueError(f"No samples available for stats key '{key}'")
    stacked = np.concatenate(arrays, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def _direct_horizons_for_length(n_steps: int) -> List[int]:
    return [horizon for horizon in DIRECT_COMPACT_HORIZONS if horizon <= n_steps]


def _build_dense_trajectory_arrays(
    state_trajectory: Sequence[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-timestep NI and geometry from CDF trajectory (physical times 1..T).

    trajectory[k] is the state at time index k+1 (see _build_direct_targets index=h-1).
    """
    ni_rows: List[np.ndarray] = []
    geom_rows: List[np.ndarray] = []
    for state in state_trajectory:
        kinetic = state.get("kinetic_profiles", {}) if isinstance(state, dict) else {}
        ni = kinetic.get("NI") if isinstance(kinetic, dict) else None
        geom = state.get("geometry_tensor") if isinstance(state, dict) else None
        ni_rows.append(_pad_or_trim_1d(np.asarray(ni, dtype=np.float32), 40) if ni is not None else np.zeros(40, dtype=np.float32))
        geom_rows.append(
            _pad_or_trim_2d(np.asarray(geom, dtype=np.float32), (40, 66))
            if geom is not None
            else np.zeros((40, 66), dtype=np.float32)
        )
    if not ni_rows:
        return np.zeros((0, 40), dtype=np.float32), np.zeros((0, 40, 66), dtype=np.float32)
    return np.stack(ni_rows, axis=0).astype(np.float32), np.stack(geom_rows, axis=0).astype(np.float32)


# Time-resolved exogenous drivers (actuators) -> drivers_traj [T, N_DRIVERS_BUILD]. Known control inputs
# at each step (not leakage), which make the transport coefficients drive-responsive. Only exogenous
# actuators here -- never state diagnostics (no TAUEA/BETAT/stored energy, which are consequences of NI).
N_DRIVERS_BUILD = 8
# Of this dataset's per-timestep global scalars, only PINJ and PCUR are clean exogenous actuators; the
# rest are consequences of the predicted state or redundant with the geometry input, so feeding them at
# the target time would leak the answer. Gas/fuelling is usually a profile in TRANSP, not a scalar.
DRIVER_ACTUATOR_KEYS: List[str] = [
    "PINJ",      # NBI injected power (heating + beam particle source)
    "PCUR",      # plasma current (programmed/feedback-controlled)
]
# Per-channel fallback keys for machines that name the actuator differently. KSTR/EAST have no 'PINJ'
# scalar; their NBI lives in beam-power components (PBE/PBI absorbed power, PBTH thermalization), mapped
# into the same channel so the learned NBI response transfers zero-shot. Each alternative is a list of
# keys to sum; the first with any hit wins. Keys are searched as scalars first, then as profiles.
DRIVER_ACTUATOR_FALLBACKS: Dict[str, List[List[str]]] = {
    "PINJ": [["PBE", "PBI"], ["PBTH"], ["PBEAM"], ["PNBI"]],
}
# Profile-form drivers: each group = (vars to sum, channel label), volume-reduced into one driver channel.
# Vars are searched across all state sub-dicts; absent -> 0. Completes the exogenous actuator set: NBI,
# ECH, ICRF, LH heating, and gas fuelling.
DRIVER_SOURCE_GROUPS: List[Tuple[List[str], str]] = [
    (["SESGF"],        "gas"),    # gas-flow ionization source (fuelling)
    (["PECH"],         "ECH"),    # electron-cyclotron heating
    (["PRFE", "PRFI"], "ICRF"),   # ion-cyclotron heating (electron+ion deposition, summed)
    (["PLH"],          "LH"),     # lower-hybrid heating
]


def _search_state_profile(state: Dict, var: str):
    """Find a profile variable in any dict-valued sub-entry of the state (schema-agnostic)."""
    if not isinstance(state, dict):
        return None
    for sub in state.values():
        if isinstance(sub, dict) and sub.get(var) is not None:
            return sub[var]
    return None


def _driver_channel_names() -> List[str]:
    names = list(DRIVER_ACTUATOR_KEYS) + [label for _, label in DRIVER_SOURCE_GROUPS]
    names = names[:N_DRIVERS_BUILD]
    return names + [f"pad{i}" for i in range(N_DRIVERS_BUILD - len(names))]


# ---------------------------------------------------------------------------------------
# PROVENANCE FILTER: keep ONLY confirmed-interpretive runs (drop predictive AND unverified) so the
# dataset cleanly mirrors real measured data. Populated from --provenance-csv in main().
_PROV_VERDICT: Dict[str, str] = {}   # runid -> "interpretive"/"predictive"; empty = filter off


def _norm_runid(s: str) -> str:
    return re.sub(r"TR$", "", str(s).strip(), flags=re.IGNORECASE).upper()


def _load_provenance(csv_path) -> Dict[str, str]:
    import csv as _csv
    out: Dict[str, str] = {}
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            v = str(row.get("verdict", "")).strip().lower()
            out[_norm_runid(row.get("runid", ""))] = "predictive" if v.startswith("pred") else "interpretive"
    return out


def _state_scalar_lookup(state: Dict) -> Dict:
    """Per-timestep scalar dict, tolerant to schema (global_scalars / scalars / pre_shot)."""
    if not isinstance(state, dict):
        return {}
    for key in ("global_scalars", "scalars"):
        d = state.get(key)
        if isinstance(d, dict) and d:
            return d
    psi = state.get("pre_shot_inputs", {})
    if isinstance(psi, dict):
        s = psi.get("scalars", {})
        if isinstance(s, dict) and s:
            return s
    return {}


def _build_drivers_trajectory(state_trajectory: Sequence[Dict]) -> Tuple[np.ndarray, int]:
    """Return (drivers [T, N_DRIVERS_BUILD], n_nonzero_channels). signed-log1p compressed
    (power/current magnitudes are huge); missing keys -> 0.0 so the model degrades gracefully.
    n_nonzero_channels lets the caller WARN if a shot found no real drivers (schema mismatch)."""
    keys = DRIVER_ACTUATOR_KEYS

    def _actuator_value(state: Dict, d: Dict, key: str) -> float:
        """Scalar actuator with per-machine fallbacks: primary key as scalar, then each
        fallback alternative (sum of its keys; scalar lookup first, then profile volume-sum)."""
        if isinstance(d, dict) and d.get(key) is not None:
            return float(d[key])
        for alt in DRIVER_ACTUATOR_FALLBACKS.get(key, []):
            total, found = 0.0, False
            for k in alt:
                if isinstance(d, dict) and d.get(k) is not None:
                    total += float(d[k]); found = True
                else:
                    prof = _search_state_profile(state, k)
                    if prof is not None and np.asarray(prof).size > 0:
                        total += float(np.nansum(np.asarray(prof, dtype=np.float64))); found = True
            if found:
                return total
        return 0.0

    rows: List[List[float]] = []
    for state in state_trajectory:
        d = _state_scalar_lookup(state)
        # scalar actuators (PINJ, PCUR, ...) with fallbacks (KSTR/EAST beam keys)
        row = []
        for k in keys:
            v = _actuator_value(state, d, k)
            row.append(_safe_signed_log1p(v) if v != 0.0 else 0.0)
        # volume-reduced heating/fuelling profiles (sum over radius ~ total power/rate per group)
        for varlist, _label in DRIVER_SOURCE_GROUPS:
            total = 0.0
            found = False
            for var in varlist:
                prof = _search_state_profile(state, var)
                if prof is not None and np.asarray(prof).size > 0:
                    total += float(np.nansum(np.asarray(prof, dtype=np.float64)))
                    found = True
            row.append(_safe_signed_log1p(total) if found else 0.0)
        # fit to exactly N_DRIVERS_BUILD channels (consistent ordering, pad/trim)
        row = (row + [0.0] * N_DRIVERS_BUILD)[:N_DRIVERS_BUILD]
        rows.append(row)
    if not rows:
        return np.zeros((0, N_DRIVERS_BUILD), dtype=np.float32), 0
    arr = np.asarray(rows, dtype=np.float32)
    n_nonzero = int(np.count_nonzero(np.any(arr != 0.0, axis=0)))
    return arr, n_nonzero


def _build_direct_targets(state_trajectory: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizons = _direct_horizons_for_length(len(state_trajectory))
    y_seq: List[np.ndarray] = []
    geom_seq: List[np.ndarray] = []
    target_mask: List[bool] = []
    target_steps: List[int] = []

    if not horizons:
        return (
            np.zeros((0, 40), dtype=np.float32),
            np.zeros((0, 40, 66), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=np.int32),
        )

    for horizon in horizons:
        index = horizon - 1
        state_t = state_trajectory[index]
        kinetic_profiles = state_t.get("kinetic_profiles", {})
        ni = kinetic_profiles.get("NI")
        geom = state_t.get("geometry_tensor")
        if ni is None or geom is None:
            continue
        y_seq.append(_pad_or_trim_1d(ni, 40))
        geom_seq.append(_pad_or_trim_2d(geom, (40, 66)))
        target_mask.append(True)
        target_steps.append(horizon)

    return (
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(geom_seq, dtype=np.float32),
        np.asarray(target_mask, dtype=bool),
        np.asarray(target_steps, dtype=np.int32),
    )


def _candidate_limiter_paths(cdf_root: Path) -> List[Optional[Path]]:
    return [
        cdf_root.parent / "strong_rmmd" / "data_io" / "processed_data" / "limiter_reference.npz",
        cdf_root.parent / "strong_rmmd" / "data_io" / "limiters" / "limiter_reference.npz",
        cdf_root.parent / "dgknet_baseline" / "processed_data" / "limiter_reference.npz",
        None,
    ]


def _resolve_limiter_reference(cdf_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit
    for candidate in _candidate_limiter_paths(cdf_root):
        if candidate is None:
            continue
        if candidate.exists():
            return candidate
    return None


def _build_compact_sample(sample: Dict, machine: str, cdf_path: Path) -> Optional[Dict]:
    state_t0 = sample.get("state_t0", {})
    state_trajectory = sample.get("state_trajectory", [])
    if not state_t0 or not state_trajectory:
        return None

    y_seq, geom_seq, target_mask, target_steps = _build_direct_targets(state_trajectory)
    ni_traj, geom_traj = _build_dense_trajectory_arrays(state_trajectory)
    drivers_traj, n_driver_ch = _build_drivers_trajectory(state_trajectory)

    structured_context, context_layout = _build_structured_pre_shot_context(state_t0.get("pre_shot_inputs", {}))

    compact_sample = {
        "machine": machine,
        "cdf_path": str(cdf_path),
        "shot_name": cdf_path.stem,
        "state_t0": state_t0,
        "pre_shot_context": structured_context,
        "pre_shot_inputs": state_t0.get("pre_shot_inputs", {}),
        "limiter_geometry_tensor": np.asarray(state_t0.get("limiter_geometry_tensor", np.zeros((40, 66), dtype=np.float32)), dtype=np.float32),
        "target_steps": target_steps,
        "target_mask": target_mask,
        "y_seq": y_seq,
        "geom_seq": geom_seq,
        "ni_traj": ni_traj,
        "geom_traj": geom_traj,
        "drivers_traj": drivers_traj,
        "n_driver_channels_found": n_driver_ch,
        "n_time_steps": len(state_trajectory),
        "n_direct_targets": int(len(target_steps)),
        "pre_shot_context_layout": context_layout,
    }
    return compact_sample


def _split_paths(paths: Sequence[Path], seed: int, train_frac: float, val_frac: float) -> MachineSplit:
    ordered = list(paths)
    random.Random(seed).shuffle(ordered)
    total = len(ordered)
    n_train = int(round(total * train_frac))
    n_val = int(round(total * val_frac))
    n_train = min(n_train, total)
    n_val = min(n_val, max(0, total - n_train))
    n_test = max(0, total - n_train - n_val)

    train = ordered[:n_train]
    val = ordered[n_train:n_train + n_val]
    test = ordered[n_train + n_val:n_train + n_val + n_test]
    return MachineSplit(train=train, val=val, test=test)


def _load_machine_cdfs(cdf_root: Path, machine: str) -> List[Path]:
    machine_dir = cdf_root / machine
    if not machine_dir.exists():
        return []
    return sorted([path for path in machine_dir.glob("*.CDF") if path.is_file()])


def _build_split_payload(
    machine: str,
    split_name: str,
    cdf_paths: Sequence[Path],
    limiter_reference_path: Optional[Path],
) -> List[Dict]:
    payload: List[Dict] = []
    total = len(cdf_paths)
    for index, cdf_path in enumerate(cdf_paths, start=1):
        start_time = time.time()
        print(f"[{machine}/{split_name}] {index}/{total} {cdf_path.name} ...", flush=True)
        try:
            sample = build_sample(str(cdf_path), limiter_reference_path=str(limiter_reference_path) if limiter_reference_path else None)
        except Exception as exc:
            elapsed = time.time() - start_time
            print(f"[{machine}/{split_name}] Skipping {cdf_path.name} after {elapsed:.1f}s: {exc}", flush=True)
            continue
        compact_sample = _build_compact_sample(sample, machine=machine, cdf_path=cdf_path)
        if compact_sample is None:
            elapsed = time.time() - start_time
            print(f"[{machine}/{split_name}] Skipping {cdf_path.name} after {elapsed:.1f}s: incomplete sample", flush=True)
            continue
        elapsed = time.time() - start_time
        print(f"[{machine}/{split_name}] done {cdf_path.name} in {elapsed:.1f}s", flush=True)
        payload.append(compact_sample)
    return payload


def _dense_traj_arrays_for_sample(sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw (T,40) / (T,40,66) trajectories, building from state_trajectory if needed."""
    ni_traj = sample.get("ni_traj")
    geom_traj = sample.get("geom_traj")
    if ni_traj is not None and np.asarray(ni_traj).size > 0:
        ni_arr = np.asarray(ni_traj, dtype=np.float32)
        if ni_arr.ndim == 2 and ni_arr.shape[1] == 40:
            geom_arr = (
                np.asarray(geom_traj, dtype=np.float32)
                if geom_traj is not None and np.asarray(geom_traj).size > 0
                else np.zeros((ni_arr.shape[0], 40, 66), dtype=np.float32)
            )
            if geom_arr.ndim != 3 or geom_arr.shape[0] != ni_arr.shape[0]:
                geom_arr = np.zeros((ni_arr.shape[0], 40, 66), dtype=np.float32)
            return ni_arr, geom_arr

    state_trajectory = sample.get("state_trajectory") or []
    if state_trajectory:
        return _build_dense_trajectory_arrays(state_trajectory)

    return np.zeros((0, 40), dtype=np.float32), np.zeros((0, 40, 66), dtype=np.float32)


def _strip_heavy_fields_for_storage(sample: Dict) -> Dict:
    """Drop full state_trajectory from saved .pt (dense ni_traj/geom_traj are authoritative)."""
    item = dict(sample)
    item.pop("state_trajectory", None)
    return item


def _validate_dense_traj_split(samples: List[Dict], split_name: str, machine: str = "") -> None:
    """Fail fast at build time if unit-step targets are missing."""
    if not samples:
        return
    lengths = [int(np.asarray(s.get("ni_traj", [])).shape[0]) for s in samples]
    ok = sum(1 for t in lengths if t >= 1)
    max_t = max(lengths) if lengths else 0
    prefix = f"[{machine}/{split_name}]" if machine else f"[{split_name}]"
    if ok == 0:
        raise RuntimeError(
            f"{prefix} No samples have ni_traj length >= 1 after build/normalize. "
            "Unit-step training cannot run. Check CDF extraction (state_trajectory)."
        )
    frac = ok / len(samples)
    if frac < 0.99:
        raise RuntimeError(
            f"{prefix} Only {ok}/{len(samples)} ({frac:.1%}) samples have dense ni_traj; "
            f"max length={max_t}. Expected >=99% with valid trajectories."
        )
    print(
        f"{prefix} dense traj OK: {ok}/{len(samples)} shots, max_T={max_t}, "
        f"median_T={int(np.median(lengths))}",
        flush=True,
    )
    _report_driver_coverage(samples, prefix)


def _report_driver_coverage(samples: List[Dict], prefix: str) -> None:
    """Print per-channel DRIVER hit-rate so step (1) — 'is the particle source actually in the
    drivers?' — is a single glance. Hit-rate = fraction of shots where that channel is ever
    nonzero. A near-0 channel means that actuator key was NOT found (fix DRIVER_ACTUATOR_KEYS)."""
    names = _driver_channel_names()
    n = len(samples)
    if n == 0:
        return
    hits = np.zeros(N_DRIVERS_BUILD, dtype=np.float64)
    for s in samples:
        d = np.asarray(s.get("drivers_traj", np.zeros((0, N_DRIVERS_BUILD))), dtype=np.float32)
        if d.size and d.ndim == 2 and d.shape[1] >= N_DRIVERS_BUILD:
            hits += (np.abs(d).sum(axis=0) > 0).astype(np.float64)[:N_DRIVERS_BUILD]
    rate = hits / n
    cov = "  ".join(f"{k}={r:.0%}" for k, r in zip(names, rate) if not k.startswith("pad"))
    print(f"{prefix} DRIVER coverage (hit-rate over shots): {cov}", flush=True)
    if float(rate.max()) < 0.01:
        print(
            f"{prefix} *** WARNING: NO driver channels found — drivers will be all-zero and the "
            f"model degrades to static-context behaviour. The key names in DRIVER_ACTUATOR_KEYS do "
            f"not match your per-timestep scalar dict. Introspect with:\n"
            f"    from dgknet_baseline.phases.phase0_multicdf import build_sample\n"
            f"    st = build_sample('<one_cdf>')['state_trajectory'][0]\n"
            f"    print(sorted((st.get('global_scalars') or st.get('scalars') or {{}}).keys()))\n"
            f"  then set DRIVER_ACTUATOR_KEYS to the real names (esp. the GAS/FUELLING channel).",
            flush=True,
        )
    else:
        empty = [lab for _, lab in DRIVER_SOURCE_GROUPS if lab in names and rate[names.index(lab)] < 0.01]
        if empty:
            print(
                f"{prefix} *** NOTE: heating/fuelling channels empty: {empty}. Add the underlying "
                f"vars (SESGF/PECH/PRFE/PRFI/PLH) to phase0_multicdf.build_sample. 'gas' is the top "
                f"DENSITY driver; 'ICRF' is essential for CMOD (ICRF-only).",
                flush=True,
            )


def _normalize_payload(samples: List[Dict], y_stats: Dict[str, np.ndarray], geom_stats: Dict[str, np.ndarray]) -> List[Dict]:
    """Normalize y_seq and geom_seq targets, and also normalize ni_t0/geom_t0 from
    state_t0 using the SAME stats.  This is critical: the autoregressive rollout feeds
    ni_t0 directly into the model and compares it against y_seq in the loss.  Both must
    be in the same normalized space; using different stats would break the loss signal.
    """
    normalized: List[Dict] = []
    for sample in samples:
        item = _strip_heavy_fields_for_storage(sample)
        item["y_seq"] = _normalize_with_stats(np.asarray(sample["y_seq"], dtype=np.float32), y_stats)
        item["geom_seq"] = _normalize_with_stats(np.asarray(sample["geom_seq"], dtype=np.float32), geom_stats)

        ni_raw, geom_raw = _dense_traj_arrays_for_sample(sample)
        if ni_raw.shape[0] > 0:
            item["ni_traj"] = _normalize_with_stats(ni_raw, y_stats).astype(np.float32)
            item["geom_traj"] = _normalize_with_stats(geom_raw, geom_stats).astype(np.float32)
            item["traj_len"] = int(ni_raw.shape[0])
        else:
            item["ni_traj"] = np.zeros((0, 40), dtype=np.float32)
            item["geom_traj"] = np.zeros((0, 40, 66), dtype=np.float32)
            item["traj_len"] = 0

        # Normalize ni_t0 and geom_t0 with the SAME per-element stats as y_seq/geom_seq.
        # This guarantees model input (ni_t0) and training targets (y_seq) are on the
        # exact same scale, which is essential for the autoregressive rollout loss.
        state_t0 = sample.get("state_t0") or {}
        kinetic = state_t0.get("kinetic_profiles") or {} if isinstance(state_t0, dict) else {}
        ni_raw = kinetic.get("NI") if isinstance(kinetic, dict) else None
        if ni_raw is not None:
            ni_arr = _pad_or_trim_1d(np.asarray(ni_raw, dtype=np.float32), 40)
            item["ni_t0"] = _normalize_with_stats(ni_arr, y_stats).astype(np.float32)
        else:
            item["ni_t0"] = np.zeros(40, dtype=np.float32)

        geom_raw = state_t0.get("geometry_tensor") if isinstance(state_t0, dict) else None
        if geom_raw is not None:
            geom_arr = _pad_or_trim_2d(np.asarray(geom_raw, dtype=np.float32), (40, 66))
            item["geom_t0"] = _normalize_with_stats(geom_arr, geom_stats).astype(np.float32)
        else:
            item["geom_t0"] = np.zeros((40, 66), dtype=np.float32)

        normalized.append(item)
    return normalized


def _build_normalization_stats_obj(y_stats: Dict[str, np.ndarray], geom_stats: Dict[str, np.ndarray]) -> Dict:
    """Build a normalization_stats dict with both scalar and per-element stats.
    The training script uses per-element stats when available (for denormalization,
    omega computation, and eval NRMSE in physical units).
    """
    return {
        "kinetic_profiles.NI": {
            "mean": float(np.mean(y_stats["mean"])),
            "std": float(np.mean(y_stats["std"])),
            "mean_per_element": y_stats["mean"].reshape(-1).tolist(),
            "std_per_element": y_stats["std"].reshape(-1).tolist(),
        },
        "geometry_tensor": {
            "mean": float(np.mean(geom_stats["mean"])),
            "std": float(np.mean(geom_stats["std"])),
            "mean_per_element": geom_stats["mean"].reshape(-1).tolist(),
            "std_per_element": geom_stats["std"].reshape(-1).tolist(),
        },
    }


def _save_payload(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _build_machine_dataset(
    machine: str,
    cdf_root: Path,
    output_root: Path,
    limiter_reference_path: Optional[Path],
    seed: int,
    train_frac: float,
    val_frac: float,
    holdout: bool = False,
) -> Dict[str, List[Dict]]:
    cdf_paths = _load_machine_cdfs(cdf_root, machine)
    if not cdf_paths:
        print(f"[{machine}] No CDF files found under {cdf_root / machine}")
        return {"train": [], "val": [], "test": []}

    # PROVENANCE FILTER: keep ONLY confirmed-interpretive shots (drop predictive + unverified).
    if _PROV_VERDICT:
        before = len(cdf_paths)
        kept, n_pred, n_unver = [], 0, 0
        for p in cdf_paths:
            v = _PROV_VERDICT.get(_norm_runid(p.stem))
            if v == "interpretive":
                kept.append(p)
            elif v == "predictive":
                n_pred += 1
            else:
                n_unver += 1
        cdf_paths = kept
        print(f"[{machine}] provenance: kept {len(cdf_paths)}/{before} interpretive "
              f"(dropped {n_pred} predictive, {n_unver} unverified)", flush=True)

    machine_output = output_root / machine
    if holdout:
        # ZERO-SHOT EXTRAPOLATION SET: NO split — ALL interpretive shots go into one 'test' file,
        # normalized by this machine's OWN stats (consistent with the per-machine normalization
        # used in training). Splitting a never-trained-on machine would just waste eval data.
        train_paths, val_paths, test_paths = [], [], cdf_paths
        print(f"[{machine}] HOLDOUT: all {len(cdf_paths)} shots -> single eval set "
              f"(no split; eval on dataset_test_compact.pt)", flush=True)
    else:
        split = _split_paths(cdf_paths, seed=seed, train_frac=train_frac, val_frac=val_frac)
        train_paths, val_paths, test_paths = split.train, split.val, split.test
        print(f"[{machine}] building {len(cdf_paths)} shots: "
              f"train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}", flush=True)

    train_raw = _build_split_payload(machine, "train", train_paths, limiter_reference_path)
    val_raw = _build_split_payload(machine, "val", val_paths, limiter_reference_path)
    test_raw = _build_split_payload(machine, "test", test_paths, limiter_reference_path)

    # In holdout there is no train split; normalize by this machine's own stats (from test).
    stats_src = train_raw if train_raw else test_raw
    if not stats_src:
        raise ValueError(f"Machine {machine} produced no samples")
    y_stats = _compute_stats(stats_src, "y_seq")
    geom_stats = _compute_stats(stats_src, "geom_seq")

    train_payload = _normalize_payload(train_raw, y_stats, geom_stats)
    val_payload = _normalize_payload(val_raw, y_stats, geom_stats)
    test_payload = _normalize_payload(test_raw, y_stats, geom_stats)
    _validate_dense_traj_split(train_payload, "train", machine=machine)
    _validate_dense_traj_split(val_payload, "val", machine=machine)
    _validate_dense_traj_split(test_payload, "test", machine=machine)

    split_meta = {
        "machine": machine,
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": 1.0 - train_frac - val_frac,
        "source_cdf_root": str(cdf_root),
        "limiter_reference_path": str(limiter_reference_path) if limiter_reference_path else None,
        "direct_horizons": list(DIRECT_COMPACT_HORIZONS),
        "y_stats": {"mean": y_stats["mean"], "std": y_stats["std"]},
        "geom_stats": {"mean": geom_stats["mean"], "std": geom_stats["std"]},
    }

    # Build a normalization_stats object for embedding into payloads.
    # The training script reads this to normalize ni_t0/geom_t0 at load time if the
    # samples do not already contain pre-stored ni_t0/geom_t0 (backward compat).
    norm_stats_obj = _build_normalization_stats_obj(y_stats, geom_stats)

    print(f"[{machine}] Saving per-split compact datasets to {machine_output}", flush=True)
    _save_payload(machine_output / "dataset_train_compact.pt", {
        "data": train_payload, "samples": train_payload,
        "metadata": {**split_meta, "split": "train"},
        "normalization_stats": norm_stats_obj,
    })
    _save_payload(machine_output / "dataset_val_compact.pt", {
        "data": val_payload, "samples": val_payload,
        "metadata": {**split_meta, "split": "val"},
        "normalization_stats": norm_stats_obj,
    })
    _save_payload(machine_output / "dataset_test_compact.pt", {
        "data": test_payload, "samples": test_payload,
        "metadata": {**split_meta, "split": "test"},
        "normalization_stats": norm_stats_obj,
    })

    return {"train": train_payload, "val": val_payload, "test": test_payload}


def _combine_splits(all_machine_payloads: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, List[Dict]]:
    combined = {"train": [], "val": [], "test": []}
    for payload in all_machine_payloads.values():
        for split_name in combined.keys():
            combined[split_name].extend(payload.get(split_name, []))
    return combined


def _save_combined(output_root: Path, combined: Dict[str, List[Dict]], metadata: Dict, norm_stats_by_machine: Dict[str, Dict] | None = None) -> None:
    """Save combined multi-machine compact datasets.

    Each sample already has ni_t0/geom_t0 normalized with its own machine's stats.
    The per-machine normalization_stats are stored in the metadata for reference, but
    there is no single global normalization_stats for the combined payload.  The training
    script handles this by falling back to scalar stats computed from the combined data.
    """
    print(f"[data_build] Saving combined compact datasets to {output_root}", flush=True)
    for split_name, samples in combined.items():
        payload = {
            "data": samples,
            "samples": samples,
            "metadata": {**metadata, "split": split_name},
        }
        if norm_stats_by_machine:
            payload["normalization_stats_by_machine"] = norm_stats_by_machine
        _save_payload(output_root / f"dataset_{split_name}_compact.pt", payload)


def build_phase0new_from_cdfs(
    cdf_root: Path,
    output_root: Path,
    machines: Sequence[str],
    limiter_reference_path: Optional[Path],
    seed: int,
    train_frac: float,
    val_frac: float,
    holdout: bool = False,
) -> Dict[str, Dict[str, List[Dict]]]:
    output_root.mkdir(parents=True, exist_ok=True)
    limiter_reference_path = _resolve_limiter_reference(cdf_root, limiter_reference_path)

    all_machine_payloads: Dict[str, Dict[str, List[Dict]]] = {}
    machine_summaries = []

    for machine in machines:
        print(f"[data_build] Building {machine} from raw CDFs")
        payload = _build_machine_dataset(
            machine=machine,
            cdf_root=cdf_root,
            output_root=output_root,
            limiter_reference_path=limiter_reference_path,
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
            holdout=holdout,
        )
        all_machine_payloads[machine] = payload
        machine_summaries.append(
            {
                "machine": machine,
                "train": len(payload["train"]),
                "val": len(payload["val"]),
                "test": len(payload["test"]),
            }
        )

    combined = _combine_splits(all_machine_payloads)
    combined_meta = {
        "source_cdf_root": str(cdf_root),
        "limiter_reference_path": str(limiter_reference_path) if limiter_reference_path else None,
        "machines": list(machines),
        "direct_horizons": list(DIRECT_COMPACT_HORIZONS),
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": 1.0 - train_frac - val_frac,
        "machine_summaries": machine_summaries,
    }
    # Collect per-machine normalization stats so the combined payload carries them.
    # The training script uses these to handle per-sample denormalization correctly.
    norm_stats_by_machine: Dict[str, Dict] = {}
    for machine in machines:
        machine_output = output_root / machine
        try:
            import torch as _torch
            train_pt = machine_output / "dataset_train_compact.pt"
            if train_pt.exists():
                _tmp = _torch.load(str(train_pt), map_location="cpu")
                if isinstance(_tmp, dict) and "normalization_stats" in _tmp:
                    norm_stats_by_machine[machine] = _tmp["normalization_stats"]
        except Exception:
            pass
    for split_name, split_samples in combined.items():
        _validate_dense_traj_split(split_samples, split_name, machine="combined")

    _save_combined(output_root, combined, combined_meta, norm_stats_by_machine=norm_stats_by_machine or None)

    summary_path = output_root / "phase0new_summary.json"
    summary_path.write_text(json.dumps({"metadata": combined_meta, "machine_summaries": machine_summaries}, indent=2))
    return all_machine_payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdf-root", type=Path, default=DEFAULT_CDF_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limiter-reference", type=Path, default=None)
    parser.add_argument("--machines", nargs="*", default=list(DEFAULT_MACHINES))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--val-frac", type=float, default=0.125)  # test = 1 - 0.75 - 0.125 = 0.125
    parser.add_argument(
        "--provenance-csv", type=Path, default=None,
        help="CSV with columns runid,verdict (from the namelist scan). KEEP-ONLY-INTERPRETIVE: "
             "predictive AND unverified shots are dropped, so the dataset mirrors real measured "
             "data for the extrapolation/SUT claim.",
    )
    parser.add_argument(
        "--holdout", action="store_true",
        help="Zero-shot extrapolation build: NO train/val/test split — ALL interpretive shots of "
             "each --machines go into one file (dataset_test_compact.pt), normalized by that "
             "machine's own stats. Use for the held-out machine (EAST).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError("train-frac + val-frac must be less than 1.0")
    if args.holdout and Path(args.output_root).resolve() == Path(DEFAULT_OUTPUT_ROOT).resolve():
        raise ValueError(
            "--holdout writes ALL shots into dataset_test_compact.pt; pass a SEPARATE --output-root "
            "(e.g. /scratch/gpfs/USER/strong_rmmd/data_build_east) so it does not overwrite the "
            "main training dataset's test split."
        )
    if args.provenance_csv:
        global _PROV_VERDICT
        _PROV_VERDICT = _load_provenance(args.provenance_csv)
        n_int = sum(1 for v in _PROV_VERDICT.values() if v == "interpretive")
        n_pred = sum(1 for v in _PROV_VERDICT.values() if v == "predictive")
        print(f"[provenance] keep-only-interpretive: {n_int} interpretive kept, {n_pred} predictive "
              f"+ all unverified (absent from CSV) dropped (from {args.provenance_csv})", flush=True)

    build_phase0new_from_cdfs(
        cdf_root=args.cdf_root,
        output_root=args.output_root,
        machines=args.machines,
        limiter_reference_path=args.limiter_reference,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        holdout=args.holdout,
    )


if __name__ == "__main__":
    main()
