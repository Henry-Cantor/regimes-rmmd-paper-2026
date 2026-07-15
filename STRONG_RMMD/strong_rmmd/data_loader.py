"""Multi-machine CDF loader for STRONG-RMMD.

This module reuses the tested DGKNet Phase-0 extractor where possible, while
providing machine-aware bookkeeping and light-weight validation helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import netCDF4 as nc

from dgknet_baseline.phases.phase0_data_pipeline import CDFVariableExtractor, _infer_machine_from_path
from strong_rmmd.config import MACHINES


@dataclass
class MachineShot:
    machine_name: str
    shot_id: str
    cdf_path: str
    profiles: Dict[str, np.ndarray]
    metadata: Dict[str, object]
    geometry: Dict[str, np.ndarray]


class MultiMachineDataLoader:
    """Load CDF shots across machines with machine-aware metadata."""

    def __init__(self, cdf_root_dir: Optional[str] = None, limiter_reference_path: Optional[str] = None):
        self.cdf_root_dir = Path(cdf_root_dir) if cdf_root_dir else None
        self.limiter_reference_path = Path(limiter_reference_path) if limiter_reference_path else None

    def _machine_root(self, machine_name: str) -> Path:
        if self.cdf_root_dir is None:
            spec = MACHINES[machine_name]
            return Path(spec.cdf_root_dir)
        return self.cdf_root_dir / machine_name

    def discover_cdfs(self, machine_name: str) -> List[Path]:
        root = self._machine_root(machine_name)
        if not root.exists():
            return []
        return sorted(root.rglob('*.CDF')) + sorted(root.rglob('*.cdf'))

    def verify_cdf_integrity(self, cdf_path: Path) -> Tuple[bool, str]:
        try:
            with nc.Dataset(str(cdf_path)) as dataset:
                if len(dataset.dimensions) == 0:
                    return False, 'no_dimensions'
                if 'TIME3' not in dataset.variables and 'TIME' not in dataset.variables:
                    return False, 'missing_time_axis'
            return True, 'ok'
        except Exception as exc:
            return False, str(exc)

    def extract_geometry(self, cdf_path: Path) -> Dict[str, np.ndarray]:
        extractor = CDFVariableExtractor(str(cdf_path), verbose=False, limiter_reference_path=str(self.limiter_reference_path) if self.limiter_reference_path else None)
        try:
            return {
                'plasma': extractor.extract_geometry_tensor_t0(),
                'limiter': extractor.extract_limiter_geometry_tensor_t0(),
            }
        finally:
            extractor.close()

    def normalize_profiles(self, profiles: Dict[str, np.ndarray], machine_name: str) -> Dict[str, np.ndarray]:
        normalized = {}
        for key, arr in profiles.items():
            data = np.asarray(arr, dtype=np.float32)
            minimum = float(np.nanmin(data)) if data.size else 0.0
            maximum = float(np.nanmax(data)) if data.size else 0.0
            scale = maximum - minimum
            normalized[key] = (data - minimum) / scale if scale > 1e-12 else np.zeros_like(data)
        return normalized

    def split_heating_regime(self, shots: List[MachineShot], pinj_threshold: float = 1e6) -> Tuple[List[MachineShot], List[MachineShot]]:
        nbi_shots: List[MachineShot] = []
        ohmic_shots: List[MachineShot] = []
        for shot in shots:
            pinj = shot.metadata.get('PINJ') or shot.metadata.get('PHEAT_IN') or 0.0
            try:
                pinj_val = float(pinj)
            except Exception:
                pinj_val = 0.0
            if pinj_val >= pinj_threshold:
                nbi_shots.append(shot)
            else:
                ohmic_shots.append(shot)
        return nbi_shots, ohmic_shots

    def load_machine_shots(self, machine_name: str, n_shots: Optional[int] = None) -> List[MachineShot]:
        shots: List[MachineShot] = []
        for cdf_path in self.discover_cdfs(machine_name):
            if n_shots is not None and len(shots) >= n_shots:
                break
            ok, reason = self.verify_cdf_integrity(cdf_path)
            if not ok:
                continue
            extractor = CDFVariableExtractor(str(cdf_path), verbose=False, limiter_reference_path=str(self.limiter_reference_path) if self.limiter_reference_path else None)
            try:
                state = extractor.extract_full_state_vector_t0()
                pre_shot = state.get('pre_shot_inputs', {})
                scalar_meta = dict(pre_shot.get('scalars', {}))
                scalar_meta['machine_name'] = machine_name
                scalar_meta['cdf_path'] = str(cdf_path)
                geometry = self.extract_geometry(cdf_path)
                shots.append(MachineShot(
                    machine_name=machine_name,
                    shot_id=cdf_path.stem,
                    cdf_path=str(cdf_path),
                    profiles=self.normalize_profiles(state.get('kinetic_profiles', {}), machine_name),
                    metadata=scalar_meta,
                    geometry=geometry,
                ))
            finally:
                extractor.close()
        return shots
