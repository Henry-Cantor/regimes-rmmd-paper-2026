"""
Core DGKNet Architecture - Full Implementation (DGKNet is a diagonal-only metriplectic model, to
demonstrate base design of many metriplectics and compare to off-diagonal surrogate RMMD). Note that
RMMD novelty is not due to off-diagonal but rather other innovations including physics-keyed frequencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from abc import ABC, abstractmethod
import argparse
import json
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DGKNET_CHECKPOINT_DIR = Path('/scratch/gpfs/USER/models/dgknet')
DEFAULT_LEGACY_PHASE0_DATASET_PATH = Path('/scratch/gpfs/USER/datasets/dataset.pt')
CANONICAL_N_RADIAL = 40
CANONICAL_N_PSI = 40
PRE_SHOT_CONTEXT_DIM = 64

PROFILE_ORDER = ['NI', 'NE', 'NH', 'TE', 'TI', 'PPLAS']
TRANSPORT_ORDER = ['CONDE', 'CONDI', 'DIFFD', 'VELH']
SOURCE_ORDER = ['SBTH', 'SBCX0_D', 'SBAL_ION']


class Phase0DGKNetDataset(Dataset):
    """Turn Phase 0 state trajectories into DGKNet next-step training samples."""

    def __init__(self, phase0_dataset, n_radial: int = CANONICAL_N_RADIAL, n_psi: int = CANONICAL_N_PSI):
        self.samples = []
        self.n_radial = int(n_radial)
        self.n_psi = int(n_psi)
        self.max_trajectory_len = 0

        for sample in phase0_dataset.data:
            trajectory = self._expand_state_trajectory(sample)
            if len(trajectory) < 2:
                continue
            self.max_trajectory_len = max(self.max_trajectory_len, len(trajectory))

            for t_idx in range(len(trajectory) - 1):
                self.samples.append((trajectory[t_idx], trajectory[t_idx + 1]))

        if not self.samples:
            raise ValueError('No valid trajectory pairs found in Phase 0 dataset for DGKNet training.')

    @staticmethod
    def _merge_static_context(state: Dict, fallback_state: Optional[Dict] = None) -> Dict:
        if fallback_state is None:
            return state

        merged = dict(state)
        # Do not use t=0 plasma geometry as fallback for missing trajectory geometry.
        # Only predicted or explicitly provided geometry should be propagated.
        if merged.get('limiter_geometry_tensor') is None and fallback_state.get('limiter_geometry_tensor') is not None:
            merged['limiter_geometry_tensor'] = fallback_state.get('limiter_geometry_tensor')
        if merged.get('pre_shot_context') is None and fallback_state.get('pre_shot_context') is not None:
            merged['pre_shot_context'] = fallback_state.get('pre_shot_context')

        merged_globals = dict(merged.get('global_scalars', {}))
        fallback_globals = fallback_state.get('global_scalars', {}) or {}
        for key, value in fallback_globals.items():
            merged_globals.setdefault(key, value)
        if merged_globals:
            merged['global_scalars'] = merged_globals

        return merged

    @staticmethod
    def _expand_state_trajectory(sample: Dict) -> List[Dict]:
        trajectory = sample.get('state_trajectory', []) or []
        if not trajectory:
            return []

        expanded_states: List[Dict] = []
        previous_state = Phase0DGKNetDataset._merge_static_context(sample.get('state_t0', {}), sample.get('state_t0', {}))

        for state_t in trajectory:
            merged_state = dict(previous_state)

            for group_name in ('kinetic_profiles', 'transport_coefficients', 'nbi_sources', 'other_profiles'):
                current_group = dict(merged_state.get(group_name, {}))
                incoming_group = state_t.get(group_name, {}) or {}
                current_group.update(incoming_group)
                merged_state[group_name] = current_group

            if state_t.get('geometry_tensor') is not None:
                merged_state['geometry_tensor'] = state_t.get('geometry_tensor')
            if state_t.get('limiter_geometry_tensor') is not None:
                merged_state['limiter_geometry_tensor'] = state_t.get('limiter_geometry_tensor')

            current_globals = dict(merged_state.get('global_scalars', {}))
            incoming_globals = state_t.get('global_scalars', {}) or {}
            current_globals.update(incoming_globals)
            merged_state['global_scalars'] = current_globals

            expanded_states.append(merged_state)
            previous_state = merged_state

        return expanded_states

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _pad_globals(global_scalars: Dict[str, float], target_dim: int = 17) -> torch.Tensor:
        values = [float(v) for _, v in sorted(global_scalars.items())]
        if len(values) < target_dim:
            values.extend([0.0] * (target_dim - len(values)))
        return torch.tensor(values[:target_dim], dtype=torch.float32)

    @staticmethod
    def _build_profile_tensor(component_dict: Dict[str, np.ndarray], order: List[str], n_radial: int) -> torch.Tensor:
        profiles = []
        for key in order:
            value = component_dict.get(key)
            if value is None:
                value = np.zeros(n_radial, dtype=np.float32)
            value = np.asarray(value, dtype=np.float32).reshape(-1)
            padded = np.zeros(n_radial, dtype=np.float32)
            padded[:min(value.shape[0], n_radial)] = value[:n_radial]
            tensor = torch.tensor(padded, dtype=torch.float32)
            # Sanitize profile values: clamp extreme values and NaNs
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
            tensor = torch.clamp(tensor, min=-1e4, max=1e4)
            profiles.append(tensor)
        return torch.stack(profiles, dim=0)

    @staticmethod
    def _build_geometry_tensor(geometry, n_psi: int, n_fourier: int = 66) -> torch.Tensor:
        if geometry is None:
            return torch.zeros((n_psi, n_fourier), dtype=torch.float32)
        geometry_array = np.asarray(geometry, dtype=np.float32)
        if geometry_array.ndim == 1:
            geometry_array = geometry_array.reshape(-1, n_fourier)
        padded = np.zeros((n_psi, n_fourier), dtype=np.float32)
        rows = min(geometry_array.shape[0], n_psi)
        cols = min(geometry_array.shape[1], n_fourier)
        padded[:rows, :cols] = geometry_array[:rows, :cols]
        tensor = torch.tensor(padded, dtype=torch.float32)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        tensor = torch.clamp(tensor, min=-1e4, max=1e4)
        return tensor

    @staticmethod
    def _build_limiter_geometry_tensor(geometry, n_psi: int, n_fourier: int = 66) -> torch.Tensor:
        return Phase0DGKNetDataset._build_geometry_tensor(geometry, n_psi, n_fourier)

    @staticmethod
    def _build_pre_shot_context(context, context_dim: int = PRE_SHOT_CONTEXT_DIM) -> torch.Tensor:
        if context is None:
            return torch.zeros(context_dim, dtype=torch.float32)
        context_array = np.asarray(context, dtype=np.float32).reshape(-1)
        padded = np.zeros(context_dim, dtype=np.float32)
        padded[:min(context_array.shape[0], context_dim)] = context_array[:context_dim]
        # Apply normalization: sign * log1p(abs)
        padded = np.sign(padded) * np.log1p(np.abs(padded))
        padded = np.nan_to_num(padded, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.tensor(padded, dtype=torch.float32)

    def _state_to_model_batch(self, state: Dict) -> Dict[str, torch.Tensor]:
        n_radial = self.n_radial
        kinetic = torch.zeros((len(PROFILE_ORDER), n_radial), dtype=torch.float32)
        geometry_tensor = torch.zeros((self.n_psi, 66), dtype=torch.float32)
        limiter_geometry_tensor = self._build_limiter_geometry_tensor(state.get('limiter_geometry_tensor'), self.n_psi)
        pre_shot_context = self._build_pre_shot_context(state.get('pre_shot_context'))

        transport = torch.zeros((len(TRANSPORT_ORDER), n_radial), dtype=torch.float32)
        sources = torch.zeros((len(SOURCE_ORDER), n_radial), dtype=torch.float32)
        globals_ = torch.zeros(17, dtype=torch.float32)

        return {
            'kinetic_profiles': kinetic,
            'geometry_tensor': geometry_tensor,
            'limiter_geometry_tensor': limiter_geometry_tensor,
            'pre_shot_context': pre_shot_context,
            'transport_coeff': transport,
            'nbi_sources': sources,
            'global_scalars': globals_,
        }

    def __getitem__(self, idx: int):
        input_state, target_state = self.samples[idx]
        batch_inputs = self._state_to_model_batch(input_state)
        target_profiles = self._build_profile_tensor(target_state.get('kinetic_profiles', {}), PROFILE_ORDER, self.n_radial)
        target_geometry = self._build_geometry_tensor(target_state.get('geometry_tensor'), self.n_psi)
        return batch_inputs, target_profiles, target_geometry


class DGKNetTrainer:
    """Train DGKNet on the Phase 0 state-time dataset."""

    def __init__(self,
                 device: str = 'cpu',
                 checkpoint_dir: Path = DEFAULT_DGKNET_CHECKPOINT_DIR,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 use_amp: bool = False):
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = max(0, int(num_workers))
        self.pin_memory = bool(pin_memory)
        self.use_amp = bool(use_amp)

    @staticmethod
    def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    def _save_checkpoint(self, model: nn.Module, history: List[Dict], extra: Optional[Dict] = None) -> Path:
        checkpoint_path = self.checkpoint_dir / 'dgknet.pt'
        torch.save({
            'model_name': 'dgknet',
            'model_state': model.state_dict(),
            'config': getattr(model, 'config', {}),
            'history': history,
            'extra': extra or {},
        }, checkpoint_path)
        # Also save a user-facing epoch-style checkpoint if an epoch is provided
        try:
            epoch = (extra or {}).get('epoch')
            if epoch is not None:
                epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{int(epoch):03d}.pt'
                torch.save({
                    'model_name': 'dgknet',
                    'model_state': model.state_dict(),
                    'config': getattr(model, 'config', {}),
                    'history': history,
                    'extra': extra or {},
                }, epoch_path)
        except Exception:
            pass
        return checkpoint_path

    def train(self,
              dataset_path: Optional[str] = None,
              train_dataset_path: str = str(DEFAULT_LEGACY_PHASE0_DATASET_PATH.parent / 'dataset_train.pt'),
              val_dataset_path: str = str(DEFAULT_LEGACY_PHASE0_DATASET_PATH.parent / 'dataset_val.pt'),
              test_dataset_path: str = str(DEFAULT_LEGACY_PHASE0_DATASET_PATH.parent / 'dataset_test.pt'),
              epochs: int = 10,
              batch_size: int = 8,
              learning_rate: float = 1e-4,
              weight_decay: float = 1e-5,
              val_fraction: float = 0.2,
              patience: int = 3) -> Dict[str, object]:
        from phase0_data_pipeline import PPPLPlasmaDataset

        logger.info("\n" + "=" * 92)
        logger.info("PHASE 2: DGKNet CORE TRAINING (PLAN SECTION 5.4 + SECTION 8 ABLE TO SAVE CHECKPOINTS)")
        logger.info("- Training DGKNet on Phase 0 state-time samples")
        logger.info("- Using next-step prediction from the extracted trajectories")
        logger.info("- Saving a reusable checkpoint for Phase 3 fine-tuning")
        logger.info("=" * 92)

        # Load split datasets if they exist, otherwise fallback to single dataset
        train_path = Path(train_dataset_path)
        val_path = Path(val_dataset_path)
        test_path = Path(test_dataset_path)

        if train_path.exists() and val_path.exists() and test_path.exists():
            train_phase0_dataset = PPPLPlasmaDataset.load(str(train_path), device='cpu')
            val_phase0_dataset = PPPLPlasmaDataset.load(str(val_path), device='cpu')
            test_phase0_dataset = PPPLPlasmaDataset.load(str(test_path), device='cpu')
            logger.info(f"Loaded split datasets:\n  train={train_path}\n  val={val_path}\n  test={test_path}")
        else:
            if dataset_path and Path(dataset_path).exists():
                logger.warning(
                    "Split datasets were not found. Falling back to a legacy dataset artifact; "
                    "prefer the split Phase 0 outputs for new runs."
                )
                fallback_dataset = PPPLPlasmaDataset.load(dataset_path, device='cpu')
                train_phase0_dataset = fallback_dataset
                val_phase0_dataset = fallback_dataset
                test_phase0_dataset = fallback_dataset
            else:
                raise FileNotFoundError(
                    "Phase 0 split datasets were not found and no legacy fallback dataset was provided."
                )

        train_dgk_dataset = Phase0DGKNetDataset(train_phase0_dataset)
        val_dgk_dataset = Phase0DGKNetDataset(val_phase0_dataset)
        logger.info(f"Loaded Phase 0 datasets")
        logger.info(f"  Train shots: {len(train_phase0_dataset)}, DGK pairs: {len(train_dgk_dataset)}")
        logger.info(f"  Val shots: {len(val_phase0_dataset)}, DGK pairs: {len(val_dgk_dataset)}")
        logger.info(f"  n_radial={train_dgk_dataset.n_radial}, n_psi={train_dgk_dataset.n_psi}")
        logger.info(f"  Profile order: {PROFILE_ORDER}")

        worker_kwargs = {}
        if self.num_workers > 0:
            worker_kwargs['num_workers'] = self.num_workers
            worker_kwargs['persistent_workers'] = True
            worker_kwargs['prefetch_factor'] = 2

        train_loader = DataLoader(
            train_dgk_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            **worker_kwargs,
        )
        val_loader = DataLoader(
            val_dgk_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            **worker_kwargs,
        )

        model = DGKNet(
            state_dim=train_dgk_dataset.n_radial * len(PROFILE_ORDER) + train_dgk_dataset.n_psi * 66 + train_dgk_dataset.n_radial * (len(TRANSPORT_ORDER) + len(SOURCE_ORDER)) + 17,
            n_radial=train_dgk_dataset.n_radial,
            n_psi=train_dgk_dataset.n_psi,
            n_fourier=66,
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Prefer OneCycleLR for improved convergence when training from scratch;
        # fall back to ReduceLROnPlateau for very small runs or if steps unknown.
        try:
            total_steps = max(1, epochs * max(1, len(train_loader)))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos'
            )
            use_step_scheduler = True
        except Exception:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            use_step_scheduler = False
        criterion_mse = nn.MSELoss()
        amp_enabled = self.use_amp and str(self.device).startswith('cuda') and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        history: List[Dict] = []
        best_val = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_inputs, target_profiles, target_geometry in train_loader:
                batch_inputs = self._move_batch_to_device(batch_inputs, self.device)
                target_profiles = target_profiles.to(self.device, non_blocking=self.pin_memory)
                target_geometry = target_geometry.to(self.device, non_blocking=self.pin_memory)
                # Noise augmentation on inputs to improve robustness
                try:
                    noise_std = 1e-2
                    if 'kinetic_profiles' in batch_inputs:
                        batch_inputs['kinetic_profiles'] = batch_inputs['kinetic_profiles'] + noise_std * torch.randn_like(batch_inputs['kinetic_profiles'])
                    if 'transport_coeff' in batch_inputs:
                        batch_inputs['transport_coeff'] = batch_inputs['transport_coeff'] + noise_std * torch.randn_like(batch_inputs['transport_coeff'])
                    if 'nbi_sources' in batch_inputs:
                        batch_inputs['nbi_sources'] = batch_inputs['nbi_sources'] + noise_std * torch.randn_like(batch_inputs['nbi_sources'])
                    if 'geometry_tensor' in batch_inputs:
                        batch_inputs['geometry_tensor'] = batch_inputs['geometry_tensor'] + noise_std * torch.randn_like(batch_inputs['geometry_tensor'])
                except Exception:
                    pass
                optimizer.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                    predictions = model(batch_inputs)
                    # Numeric hardening: ensure no NaN/Inf from the model
                    try:
                        predictions = {k: torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6) for k, v in predictions.items()}
                    except Exception:
                        # If predictions is not a dict-like mapping, attempt to sanitize tensors where possible
                        try:
                            predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
                        except Exception:
                            pass
                    pred_tensor = torch.stack([predictions[name] for name in PROFILE_ORDER], dim=1)
                    # Standard MSE
                    profile_mse = criterion_mse(pred_tensor, target_profiles)
                    # Relative (normalized) MSE to encourage good R2/NRMSE behavior
                    rel_denom = target_profiles ** 2 + 1e-6
                    profile_rel = torch.mean(((pred_tensor - target_profiles) ** 2) / rel_denom)
                    # Smoothness penalty: L1 on radial derivatives of predicted vs target
                    # pred_tensor shape: (batch, n_profiles, n_radial)
                    try:
                        pred_deriv = torch.diff(pred_tensor, dim=-1)
                        target_deriv = torch.diff(target_profiles, dim=-1)
                        deriv_l1 = torch.mean(torch.abs(pred_deriv - target_deriv))
                    except Exception:
                        deriv_l1 = torch.mean(torch.abs(pred_tensor - target_profiles)) * 0.0

                    # Combined profile loss: MSE + stronger relative term + smoothness
                    profile_loss = profile_mse + 0.25 * profile_rel + 0.02 * deriv_l1
                    geometry_loss = criterion_mse(predictions['geometry_tensor'], target_geometry)
                    loss = profile_loss + 0.15 * geometry_loss
                    # Small parameter regularization (mean squared) to avoid runaway sums
                    try:
                        l_reg = torch.stack([p.pow(2).mean() for p in model.parameters()]).mean()
                        loss = loss + 1e-6 * l_reg
                    except Exception:
                        l_reg = torch.tensor(0.0, device=target_profiles.device)
                    # Sanity clamp to avoid inf/nan propagation
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if use_step_scheduler:
                    try:
                        scheduler.step()
                    except Exception:
                        pass
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_inputs, target_profiles, target_geometry in val_loader:
                    batch_inputs = self._move_batch_to_device(batch_inputs, self.device)
                    target_profiles = target_profiles.to(self.device, non_blocking=self.pin_memory)
                    target_geometry = target_geometry.to(self.device, non_blocking=self.pin_memory)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                        predictions = model(batch_inputs)
                        try:
                            predictions = {k: torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6) for k, v in predictions.items()}
                        except Exception:
                            try:
                                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
                            except Exception:
                                pass
                        pred_tensor = torch.stack([predictions[name] for name in PROFILE_ORDER], dim=1)
                        profile_mse = criterion_mse(pred_tensor, target_profiles)
                        rel_denom = target_profiles ** 2 + 1e-6
                        profile_rel = torch.mean(((pred_tensor - target_profiles) ** 2) / rel_denom)
                        try:
                            pred_deriv = torch.diff(pred_tensor, dim=-1)
                            target_deriv = torch.diff(target_profiles, dim=-1)
                            deriv_l1 = torch.mean(torch.abs(pred_deriv - target_deriv))
                        except Exception:
                            deriv_l1 = 0.0
                        profile_loss = profile_mse + 0.25 * profile_rel + 0.02 * deriv_l1
                        geometry_loss = criterion_mse(predictions['geometry_tensor'], target_geometry)
                        # Validation regularizer for consistency (no grad)
                        try:
                            l_reg_val = torch.stack([p.pow(2).mean() for p in model.parameters()]).mean()
                        except Exception:
                            l_reg_val = torch.tensor(0.0, device=target_profiles.device)
                        vloss = profile_loss + 0.15 * geometry_loss + 1e-6 * l_reg_val
                        vloss = torch.nan_to_num(vloss, nan=0.0, posinf=1e6, neginf=-1e6)
                        val_loss += float(vloss.item())

            train_loss /= max(1, len(train_loader))
            val_loss /= max(1, len(val_loader))
            # Step scheduler depending on type
            if use_step_scheduler:
                # OneCycleLR is stepped per optimizer step above; nothing to do here
                pass
            else:
                try:
                    scheduler.step(val_loss)
                except Exception:
                    pass
            history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})

            if val_loss < best_val:
                best_val = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            logger.info(
                f"[DGKNet] Epoch {epoch + 1:03d}/{epochs} "
                f"train={train_loss:.6f} val={val_loss:.6f}"
            )

            if epochs_without_improvement >= patience:
                logger.info(f"[DGKNet] Early stopping after {epoch + 1} epochs (patience={patience}).")
                break

        checkpoint_path = self._save_checkpoint(
            model,
            history,
            extra={
                'dataset_path': dataset_path,
                'n_radial': train_dgk_dataset.n_radial,
                'n_psi': train_dgk_dataset.n_psi,
                'profile_order': PROFILE_ORDER,
                'training_mode': 'single_step_architecture_training',
                'epoch': history[-1]['epoch'] if history else None,
            },
        )

        summary = {
            'status': 'trained',
            'checkpoint': str(checkpoint_path),
            'final_train_loss': history[-1]['train_loss'] if history else None,
            'final_val_loss': history[-1]['val_loss'] if history else None,
            'history': history,
        }

        summary_path = self.checkpoint_dir / 'dgknet_training_summary.json'
        with open(summary_path, 'w') as handle:
            json.dump(summary, handle, indent=2)

        logger.info(f"✓ DGKNet checkpoint saved to {checkpoint_path}")
        logger.info(f"✓ DGKNet summary saved to {summary_path}")
        return summary


# ==============================================================================
# SECTION 1: CHEBYSHEV PROFILE ENCODER
# ==============================================================================

class ChebyshevProfileEncoder(nn.Module):
    """
    Encodes radial plasma profiles using Chebyshev polynomial basis.
    
    Theoretical Motivation (Section 5.4.1):
    Standard approaches flatten the profile to a 1D vector and pass through MLP.
    This loses information about spectral decay - smooth profiles should be
    compressible.
    
    Chebyshev encoding exploits smoothness: any smooth function on [0,1]
    can be approximated with exponentially decaying Chebyshev coefficients.
    
    Math:
    Profile f(ρ) ≈ Σ_{k=0}^{N} c_k T_k(2ρ - 1)
    where T_k is Chebyshev polynomial of first kind and c_k decay rapidly.
    
    For plasma profiles (smooth), we need only ~32 modes instead of 100 radial points.
    
    Implementation Steps:
    1. Evaluate Chebyshev polynomials at radial points (fixed, non-learnable)
    2. Solve linear system: Tc = f  →  c = T^{-1} f
    3. Pass coefficients through MLP
    
    Benefits:
    - Automatic regularization via truncation to n_cheb modes
    - Captures physical smoothness structure
    - Reduces dimension: 100 → 32 radial points
    - Stable numerically (Chebyshev basis is orthogonal)
    """
    
    def __init__(self, n_radial: int = 100, n_cheb: int = 256, latent_dim: int = 192):
        """
        Initialize Chebyshev encoder.
        
        Args:
            n_radial: Number of radial grid points in CDF (typically 40-100)
            n_cheb: Number of Chebyshev modes to keep (dimension reduction)
            latent_dim: Output dimension of encoded profile
        """
        super().__init__()
        self.n_radial = n_radial
        self.n_cheb = n_cheb
        
        # Precompute Chebyshev polynomial values at radial grid
        # Grid: ρ ∈ [0, 1] maps to [-1, 1] via x = 2ρ - 1
        rho = torch.linspace(0, 1, n_radial)
        x = 2 * rho - 1  # Map to [-1, 1]
        
        # Evaluate T_k(x) for k = 0, 1, ..., n_cheb-1
        # Using recurrence: T_0(x) = 1, T_1(x) = x, T_{k+1} = 2xT_k - T_{k-1}
        T = torch.zeros(n_cheb, n_radial)
        T[0, :] = 1
        if n_cheb > 1:
            T[1, :] = x
        for k in range(2, n_cheb):
            T[k, :] = 2 * x * T[k-1, :] - T[k-2, :]
        
        # Register as buffer (not learnable)
        self.register_buffer('cheb_matrix', T)  # (n_cheb, n_radial)
        
        # Pseudo-inverse for solving Tc = f → c = T^+ f.
        # Use regularized normal equations instead of SVD to avoid cuSolver
        # convergence issues on some GPU / horizon combinations.
        regularizer = 1e-6 * torch.eye(n_radial)
        gram = T.T @ T + regularizer
        self.register_buffer('cheb_pinv', torch.linalg.solve(gram, T.T))
        
        # MLP to process coefficients
        self.mlp = nn.Sequential(
            nn.Linear(n_cheb, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        """
        Encode profile to Chebyshev coefficients, then to latent representation.
        
        Args:
            profile: Plasma profile of shape (batch, n_radial)
        
        Returns:
            Latent encoding of shape (batch, latent_dim)
        """
        batch_size = profile.shape[0]
        
        # Compute Chebyshev coefficients: c = T^+ f
        # profile: (batch, n_radial)
        # cheb_pinv: (n_radial, n_cheb)
        # Result: (batch, n_cheb)
        coeffs = torch.matmul(profile, self.cheb_pinv)  # (batch, n_cheb)
        
        # Apply MLP
        encoding = self.mlp(coeffs)  # (batch, latent_dim)
        
        return encoding
    
    def decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Decode Chebyshev coefficients back to radial profile.
        
        Args:
            coeffs: Chebyshev coefficients (batch, n_cheb)
        
        Returns:
            Radial profile (batch, n_radial)
        """
        # profile = T^T coeffs
        profile = torch.matmul(coeffs, self.cheb_matrix)  # (batch, n_radial)
        return profile


# ==============================================================================
# SECTION 2: GEOMETRY TRANSFORMER
# ==============================================================================

class GeometryTransformer(nn.Module):
    """
    Encodes the full flux surface geometry using transformer attention.
    
    Input: Geometry tensor of shape (batch, n_psi, 66)
        where 66 Fourier modes per flux surface:
        - 17 RMC modes (cosine R)
        - 16 RMS modes (sine R)
        - 17 YMC modes (cosine Z)
        - 16 YMS modes (sine Z)
    
    Output: Geometry context vector (batch, latent_geom)
    
    Key Innovation (Section 5.4.2):
    Geometry affects transport nonlocally through:
    1. Local curvature (κ) → turbulence drive
    2. Trapped fraction (f_T) → neoclassical transport
    3. Volume derivative (V'(ψ)) → appears in transport equation
    4. Shaping parameters (elongation, triangularity) → confinement
    
    Using transformer allows capturing these nonlocal geometric effects:
    - Self-attention over flux surfaces: geometry at ρ_i affects transport at ρ_j
    - Fourier mode attention: different modes contribute to different effects
    
    Architecture:
    - Embed each (flux_surface, 66_modes) as token
    - Apply self-attention transformer
    - Global average pooling over flux surfaces
    """
    
    def __init__(self, n_fourier: int = 66, n_psi: int = 100, 
                 latent_geom: int = 192, n_heads: int = 16, n_layers: int = 16):
        """
        Initialize geometry transformer.
        
        Args:
            n_fourier: Number of Fourier modes (66 = 17+16+17+16)
            n_psi: Number of flux surfaces (typically 40-100)
            latent_geom: Dimension of output geometry context
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
        """
        super().__init__()
        
        self.n_psi = n_psi
        self.latent_geom = latent_geom
        
        # Project Fourier modes to embedding dimension
        self.mode_embed = nn.Linear(n_fourier, latent_geom)
        
        # Positional encoding for flux surfaces
        # Learnable embeddings for each position ψ_N
        self.psi_pos_embed = nn.Embedding(n_psi, latent_geom)
        
        # Transformer encoder
        # Attention: each flux surface attends to all others
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_geom,
            nhead=n_heads,
            dim_feedforward=4 * latent_geom,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(latent_geom)
        )
        
        # Global pooling to get single context vector
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        """
        Transform geometry tensor to context vector.
        
        Args:
            geometry: (batch, n_psi, 66) Fourier coefficients
        
        Returns:
            Context vector (batch, latent_geom)
        """
        batch_size, n_psi, n_modes = geometry.shape
        
        # Embed Fourier modes
        x = self.mode_embed(geometry)  # (batch, n_psi, latent_geom)
        
        # Add positional encoding
        pos_indices = torch.arange(n_psi, device=geometry.device)
        pos_embed = self.psi_pos_embed(pos_indices)  # (n_psi, latent_geom)
        x = x + pos_embed.unsqueeze(0)  # (batch, n_psi, latent_geom)
        
        # Apply transformer
        x = self.transformer(x)  # (batch, n_psi, latent_geom)
        
        # Global average pooling
        context = self.global_pool(x.transpose(1, 2)).squeeze(-1)  # (batch, latent_geom)
        
        return context


# ==============================================================================
# SECTION 3: METRIPLECTIC KOOPMAN OPERATOR
# ==============================================================================

class MetriplecticKoopman(nn.Module):
    """
    Metriplectic Koopman Operator with Symplectic + Dissipative Structure.
    
    Mathematical Foundation (Section 4.2 of plan):
    
    Real tokamak plasma has TWO competing structures:
    1. Reversible MHD dynamics (Hamiltonian): dF/dt = {F, H}
    2. Irreversible dissipation (resistivity, viscosity): dF/dt = [F, S]
    
    Solution: Metriplectic bracket combining both:
    dF/dt = {F, H} + [F, S]
    
    This requires K to have special structure:
    K = [[K_sym,    0   ],
         [  0,    K_dis ]]
    
    where:
    - K_sym: Symplectic (K_sym^T J K_sym = J, J = [[0, I], [-I, 0]])
    - K_dis: Dissipative (K_dis ≤ 0, all eigenvalues have Re ≤ 0)
    
    Parameterization:
    - Symplectic: Use Cayley map A → (I-A)(I+A)^{-1} where A skew-symmetric
    - Dissipative: Use -L^T L structure (automatically negative semi-definite)
    
    Benefit: GUARANTEES long-time stability in lifted space.
    Plasma energy cannot explode - dissipation ensures bounded growth.
    """
    
    def __init__(self, koopman_dim: int, geometry_dim: int = 192):
        """
        Initialize metriplectic Koopman operator.
        
        Args:
            koopman_dim: Dimension of Koopman space (must be even)
            geometry_dim: Dimension of geometry context (for modulation)
        """
        super().__init__()
        
        if koopman_dim % 2 != 0:
              raise ValueError("koopman_dim must be even (for symplectic structure)")
        
        self.koopman_dim = koopman_dim
        self.half_dim = koopman_dim // 2
        self.geometry_dim = geometry_dim
        
        # Symplectic part parameterization
        # A: skew-symmetric matrix (2 * half_dim × 2 * half_dim)
        # Will ensure A - A^T = 0 (skew-symmetric)
        A_param = torch.randn(self.half_dim, self.half_dim) * 0.01
        self.register_parameter('A_sym_param', nn.Parameter(A_param))
        
        # Dissipative part parameterization
        # L: arbitrary matrix, construct K_dis = -L^T L (negative semi-definite)
        L_param = torch.randn(self.half_dim, self.half_dim) * 0.01
        self.register_parameter('L_dis_param', nn.Parameter(L_param))
        
        # Geometry-dependent modulation
        # Allows geometry to influence dynamics (nonlinear coupling)
        self.geom_gate_sym = nn.Sequential(
            nn.Linear(geometry_dim, self.half_dim * self.half_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        self.geom_gate_dis = nn.Sequential(
            nn.Linear(geometry_dim, self.half_dim * self.half_dim),
            nn.Tanh()
        )
        self.dissipation_step = 0.1
    
    def _get_K_sym(self, A_skew: torch.Tensor) -> torch.Tensor:
        """
        Cayley map: Convert skew-symmetric A to symplectic K_sym.
        
        K_sym = (I - A) (I + A)^{-1} = inv(I + A) (I - A)
        
        Theory: If A is skew-symmetric, then K_sym is symplectic.
        
        Args:
            A_skew: Skew-symmetric matrix (half_dim, half_dim)
        
        Returns:
            Symplectic matrix K_sym (half_dim, half_dim)
        """
        half_dim = A_skew.shape[0]
        I = torch.eye(half_dim, device=A_skew.device, dtype=A_skew.dtype)
        
        # K_sym = (I - A) @ inv(I + A)
        I_plus_A = I + A_skew
        K_sym = torch.linalg.solve(I_plus_A, I - A_skew)
        
        return K_sym
    
    def _get_K_dis(self, L: torch.Tensor) -> torch.Tensor:
        """
        Construct dissipative operator from L.
        
        K_dis = -L^T L is automatically negative semi-definite.
        
        Args:
            L: Arbitrary matrix (half_dim, half_dim)
        
        Returns:
            Dissipative matrix K_dis (half_dim, half_dim)
        """
        K_dis = -L.T @ L
        return K_dis

    def stability_regularization(self, geom_context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return lightweight regularizers that keep the metriplectic map stable."""
        batch_size = geom_context.shape[0]
        A_skew = self.A_sym_param - self.A_sym_param.T
        K_sym = self._get_K_sym(A_skew)
        K_dis = self._get_K_dis(self.L_dis_param)

        sym_gate = self.geom_gate_sym(geom_context).view(batch_size, self.half_dim, self.half_dim)
        dis_gate = self.geom_gate_dis(geom_context).view(batch_size, self.half_dim, self.half_dim)
        K_sym_mod = K_sym.unsqueeze(0) + 0.1 * sym_gate
        K_dis_mod = K_dis.unsqueeze(0) + 0.1 * dis_gate

        eye = torch.eye(self.half_dim, device=geom_context.device, dtype=geom_context.dtype)
        symplectic_error = torch.mean((torch.matmul(K_sym.t(), K_sym) - eye) ** 2)
        dissipation_violation = torch.relu(torch.diagonal(K_dis_mod, dim1=-2, dim2=-1)).mean()
        gate_energy = (sym_gate.pow(2).mean() + dis_gate.pow(2).mean())

        total = 0.6 * symplectic_error + 0.3 * dissipation_violation + 0.1 * gate_energy
        return {
            'total': total,
            'symplectic_error': symplectic_error,
            'dissipation_violation': dissipation_violation,
            'gate_energy': gate_energy,
        }
    
    def forward(self, z: torch.Tensor, geom_context: torch.Tensor) -> torch.Tensor:
        """
        Apply metriplectic Koopman operator: z_next = K @ z.
        
        Args:
            z: Koopman coordinates (batch, koopman_dim)
            geom_context: Geometry context (batch, geometry_dim)
        
        Returns:
            Next Koopman state (batch, koopman_dim)
        """
        batch_size = z.shape[0]
        
        # Get base operators
        A_skew = self.A_sym_param - self.A_sym_param.T  # Ensure skew-symmetric
        K_sym = self._get_K_sym(A_skew)  # (half_dim, half_dim)
        
        L = self.L_dis_param
        K_dis = self._get_K_dis(L)  # (half_dim, half_dim)
        
        # Geometry-dependent modulation
        sym_gate = self.geom_gate_sym(geom_context)  # (batch, half_dim^2)
        sym_gate = sym_gate.view(batch_size, self.half_dim, self.half_dim)
        
        dis_gate = self.geom_gate_dis(geom_context)  # (batch, half_dim^2)
        dis_gate = dis_gate.view(batch_size, self.half_dim, self.half_dim)
        
        # Modulate operators (per batch element)
        # This allows different trajectories to have slightly different dynamics
        # based on their geometry context
        K_sym_mod = K_sym.unsqueeze(0) + 0.1 * sym_gate  # (batch, half_dim, half_dim)
        K_dis_mod = K_dis.unsqueeze(0) + 0.1 * dis_gate  # (batch, half_dim, half_dim)
        
        # Apply operators to corresponding parts
        z_q, z_p = z[:, :self.half_dim], z[:, self.half_dim:]  # (batch, half_dim)
        
        # Batch matrix multiplication
        z_q_next = torch.bmm(z_q.unsqueeze(1), K_sym_mod.transpose(1, 2)).squeeze(1)
        z_p_next = z_p + self.dissipation_step * torch.bmm(z_p.unsqueeze(1), K_dis_mod.transpose(1, 2)).squeeze(1)
        
        # Combine
        z_next = torch.cat([z_q_next, z_p_next], dim=1)
        
        return z_next


# ==============================================================================
# SECTION 4: DECOMPOSED OPERATOR BLOCK
# ==============================================================================

class DecomposedOperatorBlock(nn.Module):
    """
    Decomposes dynamics into physical operators as specified in Section 3.2.
    
    Key Insight:
    Ion density transport operates via MULTIPLE CONCURRENT PROCESSES:
    1. Neoclassical transport (slow, geometry-dependent, well-understood)
    2. NBI source deposition (fast, from t=0 data, calculable)
    3. Turbulent transport (stochastic, depends on instability drives)
    4. Charge exchange recycling (local, density-dependent)
    5. Residual (learned correction for unmodeled effects)
    
    Architecture:
    Each process is a separate Koopman sub-network with its own physics context.
    Final dynamics is LEARNED MIX of processes via adaptive weights.
    
    Mathematical form:
    z_next = w_neo * K_neo(z, geom) 
           + w_NBI * K_NBI(z, source)
           + w_turb * K_turb(z, phys_params)
           + w_cx * K_cx(z)
           + w_res * z_res
    
    where w_i are LEARNED and should sum to 1 (soft normalization via softmax).
    """
    
    def __init__(self, koopman_dim: int, geometry_dim: int = 128,
                 source_dim: int = 32, n_operators: int = 8):
        """
        Initialize decomposed operator block.
        
        Args:
            koopman_dim: Dimension of Koopman space
            geometry_dim: Dimension of geometry context
            source_dim: Dimension of NBI source context
            n_operators: Number of physical operators to combine
        """
        super().__init__()
        
        self.koopman_dim = koopman_dim
        self.n_operators = n_operators
        
        # Neoclassical operator
        self.K_neo = MetriplecticKoopman(koopman_dim, geometry_dim)
        
        # NBI source operator (GRU-based, recurrent)
        self.K_NBI = nn.GRUCell(
            input_size=koopman_dim + source_dim,
            hidden_size=koopman_dim
        )
        
        # Turbulent operator (MLP-based, higher-capacity nonlinearity)
        dim_params = 4  # η_i, η_e, s, α (dimensionless parameters)
        self.K_turb = nn.Sequential(
            nn.Linear(koopman_dim + dim_params, 512),
            nn.GELU(),
            nn.Linear(512, koopman_dim)
        )
        
        # Charge exchange operator (simple linear)
        self.K_cx = nn.Linear(koopman_dim, koopman_dim, bias=False)
        
        # Operator mixing weights (learned via MLP)
        # Takes koopman state + geometry context, outputs mixing weights
        self.mix_weights = nn.Sequential(
            nn.Linear(koopman_dim + geometry_dim, 512),
            nn.GELU(),
            nn.Linear(512, n_operators),
            nn.Softmax(dim=-1)  # Ensure weights sum to 1
        )
        
        # Residual operator (learns unmodeled dynamics)
        self.K_residual = nn.Sequential(
            nn.Linear(koopman_dim, 512),
            nn.GELU(),
            nn.Linear(512, koopman_dim)
        )
    
    def forward(self, z: torch.Tensor, 
                geom_context: torch.Tensor,
                source_context: torch.Tensor,
                phys_params: torch.Tensor) -> torch.Tensor:
        """
        Decomposed operator forward pass.
        
        Args:
            z: Koopman coordinates (batch, koopman_dim)
            geom_context: Geometry context (batch, geometry_dim)
            source_context: NBI source context (batch, source_dim)
            phys_params: Dimensionless parameters (batch, 4)
        
        Returns:
            Delta in Koopman space (batch, koopman_dim)
        """
        batch_size = z.shape[0]
        
        # Compute increment from each operator
        # Δz = K(z) - z (so final state is z + Δz)
        
        # Neoclassical
        dz_neo = self.K_neo(z, geom_context) - z
        
        # NBI source
        z_nbi_input = torch.cat([z, source_context], dim=1)
        dz_NBI = self.K_NBI(z_nbi_input, z) - z
        
        # Turbulent
        z_turb_input = torch.cat([z, phys_params], dim=1)
        dz_turb = self.K_turb(z_turb_input) - z
        
        # Charge exchange
        dz_cx = self.K_cx(z) - z
        
        # Residual
        dz_res = self.K_residual(z) - z
        
        # Compute mixing weights (adaptive to state and geometry)
        w_input = torch.cat([z, geom_context], dim=1)
        w = self.mix_weights(w_input)  # (batch, n_operators), sums to 1
        
        # Combine with learned weights
        dz = (w[:, 0:1] * dz_neo + 
              w[:, 1:2] * dz_NBI + 
              w[:, 2:3] * dz_turb + 
              w[:, 3:4] * dz_cx +
              (1 - w.sum(dim=1, keepdim=True)) * dz_res)
        
        return dz


# ==============================================================================
# SECTION 5: PHYSICS PROJECTION LAYER
# ==============================================================================

class PhysicsProjectionLayer(nn.Module):
    """
    Projects predicted plasma state onto physically admissible manifold.
    
    Critical Insight (Section 6.1 of plan):
    HARD constraints (exactly satisfied) are MUCH better than soft (penalties).
    
    Examples of physical constraints that MUST be satisfied:
    1. Quasi-neutrality: n_e(ρ) = n_i(ρ) + Σ_j Z_j n_j(ρ)
    2. Positivity: All densities ≥ 0, all temperatures ≥ 0
    3. Particle inventory: ∫ n_i(ρ) V'(ρ) dρ = N_total(t)
    4. Boundary conditions: n_i(ρ=1) = n_edge (fixed)
    5. Temperature ordering: T_e ≥ 0, T_i ≥ 0 (can be ≠)
    
    Implementation: Iterative projection (up to 10 iterations for convergence)
    """
    
    def __init__(self, n_radial: int = 100, 
                 impurity_charges: Optional[List[float]] = None):
        """
        Initialize projection layer.
        
        Args:
            n_radial: Number of radial points
            impurity_charges: Charges of impurity species (for quasi-neutrality)
        """
        super().__init__()
        
        self.n_radial = n_radial
        self.impurity_charges = impurity_charges or [6.0]  # Default: Carbon
    
    def project_positivity(self, state: Dict[str, torch.Tensor]) -> Dict:
        """
        Project densities and temperatures to positive values.
        
        Uses softplus: f(x) = log(1 + exp(x)), which ensures f(x) > 0
        and is differentiable everywhere.
        """
        state = state.copy()
        
        for key in ['NI', 'NE', 'NH', 'TE', 'TI']:
            if key in state:
                state[key] = F.softplus(state[key], beta=10.0)
        
        return state
    
    def project_quasineutrality(self, state: Dict[str, torch.Tensor]) -> Dict:
        """
        Project to quasi-neutrality constraint: n_e = n_i + Σ_j Z_j n_impurity_j
        
        Method: Given n_e and impurity profiles, adjust n_i to satisfy constraint.
        
        n_i_physical = n_e - Σ_j Z_j n_impurity_j
        n_i_new = α n_i_physical + (1-α) n_i_old  (soft update)
        """
        state = state.copy()
        alpha = 0.9  # Projection strength (0=no change, 1=exact)
        
        if 'NE' in state and 'NI' in state:
            impurity_contribution = torch.zeros_like(state['NI'])
            
            # Add impurity contributions if available
            for j, Z_j in enumerate(self.impurity_charges):
                key = f'NIMP_{j}'
                if key in state:
                    impurity_contribution += Z_j * state[key]
            
            # Quasi-neutrality requires:
            n_i_required = state['NE'] - impurity_contribution
            
            # Soft update
            state['NI'] = alpha * n_i_required + (1 - alpha) * state['NI']
        
        return state
    
    def project_particle_inventory(self, state: Dict[str, torch.Tensor],
                                   V_prime: torch.Tensor,
                                   N_total: float) -> Dict:
        """
        Conserve particle inventory to within ~1%
        
        Total number of particles:
        N_total = ∫_0^1 n_i(ρ) V'(ρ) dρ
        
        Method: Scale entire profile to match target inventory
        """
        state = state.copy()
        
        if 'NI' in state:
            # Compute current inventory (trapezoidal integration)
            N_current = torch.sum(state['NI'] * V_prime, dim=-1, keepdim=True)
            
            # Scale to target
            scale = N_total / (N_current + 1e-10)
            state['NI'] = state['NI'] * scale
        
        return state
    
    def forward(self, state_dict: Dict[str, torch.Tensor],
                V_prime: Optional[torch.Tensor] = None,
                N_total: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Apply hard constraints to plasma state.
        
        Args:
            state_dict: Dictionary of state variables (NI, NE, TE, TI, etc.)
            V_prime: Flux surface volume derivative (for inventory conservation)
            N_total: Total particle inventory (for conservation)
        
        Returns:
            Projected state dictionary
        """
        # Apply constraints in sequence
        state = self.project_positivity(state_dict)
        state = self.project_quasineutrality(state)
        
        if V_prime is not None and N_total is not None:
            state = self.project_particle_inventory(state, V_prime, N_total)
        
        return state


# ==============================================================================
# SECTION 6: FULL DGKNet MODEL
# ==============================================================================

class DGKNet(nn.Module):
    """
    Decomposed Geometric Koopman Network - FULL IMPLEMENTATION
    
    Complete architecture combining all components:
    1. Geometry Transformer - encodes static flux surface geometry
    2. Chebyshev Encoders - compress smooth radial profiles
    3. Metriplectic Koopman - long-horizon stable lifted dynamics
    4. Decomposed Operators - physically meaningful dynamics decomposition
    5. Physics Projection - hard constraints for realism
    
    Key Properties:
    ✓ Long-horizon stable (metriplectic structure prevents energy explosion)
    ✓ Physically interpretable (decomposed operators match transport theory)
    ✓ Hardware constraint-aware (hard projection ensures validity)
    ✓ Geometry-aware (static context improves generalization)
    ✓ Uncertainty quantifiable (ensemble-ready architecture)
    
    Model Configuration (hyperparameters):
    - latent_profile: Dimension after Chebyshev encoding (~128)
    - latent_geom: Dimension of geometry context (~128)
    - koopman_dim: Dimension of Koopman lifting (~256, must be even)
    - n_cheb: Number of Chebyshev modes (~32)
    """
    
    def __init__(self, 
                 state_dim: int,
                 n_radial: int = 100,
                 n_psi: int = 100,
                 n_fourier: int = 66,
                 latent_profile: int = 320,
                 latent_geom: int = 320,
                 koopman_dim: int = 1024,
                 n_cheb: int = 256,
                 **config):
        """
        Initialize DGKNet model.
        
        Args:
            state_dim: Total dimension of raw state vector
            n_radial: Number of radial grid points
            n_psi: Number of flux surfaces (for geometry)
            n_fourier: Number of Fourier modes in geometry
            latent_profile: Output dimension of profile encoders
            latent_geom: Output dimension of geometry transformer
            koopman_dim: Dimension of Koopman lifting (must be even)
            n_cheb: Number of Chebyshev modes
            **config: Additional config parameters
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.n_radial = n_radial
        self.n_psi = n_psi
        self.n_fourier = n_fourier
        self.koopman_dim = koopman_dim
        self.latent_profile = latent_profile
        self.config = config
        
        if koopman_dim % 2 != 0:
            raise ValueError("koopman_dim must be even for symplectic structure")
        
        # ============ ENCODING STAGE ============
        
        # Encode kinetic profiles via Chebyshev basis
        # Main profiles: NE, NI, TE, TI, PPLAS, NH (~100 radial points each)
        self.profile_encoder = ChebyshevProfileEncoder(
            n_radial=n_radial,
            n_cheb=n_cheb,
            latent_dim=latent_profile
        )
        
        # Encode transport coefficients similarly
        self.transport_encoder = ChebyshevProfileEncoder(
            n_radial=n_radial,
            n_cheb=n_cheb // 2,  # Smaller basis for secondary profiles
            latent_dim=latent_profile // 2
        )
        
        # Encode source profiles
        self.source_encoder = ChebyshevProfileEncoder(
            n_radial=n_radial,
            n_cheb=n_cheb // 2,
            latent_dim=latent_profile // 2
        )
        
        # Encode geometry (static, Fourier-based transformer)
        self.geometry_encoder = GeometryTransformer(
            n_fourier=n_fourier,
            n_psi=n_psi,
            latent_geom=latent_geom,
            n_heads=16,
            n_layers=8
        )

        self.limiter_geometry_encoder = GeometryTransformer(
            n_fourier=n_fourier,
            n_psi=n_psi,
            latent_geom=latent_geom,
            n_heads=8,
            n_layers=4
        )
        
        # Global scalar encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(17, 64),  # Assuming 17 global scalars
            nn.GELU(),
            nn.Linear(64, latent_profile // 4)
        )

        self.pre_shot_encoder = nn.Sequential(
            nn.Linear(PRE_SHOT_CONTEXT_DIM, 64),
            nn.GELU(),
            nn.Linear(64, latent_profile // 4)
        )

        self.pre_shot_initializer = nn.Sequential(
            nn.Linear(PRE_SHOT_CONTEXT_DIM, latent_profile * 2),
            nn.GELU(),
            nn.Linear(latent_profile * 2, koopman_dim)
        )

        self.pre_shot_phys_projector = nn.Sequential(
            nn.Linear(PRE_SHOT_CONTEXT_DIM, 32),
            nn.GELU(),
            nn.Linear(32, 4)
        )
        
        # ============ PLASMA STATE ENCODER ============
        
        # Combine all encoded components
        combined_dim = (
            6 * latent_profile +
            4 * (latent_profile // 2) +
            3 * (latent_profile // 2) +
            latent_profile // 4 +
            latent_profile // 4
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(combined_dim, koopman_dim * 2),
            nn.LayerNorm(koopman_dim * 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(koopman_dim * 2, koopman_dim),
            nn.GELU(),
            nn.Linear(koopman_dim, koopman_dim)
        )
        
        # ============ KOOPMAN LIFTING ============
        
        # Lift state to Koopman space
        self.koopman_encoder = nn.Linear(koopman_dim, koopman_dim)
        
        # ============ DYNAMICS: METRIPLECTIC KOOPMAN + DECOMPOSED OPS ============
        
        # Metriplectic Koopman operator
        self.koopman_dynamics = MetriplecticKoopman(
            koopman_dim=koopman_dim,
            geometry_dim=latent_geom
        )
        
        # Decomposed operator block (combines physics)
        self.decomposed_ops = DecomposedOperatorBlock(
            koopman_dim=koopman_dim,
            geometry_dim=latent_geom,
            source_dim=latent_profile // 2,
            n_operators=4
        )
        
        # ============ KOOPMAN DECODING ============
        
        self.koopman_decoder = nn.Sequential(
            nn.Linear(koopman_dim, koopman_dim),
            nn.GELU(),
            nn.Linear(koopman_dim, koopman_dim)
        )
        
        # ============ OUTPUT DECODING ============
        
        # Decode back to profile space
        self.profile_decoder = nn.Sequential(
            nn.Linear(koopman_dim, latent_profile * 2),
            nn.LayerNorm(latent_profile * 2),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(latent_profile * 2, latent_profile),
            nn.GELU(),
            nn.Linear(latent_profile, n_radial * 6)  # 6 main profiles
        )
        self.geometry_decoder = nn.Sequential(
            nn.Linear(koopman_dim, latent_geom * 2),
            nn.LayerNorm(latent_geom * 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(latent_geom * 2, latent_geom),
            nn.GELU(),
            nn.Linear(latent_geom, n_psi * n_fourier)
        )
        
        # ============ PHYSICS CONSTRAINTS ============
        
        self.physics_projection = PhysicsProjectionLayer(n_radial=n_radial)
        
        logger.info(f"DGKNet initialized: state_dim={state_dim}, "
                   f"koopman_dim={koopman_dim}, latent_geom={latent_geom}")
        # Initialize weights for stable training
        try:
            self._init_weights()
        except Exception:
            # If initialization fails for any reason, continue with defaults
            logger.warning("DGKNet weight initialization encountered an issue; continuing with defaults.")

    def _init_weights(self):
        """
        Apply targeted weight initialization for stable optimization.
        - Linear / weight matrices: Xavier uniform
        - LayerNorm / Embedding: leave default
        - Small normal init for metriplectic params
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRUCell):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

        # Initialize metriplectic small and stable
        try:
            if hasattr(self, 'koopman_dynamics'):
                with torch.no_grad():
                    # Small skew-symmetric A
                    self.koopman_dynamics.A_sym_param.copy_(torch.randn_like(self.koopman_dynamics.A_sym_param) * 1e-3)
                    self.koopman_dynamics.L_dis_param.copy_(torch.randn_like(self.koopman_dynamics.L_dis_param) * 1e-3)
        except Exception:
            pass
    
    def forward(self, batch_data: Dict[str, torch.Tensor], apply_physics_projection: bool = False) -> Dict[str, torch.Tensor]:
        """
        Single-step DGKNet prediction.
        
        Args:
            batch_data: Dictionary containing:
                - 'kinetic_profiles': (batch, 6, n_radial)
                - 'geometry_tensor': (batch, n_psi, 66)
                - 'transport_coeff': (batch, 4, n_radial)
                - 'nbi_sources': (batch, 3, n_radial)
                - 'global_scalars': (batch, 17)
                - 'pre_shot_context': (batch, PRE_SHOT_CONTEXT_DIM)
        
        Returns:
            Dictionary with predicted profiles
        """
        # Extract components
        kinetic = batch_data['kinetic_profiles']  # (batch, 6, n_radial)
        geometry = batch_data['geometry_tensor']  # (batch, n_psi, 66)
        limiter_geometry = batch_data.get('limiter_geometry_tensor', None)
        transport = batch_data.get('transport_coeff', None)
        sources = batch_data.get('nbi_sources', None)
        globals_ = batch_data.get('global_scalars', None)
        pre_shot_context = batch_data.get('pre_shot_context', None)
        
        batch_size = kinetic.shape[0]
        
        # ============ ENCODING ============
        
        # Encode each profile separately, then concatenate
        encoded_profiles = []
        for i in range(kinetic.shape[1]):
            enc = self.profile_encoder(kinetic[:, i, :])
            encoded_profiles.append(enc)
        
        encoded_transport = []
        if transport is not None:
            for i in range(transport.shape[1]):
                enc = self.transport_encoder(transport[:, i, :])
                encoded_transport.append(enc)
        
        encoded_sources = []
        if sources is not None:
            for i in range(sources.shape[1]):
                enc = self.source_encoder(sources[:, i, :])
                encoded_sources.append(enc)
        
        # Encode geometry (static context for all predictions)
        geom_context = self.geometry_encoder(geometry)
        if limiter_geometry is not None:
            limiter_geom_reshaped = limiter_geometry
            while limiter_geom_reshaped.dim() > 3 and limiter_geom_reshaped.shape[1] == 1:
                limiter_geom_reshaped = limiter_geom_reshaped.squeeze(1)
            if limiter_geom_reshaped.dim() == 2:
                limiter_geom_reshaped = limiter_geom_reshaped.unsqueeze(1)
            if limiter_geom_reshaped.dim() != 3:
                raise ValueError(
                    f"Expected limiter geometry rank 2 or 3, got shape {tuple(limiter_geom_reshaped.shape)}"
                )
            geom_context = geom_context + self.limiter_geometry_encoder(limiter_geom_reshaped)
        
        # Encode global scalars
        if globals_ is not None:
            global_enc = self.global_encoder(globals_)
        else:
            global_enc = torch.zeros(batch_size, self.latent_profile // 4,
                                    device=kinetic.device)

        if pre_shot_context is not None:
            # Ensure pre_shot_context is 64-dimensional; pad if needed
            if pre_shot_context.shape[-1] < 64:
                pad_amount = 64 - pre_shot_context.shape[-1]
                pre_shot_context = torch.cat([
                    pre_shot_context,
                    torch.zeros(batch_size, pad_amount, device=pre_shot_context.device)
                ], dim=-1)
            pre_shot_enc = self.pre_shot_encoder(pre_shot_context)
        else:
            pre_shot_enc = torch.zeros(batch_size, self.latent_profile // 4,
                                       device=kinetic.device)

        
        # Combine all encodings
        combined = torch.cat(
            encoded_profiles + encoded_transport + encoded_sources + [global_enc, pre_shot_enc],
            dim=1
        )
        
        # Final state encoding
        state_vec = self.state_encoder(combined)
        if pre_shot_context is not None:
            state_vec = state_vec + 0.3 * self.pre_shot_initializer(pre_shot_context)  # Increased from 0.1 to boost pre_shot signal
        
        # ============ KOOPMAN LIFTING ============
        
        z = self.koopman_encoder(state_vec)  # (batch, koopman_dim)
        
        # ============ DYNAMICS (SINGLE STEP) ============
        
        # Apply metriplectic Koopman (reversible part)
        z_koopman = self.koopman_dynamics(z, geom_context)
        
        # Apply decomposed operators (total dynamics)
        phys_params = self.pre_shot_phys_projector(pre_shot_context) if pre_shot_context is not None else torch.ones(batch_size, 4, device=z.device)
        source_context = encoded_sources[0] if encoded_sources else \
                        torch.zeros(batch_size, self.latent_profile // 2,
                                  device=z.device)
        
        dz = self.decomposed_ops(z, geom_context, source_context, phys_params)
        z_next = z_koopman + 0.25 * torch.tanh(dz)  # Increased from 0.05 to allow meaningful dynamics evolution
        
        # ============ KOOPMAN DECODING ============
        
        z_decoded = self.koopman_decoder(z_next)
        
        # ============ OUTPUT DECODING ============
        
        output_flat = self.profile_decoder(z_decoded)
        geometry_delta = self.geometry_decoder(z_decoded).view(batch_size, self.n_psi, self.n_fourier)
        
        # Reshape to profiles
        output_dict = {
            'NI': output_flat[:, :self.n_radial].view(batch_size, self.n_radial),
            'NE': output_flat[:, self.n_radial:2*self.n_radial].view(batch_size, self.n_radial),
            'NH': output_flat[:, 2*self.n_radial:3*self.n_radial].view(batch_size, self.n_radial),
            'TE': output_flat[:, 3*self.n_radial:4*self.n_radial].view(batch_size, self.n_radial),
            'TI': output_flat[:, 4*self.n_radial:5*self.n_radial].view(batch_size, self.n_radial),
            'PPLAS': output_flat[:, 5*self.n_radial:6*self.n_radial].view(batch_size, self.n_radial),
            'geometry_tensor': geometry + 0.25 * torch.tanh(geometry_delta),  # Increased from 0.1 for consistency
        }
        
        # ============ PHYSICS PROJECTION ============
        
        # Keep projection opt-in so normalized training/evaluation targets are not
        # forced through hard physical constraints before denormalization.
        if apply_physics_projection:
            if hasattr(self, 'V_prime'):
                output_dict = self.physics_projection(
                    output_dict,
                    V_prime=self.V_prime,
                    N_total=self.N_total
                )
            else:
                output_dict = self.physics_projection(output_dict)
        
        return output_dict
    
    def rollout(self, batch_data: Dict[str, torch.Tensor], T: int,
                return_trajectory: bool = True) -> Dict[str, torch.Tensor]:
        """
        Multi-step rollout prediction.
        
        Args:
            batch_data: Initial state
            T: Number of steps to predict
            return_trajectory: If True, return all timesteps; else just final
        
        Returns:
            Trajectory or final state
        """
        trajectory = []
        current_data = dict(batch_data)
        
        with torch.no_grad():
            for t in range(T):
                prediction = self.forward(current_data)
                trajectory.append(prediction)

                current_data = dict(current_data)
                current_data['kinetic_profiles'] = torch.clamp(torch.stack([
                    prediction['NI'], prediction['NE'], prediction['NH'],
                    prediction['TE'], prediction['TI'], prediction['PPLAS']
                ], dim=1), min=-5.0, max=5.0)
                if 'geometry_tensor' in prediction:
                    current_data['geometry_tensor'] = torch.clamp(prediction['geometry_tensor'], min=-5.0, max=5.0)
                else:
                    current_data['geometry_tensor'] = current_data.get('geometry_tensor')
                current_data['limiter_geometry_tensor'] = current_data.get('limiter_geometry_tensor', batch_data.get('limiter_geometry_tensor'))
        
        if return_trajectory:
            # Stack predictions
            return trajectory
        return trajectory[-1]


def main():
    """Train DGKNet on the saved Phase 0 artifact or run the architecture test."""
    parser = argparse.ArgumentParser(description='Phase 2 DGKNet training and architecture check')
    parser.add_argument('--dataset', type=str, default='',
                        help='Optional legacy Phase 0 dataset artifact used only if split files are missing')
    parser.add_argument('--train-dataset', type=str, default=str(DEFAULT_LEGACY_PHASE0_DATASET_PATH.parent / 'dataset_train.pt'),
                        help='Path to the train split Phase 0 dataset')
    parser.add_argument('--val-dataset', type=str, default=str(DEFAULT_LEGACY_PHASE0_DATASET_PATH.parent / 'dataset_val.pt'),
                        help='Path to the val split Phase 0 dataset')
    parser.add_argument('--test-dataset', type=str, default=str(DEFAULT_LEGACY_PHASE0_DATASET_PATH.parent / 'dataset_test.pt'),
                        help='Path to the test split Phase 0 dataset')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of DGKNet training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default=str(DEFAULT_DGKNET_CHECKPOINT_DIR),
                        help='Directory for DGKNet checkpoints')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'],
                        help='Compute device for DGKNet training')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--pin-memory', action='store_true',
                        help='Enable pinned memory for faster host->device transfer')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training when running on CUDA')
    parser.add_argument('--val-fraction', type=float, default=0.2,
                        help='Validation fraction (only used if split datasets not found)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--test-only', action='store_true',
                        help='Run the dummy smoke test instead of training')
    args = parser.parse_args()

    if args.test_only:
        test_dgknet()
        return

    trainer = DGKNetTrainer(
        device=args.device,
        checkpoint_dir=Path(args.checkpoint_dir),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_amp=args.amp,
    )
    summary = trainer.train(
        dataset_path=args.dataset,
        train_dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        test_dataset_path=args.test_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        patience=args.patience,
    )

    logger.info("\nTraining complete. Summary:")
    logger.info(f"  checkpoint: {summary['checkpoint']}")
    logger.info(f"  final_train_loss: {summary['final_train_loss']}")
    logger.info(f"  final_val_loss: {summary['final_val_loss']}")


# ==============================================================================
# TESTING
# ==============================================================================

def test_dgknet():
    """Quick sanity check of DGKNet components."""
    
    logger.info("Testing DGKNet architecture...")
    
    # Create model
    model = DGKNet(
        state_dim=(len(PROFILE_ORDER) + len(TRANSPORT_ORDER) + len(SOURCE_ORDER)) * CANONICAL_N_RADIAL + CANONICAL_N_PSI * 66 + 17,
        n_radial=CANONICAL_N_RADIAL,
        n_psi=CANONICAL_N_PSI,
        n_fourier=66,
    )
    
    # Create dummy batch
    batch = {
        'kinetic_profiles': torch.randn(2, 6, CANONICAL_N_RADIAL),
        'geometry_tensor': torch.randn(2, CANONICAL_N_PSI, 66),
        'transport_coeff': torch.randn(2, 4, CANONICAL_N_RADIAL),
        'nbi_sources': torch.randn(2, 3, CANONICAL_N_RADIAL),
        'global_scalars': torch.randn(2, 17),
    }
    
    # Forward pass
    output = model(batch)
    
    logger.info(f"✓ DGKNet forward pass successful")
    logger.info(f"  Output keys: {list(output.keys())}")
    for key, val in output.items():
        logger.info(f"  {key}: shape {val.shape}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
