"""10-component loss function for multi-machine STRONG-RMMD training.

TIER 2 IMPLEMENTATION:
- Kinetics (profiles) PRIMARY (70% of data loss)
- Geometry SECONDARY (30% of data loss)
- Profile-specific physics constraints (non-negativity, smoothness)
- True energy conservation (grad_H dot S dot z = 0)
- True dissipation orthogonality (grad_H dot D dot z <= 0)
"""

from __future__ import annotations

from typing import Dict, Optional
import logging

import torch
from torch import nn
import torch.nn.functional as F


class RMMDLossFunction(nn.Module):
    """
    Composite loss combining physics and data-driven objectives.

    11 components (with annealing schedules):
    1. L_koopman:          Kinetics reconstruction (PRIMARY, always on)
    2. L_geometry:         Geometry reconstruction (SECONDARY)
    3. L_kinetics_consistency: Profile physics (non-negativity, smoothness)
    4. L_energy:           Energy conservation (ramp 0-100 epochs)
    5. L_dissip:           Dissipation positivity (ramp 0-100 epochs)
    6. L_snt:              Symmetric negative transpose (ramp 20-150 epochs)
    7. L_jarzy:            Jarzynski-like bound (ramp 50-200 epochs)
    8. L_D_res_sparse:     Off-diagonal sparsity (constant)
    9. L_delta_S:          Geometry-informed correction (constant)
    10. L_sut_align:       Universal mode alignment (ramp 100+ epochs)
    11. L_physics:         Combined physics scores (constant)
    12. L_reg:             L2 regularization (constant)
    """

    def __init__(self, model: nn.Module, config: Optional[Dict] = None):
        super().__init__()
        self.model = model
        self.config = config or {}
        self.logger = logging.getLogger("rmmd_loss")
        # Threshold for brief jarzynski logging
        self.jarzy_log_threshold = float(self.config.get("jarzy_log_threshold", 10.0))

    def _denormalize_profiles(self, tensor: torch.Tensor) -> torch.Tensor:
        stats = self.config.get("normalization_stats") or {}
        if not stats:
            return tensor
        out = tensor.clone()
        profile_order = self.config.get("profile_order")
        if not profile_order:
            profile_order = ["NI", "NE", "NH", "TE", "TI", "PPLAS"]
        if out.dim() == 1:
            out = out.unsqueeze(0)
        for idx, name in enumerate(profile_order):
            key = f"kinetic_profiles.{name}"
            entry = stats.get(key)
            if entry is None:
                continue
            mean_val = float(entry.get("mean", 0.0))
            std_val = float(entry.get("std", 1.0))
            out[:, idx * 40 : (idx + 1) * 40] = out[:, idx * 40 : (idx + 1) * 40] * std_val + mean_val
        return out

    def _data_space_profiles(self, tensor: torch.Tensor) -> torch.Tensor:
        if bool(self.config.get("use_denormalized_data_loss", True)):
            return self._denormalize_profiles(tensor)
        return tensor

    def l_koopman(self, x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        """Kinetics reconstruction loss aligned with evaluation NRMSE/NMAE."""
        if x_true is None or x_pred is None:
            return torch.tensor(0.0, device=x_pred.device if x_pred is not None else torch.device('cpu'))

        if x_true.dim() == 1:
            x_true = x_true.unsqueeze(0)
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)

        xt = self._data_space_profiles(x_true.view(x_true.shape[0], -1))
        xp = self._data_space_profiles(x_pred.view(x_pred.shape[0], -1))
        diff = xp - xt
        rmse_i = torch.sqrt(torch.mean(diff ** 2, dim=1) + 1e-12)
        mae_i = torch.mean(diff.abs(), dim=1)
        target_norm = torch.norm(xt, dim=1).clamp_min(0.05)
        return torch.mean(0.7 * (rmse_i / target_norm) + 0.3 * (mae_i / target_norm))

    def l_eval_aligned_kinetics(self, x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        """Auxiliary kinetics term in denormalized space."""
        if x_true is None or x_pred is None:
            return torch.tensor(0.0, device=x_pred.device if x_pred is not None else torch.device('cpu'))

        if x_true.dim() == 1:
            x_true = x_true.unsqueeze(0)
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)

        xt = self._data_space_profiles(x_true.view(x_true.shape[0], -1))
        xp = self._data_space_profiles(x_pred.view(x_pred.shape[0], -1))
        diff = xp - xt
        mse_i = torch.mean(diff ** 2, dim=1)
        mae_i = torch.mean(diff.abs(), dim=1)
        return torch.mean(0.5 * mse_i + 0.5 * mae_i)

    def l_latent_alignment(self, z_true: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
        """Direct latent matching loss so the rollout latent stays anchored to the encoder target."""
        if z_true is None or z_pred is None:
            return torch.tensor(0.0, device=z_pred.device if z_pred is not None else torch.device('cpu'))

        if z_true.dim() == 1:
            z_true = z_true.unsqueeze(0)
        if z_pred.dim() == 1:
            z_pred = z_pred.unsqueeze(0)

        zt = z_true.view(z_true.shape[0], -1)
        zp = z_pred.view(z_pred.shape[0], -1)
        diff = zp - zt
        return torch.mean(torch.mean(diff ** 2, dim=1))

    def l_norm_consistency(
        self,
        x_true: torch.Tensor,
        x_pred: torch.Tensor,
        z_true: torch.Tensor,
        z_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Penalty for norm drift between prediction and target in state and latent spaces."""
        if x_true is None or x_pred is None or z_true is None or z_pred is None:
            device = x_pred.device if x_pred is not None else (z_pred.device if z_pred is not None else torch.device('cpu'))
            return torch.tensor(0.0, device=device)

        if x_true.dim() == 1:
            x_true = x_true.unsqueeze(0)
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)
        if z_true.dim() == 1:
            z_true = z_true.unsqueeze(0)
        if z_pred.dim() == 1:
            z_pred = z_pred.unsqueeze(0)

        xt = x_true.view(x_true.shape[0], -1)
        xp = x_pred.view(x_pred.shape[0], -1)
        zt = z_true.view(z_true.shape[0], -1)
        zp = z_pred.view(z_pred.shape[0], -1)

        x_true_norm = torch.norm(xt, dim=1).clamp_min(1e-8)
        x_pred_norm = torch.norm(xp, dim=1).clamp_min(1e-8)
        z_true_norm = torch.norm(zt, dim=1).clamp_min(1e-8)
        z_pred_norm = torch.norm(zp, dim=1).clamp_min(1e-8)

        x_norm_gap = torch.abs(x_pred_norm - x_true_norm)
        z_norm_gap = torch.abs(z_pred_norm - z_true_norm)
        return torch.mean(0.5 * x_norm_gap + 0.5 * z_norm_gap)

    def l_koopman_hard_samples(
        self,
        x_true: torch.Tensor,
        x_pred: torch.Tensor,
        d_res: torch.Tensor | None,
    ) -> torch.Tensor:
        """Kinetics loss that upweights high-d_res samples (hard dissipation regimes).

        Gate evidence says error is correlated with d_res on key machines. Instead of
        suppressing this correlation, prioritize reducing kinetics error where d_res is large.
        """
        if x_true is None or x_pred is None:
            return torch.tensor(0.0, device=x_pred.device if x_pred is not None else torch.device('cpu'))

        if x_true.dim() == 1:
            x_true = x_true.unsqueeze(0)
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)

        bsz = x_true.shape[0]
        xt = self._data_space_profiles(x_true.view(bsz, -1))
        xp = self._data_space_profiles(x_pred.view(bsz, -1))

        mse_i = torch.mean((xp - xt) ** 2, dim=1)
        mae_i = torch.mean((xp - xt).abs(), dim=1)
        denorm_i = 0.7 * mse_i + 0.3 * mae_i

        if d_res is None:
            return torch.mean(denorm_i)

        if d_res.dim() == 4:
            d_eff = torch.mean(d_res, dim=1)
        else:
            d_eff = d_res

        d_strength = torch.mean(torch.abs(d_eff.view(d_eff.shape[0], -1)), dim=1)
        d_strength = torch.nan_to_num(d_strength, nan=0.0, posinf=0.0, neginf=0.0)
        d_mean = torch.mean(d_strength)
        if d_strength.numel() > 1:
            d_std = torch.std(d_strength, unbiased=False)
        else:
            d_std = torch.tensor(1.0, device=d_strength.device, dtype=d_strength.dtype)
        d_std = torch.clamp(d_std, min=1e-6)
        # Keep weighting mild to avoid destabilizing early optimization.
        w = 1.0 + 0.5 * torch.sigmoid((d_strength - d_mean) / d_std)
        return torch.mean(w * denorm_i)

    def l_kinetics_consistency(self, x_pred: torch.Tensor) -> torch.Tensor:
        """Physics constraints on profiles: non-negativity, smoothness."""
        n_radial = 40
        penalty = torch.tensor(0.0, device=x_pred.device, dtype=x_pred.dtype)
        
        x_pred = self._data_space_profiles(x_pred.view(x_pred.shape[0], -1))

        # Penalize negative values (all physical profiles >= 0)
        neg_penalty = torch.mean(F.relu(-x_pred))
        penalty = penalty + 0.1 * neg_penalty
        
        # Penalize extreme oscillations (smooth profiles)
        if x_pred.shape[-1] > 2:
            diffs = torch.diff(x_pred, dim=-1)
            if diffs.shape[-1] > 1:
                d2 = torch.diff(diffs, dim=-1)
                roughness = torch.mean(d2.abs())
                penalty = penalty + 0.05 * roughness
        
        return penalty

    def l_profile_bounds(self, x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        """Enforce profile-level physical bounds using batch statistics from x_true.

        - lower bound: 0 (non-negative)
        - upper bound: 5 * mean_true (per-profile, per-batch)
        """
        if x_true is None or x_pred is None:
            return torch.tensor(0.0, device=x_pred.device if x_pred is not None else torch.device('cpu'))

        # Ensure batch dim
        if x_true.dim() == 1:
            x_true = x_true.unsqueeze(0)
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)
        x_true = self._data_space_profiles(x_true.view(x_true.shape[0], -1))
        x_pred = self._data_space_profiles(x_pred.view(x_pred.shape[0], -1))

        batch = x_true.shape[0]
        profile_order = self.config.get("profile_order")
        if not profile_order:
            profile_order = ["NI", "NE", "NH", "TE", "TI", "PPLAS"]
        n_profiles = len(profile_order)
        n_radial = 40
        if x_true.shape[-1] < n_profiles * n_radial:
            return torch.tensor(0.0, device=x_pred.device)

        true_profiles = x_true.view(batch, n_profiles, n_radial)
        pred_profiles = x_pred.view(batch, n_profiles, n_radial)

        mean_true = torch.mean(true_profiles, dim=(0, 2))
        upper = mean_true.unsqueeze(0).unsqueeze(-1) * 5.0 + 1e-6

        upper_violation = F.relu(pred_profiles - upper)
        lower_violation = F.relu(-pred_profiles)

        penalty = torch.mean(upper_violation) + torch.mean(lower_violation)
        return penalty

    def l_geometry(self, geom_pred: torch.Tensor, geom_target: torch.Tensor) -> torch.Tensor:
        """Geometry reconstruction loss (SECONDARY). Normalized by target variance."""
        if geom_pred is None or geom_target is None:
            return torch.tensor(0.0, device=geom_pred.device if geom_pred is not None else torch.device('cpu'))
        mse = F.mse_loss(geom_pred, geom_target, reduction='mean')
        target_var = torch.var(geom_target) + 1e-6
        return mse / target_var

    def l_energy(self, z_true: torch.Tensor, z_pred: torch.Tensor, s_matrix: torch.Tensor | None = None) -> torch.Tensor:
        """True energy conservation: grad_H dot S dot z = 0 where H = 0.5||z||^2."""
        if s_matrix is None:
            return torch.tensor(0.0, device=z_pred.device)
        
        active_dim = min(int(s_matrix.shape[-1]), int(z_pred.shape[-1]))
        grad_h = z_pred[..., :active_dim]
        
        s_antisym = 0.5 * (s_matrix - s_matrix.transpose(-1, -2))
        s_antisym = s_antisym[..., :active_dim, :active_dim]
        
        s_z = torch.einsum("bij,bj->bi", s_antisym, grad_h)
        conservation_residual = torch.sum(s_z * grad_h, dim=-1).abs()
        
        z_norm_sq = torch.sum(grad_h ** 2, dim=-1) + 1e-8
        normalized_residual = conservation_residual / z_norm_sq
        
        return torch.mean(normalized_residual)

    def l_dissip(self, d_total: torch.Tensor, z_pred: torch.Tensor | None = None) -> torch.Tensor:
        """True dissipation: grad_H dot D dot z <= 0 (energy dissipates, not injected)."""
        d_sym = 0.5 * (d_total + d_total.transpose(-1, -2))
        
        if z_pred is None:
            z_pred = torch.ones(d_total.shape[0], d_total.shape[-1], device=d_total.device, dtype=d_total.dtype)
        
        active_dim = min(int(d_sym.shape[-1]), int(z_pred.shape[-1]))
        d_sym_active = d_sym[..., :active_dim, :active_dim]
        z_active = z_pred[..., :active_dim]
        
        d_z = torch.einsum("bij,bj->bi", d_sym_active, z_active)
        dissip_work = torch.sum(d_z * z_active, dim=-1)
        
        # Penalty for positive dissipation (energy injection forbidden)
        dissip_orthogonality_penalty = torch.mean(F.relu(dissip_work))
        
        # Enforce PSD on dissipative matrix
        psd_penalty = torch.tensor(0.0, device=d_total.device)
        jitter = 1e-6
        eye = torch.eye(d_sym_active.shape[-1], device=d_sym_active.device, dtype=d_sym_active.dtype)
        
        for _ in range(3):
            try:
                eigs = torch.linalg.eigvalsh(d_sym_active + jitter * eye)
                eigs_min = torch.min(eigs, dim=-1)[0]
                psd_penalty = torch.mean(F.relu(-eigs_min))
                break
            except RuntimeError:
                jitter *= 10.0
        
        if torch.isnan(psd_penalty) or torch.isinf(psd_penalty):
            psd_penalty = torch.tensor(0.0, device=d_total.device)
        
        return 0.5 * dissip_orthogonality_penalty + 0.5 * psd_penalty

    def l_snt(self, d_res: torch.Tensor, d_psd: torch.Tensor | None = None) -> torch.Tensor:
        """SNT: off-diagonal residual should stay structured."""
        d_plus_dt = d_res + d_res.transpose(-1, -2)
        residual_penalty = torch.mean(torch.sum(d_plus_dt ** 2, dim=(-2, -1)))
        
        if d_psd is None:
            return residual_penalty
        psd_trace = torch.mean(torch.diagonal(0.5 * (d_psd + d_psd.transpose(-1, -2)), dim1=-2, dim2=-1).sum(dim=-1))
        return residual_penalty + 0.01 * torch.abs(psd_trace)

    def l_jarzy(self, x_true: torch.Tensor, x_pred: torch.Tensor, d_total: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
        """Jarzynski: work ~ T*||D|| relates to error growth."""
        dt = x_true - x_pred
        work_norm = torch.sum(dt ** 2, dim=-1)

        latent_dim = min(int(d_total.shape[-1]), int(z_pred.shape[-1]))
        d_latent = d_total[..., :latent_dim, :latent_dim]
        z_latent = z_pred[..., :latent_dim]

        d_action = torch.einsum("bij,bj->bi", d_latent, z_latent)
        jarzynski_work = torch.sum(d_action * z_latent, dim=-1)

        # Replace non-finite values and clamp extremes to avoid overflow
        jw_safe = torch.nan_to_num(jarzynski_work, nan=0.0, posinf=20.0, neginf=-20.0)
        jw_clamped = torch.clamp(jw_safe, min=-20.0, max=20.0)
        jw_bounded = jw_clamped / (1.0 + jw_clamped.abs())
        # brief logging for extreme Jarzynski statistics
        try:
            mean_abs = float(torch.mean(jw_bounded.abs()).item())
            if mean_abs > self.jarzy_log_threshold:
                self.logger.warning(f"High jarzynski mean_abs={mean_abs:.3f}")
        except Exception:
            pass

        return torch.mean(jw_bounded ** 2 + 0.1 * work_norm)

    def l_d_res_sparse(self, d_res: torch.Tensor, d_psd: torch.Tensor | None = None) -> torch.Tensor:
        """Off-diagonal sparsity with PSD constraint."""
        sparsity = torch.mean(torch.abs(d_res))
        
        if d_psd is None:
            d_psd = d_res
        
        d_psd_sym = 0.5 * (d_psd + d_psd.transpose(-1, -2))
        psd_penalty = torch.tensor(0.0, device=d_res.device)
        
        try:
            eigs = torch.linalg.eigvalsh(d_psd_sym)
            eig_min = torch.min(eigs, dim=-1)[0]
            psd_penalty = torch.mean(F.relu(-eig_min))
        except RuntimeError:
            diag_min = torch.min(torch.diagonal(d_psd_sym, dim1=-2, dim2=-1), dim=-1)[0]
            psd_penalty = torch.mean(F.relu(-diag_min))
        
        return 0.25 * sparsity + 0.75 * psd_penalty

    def l_offdiag_dissipation(
        self,
        d_res: torch.Tensor | None,
        d_psd: torch.Tensor | None,
        target_frac: float = 0.3,
    ) -> torch.Tensor:
        """Promote RESONANCE-MEDIATED OFF-DIAGONAL dissipation — the contribution
        beyond standard metriplectic NNs, which dissipate via diagonal / independent
        modal damping.

        d_res is the off-diagonal (diagonal-removed) part of the PSD resonance operator
        d_psd.  We keep the off-diagonal Frobenius share ||d_res||_F / ||d_psd||_F at or
        above a floor so the operator dissipates by CROSS-MODE resonant coupling rather
        than collapsing to trivial diagonal damping (which would erase the novelty).
        The ratio is scale-invariant, so it cannot be gamed by rescaling, and it is a
        guardrail: it is exactly zero whenever the off-diagonal share is already healthy
        (>= target), and only pulls when the model tries to kill cross-mode coupling.
        d_psd is PSD by construction, so no validity penalty is needed here.
        """
        if d_res is None or d_psd is None:
            dev = d_res.device if d_res is not None else (d_psd.device if d_psd is not None else torch.device('cpu'))
            return torch.tensor(0.0, device=dev)
        d_psd_sym = 0.5 * (d_psd + d_psd.transpose(-1, -2))
        fro_off = d_res.reshape(d_res.shape[0], -1).norm(dim=1)
        fro_tot = d_psd_sym.reshape(d_psd_sym.shape[0], -1).norm(dim=1).clamp_min(1e-8)
        offdiag_frac = fro_off / fro_tot
        return torch.mean(F.relu(float(target_frac) - offdiag_frac))

    def l_d_res_time(self, d_res_time: torch.Tensor | None, z_pred: torch.Tensor | None = None) -> torch.Tensor:
        """Time-domain loss over D_res(tau).

        Encourages that the integrated time-domain dissipation does not inject energy and
        that temporal kernels are smooth and bounded.
        d_res_time: (bsz, n_taus, latent_dim, latent_dim)
        """
        if d_res_time is None:
            return torch.tensor(0.0, device=z_pred.device if z_pred is not None else torch.device('cpu'))

        # Ensure symmetry and zero diagonal per tau
        d_rt = 0.5 * (d_res_time + d_res_time.transpose(-1, -2))
        diag = torch.diagonal(d_rt, dim1=-2, dim2=-1)
        d_rt = d_rt - torch.diag_embed(diag)

        # compute dissipative work per batch, per tau
        bsz = d_rt.shape[0]
        n_taus = d_rt.shape[1]
        dim = d_rt.shape[-1]

        if z_pred is None:
            z = torch.ones(bsz, dim, device=d_rt.device, dtype=d_rt.dtype)
        else:
            z = z_pred[..., :dim]

        # (bsz, n_taus)
        d_z = torch.einsum("btij,bj->bti", d_rt, z)
        dissip_work = torch.sum(d_z * z.unsqueeze(1), dim=-1)

        # penalty for positive (injecting) work across taus
        pos_pen = torch.mean(F.relu(dissip_work))

        # temporal smoothness of kernels across taus
        if n_taus > 1:
            diffs = torch.diff(d_rt, dim=1)
            smooth_pen = torch.mean(torch.abs(diffs))
        else:
            smooth_pen = torch.tensor(0.0, device=d_rt.device)

        # boundedness penalty (prevent large kernel norms)
        norm_pen = torch.mean(torch.norm(d_rt.view(bsz, n_taus, -1), dim=-1))

        return 0.6 * pos_pen + 0.2 * smooth_pen + 0.2 * (norm_pen * 1e-3)

    def l_delta_s(self, s_universal: torch.Tensor) -> torch.Tensor:
        """Regularizer: keep universal shift small."""
        return torch.mean(s_universal ** 2)

    def l_sut_align(self, mode_vectors: torch.Tensor, gb_ratio: torch.Tensor | None = None) -> torch.Tensor:
        """SUT alignment: spectral universality and mode orthogonality."""
        gram = torch.matmul(mode_vectors, mode_vectors.T)
        gram_ideal = torch.eye(mode_vectors.shape[0], device=mode_vectors.device, dtype=mode_vectors.dtype)
        ortho_error = torch.norm(gram - gram_ideal, p='fro')

        # Encourage bounded, diverse mode energies.
        mode_energy = torch.mean(torch.sum(mode_vectors ** 2, dim=-1))
        diversity = torch.mean(torch.abs(gram - torch.diag(torch.diagonal(gram))))

        if gb_ratio is not None:
            ortho_error = ortho_error * torch.mean(torch.as_tensor(gb_ratio, device=mode_vectors.device, dtype=mode_vectors.dtype))

        return ortho_error + 0.1 * mode_energy + 0.1 * diversity

    def l_time_freq_consistency(
        self,
        d_res: torch.Tensor | None,
        d_res_time: torch.Tensor | None,
    ) -> torch.Tensor:
        """Consistency between static D_res and the integrated time-domain D_res(tau)."""
        if d_res is None or d_res_time is None:
            return torch.tensor(0.0, device=d_res.device if d_res is not None else (d_res_time.device if d_res_time is not None else torch.device('cpu')))

        d_static = 0.5 * (d_res + d_res.transpose(-1, -2))
        d_time = 0.5 * (d_res_time + d_res_time.transpose(-1, -2))
        # Compare mean absolute structure across taus to the static residual
        d_time_mean = torch.mean(d_time, dim=1)
        return F.mse_loss(d_time_mean, d_static, reduction='mean')

    def l_physics(self, z_pred: torch.Tensor) -> torch.Tensor:
        """Latent boundedness."""
        z_norm = torch.norm(z_pred, dim=-1)
        max_allowed = 10.0
        return torch.mean(F.relu(z_norm - max_allowed))

    def l_reg(self, model_params: torch.Tensor) -> torch.Tensor:
        """L2 weight regularization."""
        # If a tensor was passed accidentally (legacy usage), handle it safely
        try:
            if isinstance(model_params, torch.Tensor):
                return torch.sum(model_params ** 2)
        except Exception:
            pass

        # Default: compute L2 over model parameters of the associated model
        try:
            params = list(self.model.parameters())
            if not params:
                return torch.tensor(0.0)
            device = params[0].device
        except Exception:
            device = torch.device('cpu')

        total = torch.tensor(0.0, device=device)
        count = 0
        for p in self.model.parameters():
            if p is None:
                continue
            total = total + torch.sum(p ** 2)
            count += p.numel()
        if count == 0:
            return torch.tensor(0.0, device=device)
        # return mean squared (L2 per-parameter) to keep scale consistent
        return total / float(max(count, 1))

    def forward(
        self,
        x_true: torch.Tensor,
        x_pred: torch.Tensor,
        z_true: torch.Tensor,
        z_pred: torch.Tensor,
        d_total: torch.Tensor,
        d_res: torch.Tensor,
        d_res_time: torch.Tensor | None = None,
        epoch: int = 0,
        max_epochs: int = 300,
        geom_pred: torch.Tensor | None = None,
        geom_target: torch.Tensor | None = None,
        s_matrix: torch.Tensor | None = None,
        d_psd: torch.Tensor | None = None,
        shared_private_penalty: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tier-2 loss (kinetics primary).
        
        L_total = 1.0  * L_data (0.7 kinetics + 0.3 geometry + kinetics_consistency)
                + 0.10 * L_energy (ramp 0-100)
                + 0.10 * L_dissip (ramp 0-100)
                + 0.05 * L_snt (ramp 20-150)
                + 0.05 * L_jarzy (ramp 50-200)
                + 0.01 * L_d_res_sparse
                + 0.01 * L_delta_s
                + 0.05 * L_sut_align (ramp 100+)
                + 0.10 * L_physics
                + 1e-5 * L_reg
        """
        def linear_ramp(epoch_i: int, start: int, end: int) -> float:
            """Piecewise linear ramp that is exactly zero before `start`."""
            if epoch_i <= start:
                return 0.0
            if epoch_i >= end:
                return 1.0
            return float((epoch_i - start) / max(end - start, 1))

        losses = {}

        kinetics_weight = float(self.config.get("kinetics_weight", 0.55))
        geometry_weight = float(self.config.get("geometry_weight", 0.20))
        kinetics_cons_weight = float(self.config.get("kinetics_cons_weight", 0.08))
        # keep data term balanced even if custom weights are passed
        if kinetics_weight + geometry_weight <= 0:
            kinetics_weight, geometry_weight = 0.70, 0.30

        # PRIMARY DATA LOSS: KINETICS 70% + GEOMETRY 30%
        l_koopman_val = self.l_koopman(x_true, x_pred)
        l_eval_kin = self.l_eval_aligned_kinetics(x_true, x_pred)
        l_latent_align = self.l_latent_alignment(z_true, z_pred)
        losses['l_koopman'] = l_koopman_val
        losses['l_eval_aligned_kinetics'] = l_eval_kin
        losses['l_latent_alignment'] = l_latent_align
        
        l_kinetics_cons = self.l_kinetics_consistency(x_pred)
        losses['l_kinetics_consistency'] = l_kinetics_cons
        losses['l_profile_bounds'] = self.l_profile_bounds(x_true, x_pred) * float(self.config.get("profile_bounds_weight", 0.02))
        
        # Weight kinetics as the primary objective, with a smaller geometry fraction and explicit
        # normalized MAE/NMSE penalties on kinetics.
        if geom_pred is not None and geom_target is not None:
            l_geom_val = self.l_geometry(geom_pred, geom_target)
            l_data = (
                kinetics_weight * l_koopman_val
                + geometry_weight * l_geom_val
                + kinetics_cons_weight * l_kinetics_cons
            )
        else:
            l_geom_val = torch.tensor(0.0, device=x_true.device)
            l_data = kinetics_weight * l_koopman_val + kinetics_cons_weight * l_kinetics_cons

        nrmse_weight = float(self.config.get("nrmse_weight", 0.0))
        nmae_weight = float(self.config.get("nmae_weight", 0.0))
        latent_align_weight = float(self.config.get("latent_align_weight", 0.10))
        # Norm consistency is diagnostics-only now; never contribute it to optimization.
        norm_align_weight = 0.0
        losses['l_eval_nrmse'] = l_eval_kin * nrmse_weight
        losses['l_eval_nmae'] = l_eval_kin * nmae_weight
        losses['l_latent_alignment_weighted'] = l_latent_align * latent_align_weight
        l_norm_consistency = self.l_norm_consistency(x_true, x_pred, z_true, z_pred)
        losses['l_norm_consistency'] = l_norm_consistency
        losses['l_norm_consistency_weighted'] = l_norm_consistency * norm_align_weight
        l_data = l_data + losses['l_eval_nrmse'] + losses['l_eval_nmae'] + losses['l_latent_alignment_weighted']
        
        losses['l_geometry'] = l_geom_val
        losses['l_data'] = l_data * 1.0

        # Physics constraints, ramped in over training: data loss first, then energy and
        # dissipation-orthogonality, then SNT and latent boundedness, then Jarzynski and SUT
        # alignment. Ramp epochs are configurable.
        energy_base = float(self.config.get("energy_weight_base", 0.04))
        e_start = int(self.config.get("energy_ramp_start", 20))
        e_end = int(self.config.get("energy_ramp_end", 140))
        energy_weight = energy_base * linear_ramp(epoch, start=e_start, end=e_end)
        losses['l_energy'] = self.l_energy(z_true, z_pred, s_matrix=s_matrix) * energy_weight

        dissip_base = float(self.config.get("dissip_weight_base", 0.04))
        d_start = int(self.config.get("dissip_ramp_start", 20))
        d_end = int(self.config.get("dissip_ramp_end", 140))
        dissip_weight = dissip_base * linear_ramp(epoch, start=d_start, end=d_end)
        losses['l_dissip'] = self.l_dissip(d_total, z_pred=z_pred) * dissip_weight

        snt_weight = float(self.config.get("snt_weight_base", 0.01)) * linear_ramp(epoch, start=40, end=180)
        losses['l_snt'] = self.l_snt(d_res, d_psd=d_psd) * snt_weight

        # Tier-3: time-domain D_res penalties (always off during data-only phase)
        d_res_time_weight = float(self.config.get("d_res_time_weight", 0.005))
        time_freq_weight = float(self.config.get("time_freq_weight", 0.005))
        losses['l_d_res_time'] = self.l_d_res_time(d_res_time, z_pred=z_pred) * d_res_time_weight
        losses['l_time_freq_consistency'] = self.l_time_freq_consistency(d_res, d_res_time) * time_freq_weight

        jarzy_weight = float(self.config.get("jarzy_weight_base", 0.01)) * linear_ramp(epoch, start=60, end=240)
        losses['l_jarzy'] = self.l_jarzy(x_true, x_pred, d_total, z_pred) * jarzy_weight

        # Hard-sample kinetics penalty using d_res as a difficulty signal: reduce kinetics error more
        # aggressively where d_res is larger.
        try:
            hard_kin = self.l_koopman_hard_samples(x_true, x_pred, d_res)
        except Exception:
            hard_kin = torch.tensor(0.0, device=x_true.device if x_true is not None else torch.device('cpu'))

        enable_dres_hard_kin = bool(self.config.get("enable_dres_hard_kinetics", False))
        hard_weight_max = float(self.config.get("hard_kinetics_weight", 0.08))
        # Hard kinetics starts at epoch 40 (after data fit begins to establish)
        hard_weight = hard_weight_max * linear_ramp(epoch, start=40, end=160)
        losses['l_dres_hard_kinetics'] = hard_kin * hard_weight if enable_dres_hard_kin else torch.tensor(0.0, device=x_true.device)

        # Explicitly surface the hard kinetics term so training logs show it separately.
        losses['l_hard_kinetics_raw'] = hard_kin

        # REGULARIZATION (constant)
        losses['l_d_res_sparse'] = self.l_d_res_sparse(d_res, d_psd=d_psd) * float(self.config.get("d_res_sparse_weight", 0.01))

        # Off-diagonal dissipation guardrail: keep cross-mode resonant coupling as a real share of the
        # dissipation operator rather than letting it collapse to diagonal damping. Ramps in with the
        # rest of the physics.
        offdiag_w = float(self.config.get("offdiag_dissip_weight", 0.02)) * linear_ramp(
            epoch,
            start=int(self.config.get("dissip_ramp_start", 3)),
            end=int(self.config.get("dissip_ramp_end", 30)),
        )
        losses['l_offdiag_dissipation'] = self.l_offdiag_dissipation(
            d_res, d_psd, target_frac=float(self.config.get("offdiag_target_frac", 0.3))
        ) * offdiag_w
        
        if shared_private_penalty is not None:
            losses['l_delta_s'] = shared_private_penalty * float(self.config.get("delta_s_weight", 0.01))
        else:
            losses['l_delta_s'] = torch.tensor(0.0, device=x_true.device)
        
        physics_base = float(self.config.get("physics_weight_base", 0.03))
        # Latent boundedness ramps in after energy/dissip are established
        physics_weight = physics_base * linear_ramp(epoch, start=40, end=200)
        losses['l_physics'] = self.l_physics(z_pred) * physics_weight
        losses['l_reg'] = self.l_reg(None) * 1e-5

        # SUT alignment: ramp configurable so short/fast runs can enforce universality too
        # (default 60->220; the fast protocol pulls it early so ablations live in a SUT-on regime).
        sut_weight = float(self.config.get("sut_weight_base", 0.01)) * linear_ramp(
            epoch,
            start=int(self.config.get("sut_ramp_start", 60)),
            end=int(self.config.get("sut_ramp_end", 220)),
        )
        try:
            if hasattr(self.model, 'rmmd') and hasattr(self.model.rmmd, 'kernel'):
                mode_vectors = self.model.rmmd.kernel.mode_vectors
            else:
                mode_vectors = torch.eye(10, device=x_true.device, dtype=x_true.dtype)
            gb_ratio = None
            if hasattr(self.model, 'delta_s_machines'):
                try:
                    gb_ratio = torch.stack([torch.norm(p).detach() for p in self.model.delta_s_machines]).mean()
                except Exception:
                    gb_ratio = None
            losses['l_sut_align'] = self.l_sut_align(mode_vectors, gb_ratio=gb_ratio) * sut_weight
        except:
            losses['l_sut_align'] = torch.tensor(0.0, device=x_true.device)

        losses['total'] = (
            losses['l_data']
            + losses['l_energy']
            + losses['l_dissip']
            + losses['l_snt']
            + losses['l_jarzy']
            + losses.get('l_d_res_time', torch.tensor(0.0, device=x_true.device))
            + losses.get('l_time_freq_consistency', torch.tensor(0.0, device=x_true.device))
            + losses.get('l_dres_hard_kinetics', torch.tensor(0.0, device=x_true.device))
            + losses['l_d_res_sparse']
            + losses.get('l_offdiag_dissipation', torch.tensor(0.0, device=x_true.device))
            + losses['l_delta_s']
            + losses['l_sut_align']
            + losses['l_physics']
            + losses['l_reg']
        )
        
        return losses
