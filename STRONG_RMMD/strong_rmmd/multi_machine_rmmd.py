"""Multi-machine RMMD model — single-step dynamics for autoregressive rollout.

Predictive-TRANSP surrogate (initial-value problem).
Inputs at each step: current NI(t) + current geometry(t) + pre-shot context (fixed) +
                     limiter geometry (fixed) + machine id.
Output: NI(t+1) + geometry(t+1)  (one step forward).

Called autoregressively from t=0, feeding predictions back in; metrics are scored
at the report horizons. The t=0 NI and geometry are known inputs (no t>0 ground
truth is fed in during rollout). Cross-machine generalization comes from
conditioning the dynamics on geometry and the machine embedding.

Components:
- pre_shot_initializer      : maps 1280-dim pre-shot context → latent bias (fixed per shot)
- state_legacy_encoder      : maps current NI (40-dim) → latent  (updated each step)
- geometry_encoder          : maps current flux surface geometry → latent_geom context
- limiter_geometry_encoder  : maps vessel/limiter geometry → latent_geom context (fixed)
- RMMD block                : one physics-constrained latent dynamics step
- compact_ni_decoder        : latent → NI(t+1) prediction
- geometry_decoder_from_geom: latent_geom → geometry delta (added to current geometry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch import nn

from strong_rmmd.rmmd_block import RMMDBlock, RMMDOutput
from strong_rmmd.transport import ConservativeTransportStep


# NI is the only trained profile.
PROFILE_ORDER = ["NI"]
PRE_SHOT_CONTEXT_DIM = 1280


@dataclass
class MultiMachineOutput:
    x_next: torch.Tensor
    machine_embedding: torch.Tensor
    rmmd: RMMDOutput
    profile_pred: torch.Tensor | None = None
    geometry_pred: torch.Tensor | None = None
    latent_next: torch.Tensor | None = None
    transport: dict | None = None     # C: {D, v, S, Vp, L} from the conservative transport step


class FlexibleMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        hidden_dim = max(latent_dim * 4, 256)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.reshape(x.shape[0], -1)
        if x.shape[-1] < self.input_dim:
            pad = torch.zeros(x.shape[0], self.input_dim - x.shape[-1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        elif x.shape[-1] > self.input_dim:
            x = x[:, : self.input_dim]
        return self.net(x)


class FlexibleProfileDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        hidden_dim = max(latent_dim * 4, 256)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, self.output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FlexibleMatrixEncoder(nn.Module):
    def __init__(self, rows: int, cols: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = int(rows * cols)
        hidden_dim = max(latent_dim * 4, 256)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.reshape(x.shape[0], -1)
        if x.shape[-1] < self.input_dim:
            pad = torch.zeros(x.shape[0], self.input_dim - x.shape[-1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        elif x.shape[-1] > self.input_dim:
            x = x[:, : self.input_dim]
        return self.net(x)


def _unique_machines(machine_names: Sequence[str] | None) -> List[str]:
    names = [str(name) for name in (machine_names or [])]
    if not names:
        names = ["UNKNOWN"]
    if "UNKNOWN" not in names:
        names.append("UNKNOWN")
    return sorted(set(names))


class MultiMachineRMMD(nn.Module):
    def __init__(
        self,
        state_dim: int = 240,
        latent_dim: int = 512,
        machine_names: Sequence[str] | None = None,
        machine_embedding_dim: int = 32,
        n_harmonics: int = 4,
        latent_profile: int = 160,
        latent_geom: int = 160,
        n_radial: int = 40,
        n_psi: int = 40,
        n_fourier: int = 66,
        use_transport_step: bool = True,
        n_drivers: int = 8,
        ablate_drivers: bool = False,
        ablate_geometry: bool = False,
        ablate_dres: bool = False,
        drivergate: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.n_drivers = int(n_drivers)
        # ABLATIONS (each removes one novel component; ablate_transport == use_transport_step=False):
        #   drivers  -> zero the time-resolved driver inputs (back to static-context-only)
        #   geometry -> zero geometry context + geom->latent coupling
        #   dres     -> diagonal-only dissipation in the NI RMMD block (no resonant off-diagonal)
        self.ablate_drivers = bool(ablate_drivers)
        self.ablate_geometry = bool(ablate_geometry)
        self.ablate_dres = bool(ablate_dres)
        # Driver-gate (--drivergate): optional learned per-step gate that relaxes the block's contraction
        # on transients while staying contractive on quiet shots. Off by default.
        self.drivergate = bool(drivergate)
        self.last_drivergate = 0.0   # diagnostic: mean gate value on the last forward

        self.machine_names = _unique_machines(machine_names)
        self.machine_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(self.machine_names)}
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.latent_profile = int(latent_profile)
        self.latent_geom = int(latent_geom)
        self.n_radial = int(n_radial)
        self.n_psi = int(n_psi)
        self.n_fourier = int(n_fourier)
        self.machine_embedding_dim = int(machine_embedding_dim)

        self.machine_embedding = nn.Embedding(len(self.machine_names), self.machine_embedding_dim)
        self.s_universal = nn.Parameter(torch.zeros(self.latent_dim))
        self.delta_s_machines = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.latent_dim, self.latent_dim)) for _ in self.machine_names]
        )

        self.state_legacy_encoder = FlexibleMLPEncoder(self.state_dim, self.latent_dim)
        self.koopman_encoder = nn.Linear(self.latent_dim, self.latent_dim)
        self.koopman_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.profile_decoder = FlexibleProfileDecoder(self.latent_dim, self.state_dim)

        self.geometry_encoder = FlexibleMatrixEncoder(self.n_psi, self.n_fourier, self.latent_geom)
        self.limiter_geometry_encoder = FlexibleMatrixEncoder(self.n_psi, self.n_fourier, self.latent_geom)
        self.machine_geometry_encoders = nn.ModuleList(
            [FlexibleMatrixEncoder(self.n_psi, self.n_fourier, self.latent_geom) for _ in self.machine_names]
        )
        self.geometry_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_geom * 2),
            nn.GELU(),
            nn.Linear(self.latent_geom * 2, self.latent_geom),
            nn.GELU(),
            nn.Linear(self.latent_geom, self.n_psi * self.n_fourier),
        )
        self.geometry_decoder_from_geom = nn.Sequential(
            nn.Linear(self.latent_geom, self.latent_geom * 2),
            nn.GELU(),
            nn.Linear(self.latent_geom * 2, self.latent_geom),
            nn.GELU(),
            nn.Linear(self.latent_geom, self.n_psi * self.n_fourier),
        )
        self.compact_geom_feedback = nn.Sequential(
            nn.Linear(self.latent_geom, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.compact_ni_fusion = nn.Sequential(
            nn.Linear(self.latent_dim + self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.compact_ni_norm = nn.LayerNorm(self.latent_dim)
        self.compact_ni_decoder = FlexibleProfileDecoder(self.latent_dim, self.n_radial)
        self.compact_latent_norm = nn.LayerNorm(self.latent_dim)

        self.pre_shot_encoder = nn.Sequential(
            nn.Linear(PRE_SHOT_CONTEXT_DIM, 64),
            nn.GELU(),
            nn.Linear(64, self.latent_profile // 4),
        )
        self.pre_shot_initializer = nn.Sequential(
            nn.Linear(PRE_SHOT_CONTEXT_DIM, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.pre_shot_phys_projector = nn.Sequential(
            nn.Linear(PRE_SHOT_CONTEXT_DIM, 32),
            nn.GELU(),
            nn.Linear(32, 4),
        )
        # Step-size conditioning: dt is a known control input (the time grid is chosen, not
        # future data), letting one dynamics function take correctly-sized steps.
        self.compact_dt_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, self.latent_dim),
        )
        # Time-resolved exogenous drivers (NBI power, plasma current, gas): known control inputs
        # injected into the latent at each step; the raw vector also feeds the transport coefficients.
        self.driver_encoder = nn.Sequential(
            nn.Linear(self.n_drivers, 64),
            nn.GELU(),
            nn.Linear(64, self.latent_dim),
        )
        # Small-init the driver injection so the identity-at-init prior is preserved.
        self._small_init_linear(self.driver_encoder[-1], std=1e-3)

        # Driver-gate head: drivers -> g in [0,1]. The last layer is init to zero weight and negative
        # bias so g ~= 0 at start (a fresh --drivergate model matches the ungated RMMD).
        if self.drivergate:
            self.driver_gate_head = nn.Sequential(
                nn.Linear(self.n_drivers, 32), nn.SiLU(), nn.Linear(32, 1),
            )
            nn.init.zeros_(self.driver_gate_head[-1].weight)
            nn.init.constant_(self.driver_gate_head[-1].bias, -4.0)

        self.global_encoder = nn.Sequential(
            nn.Linear(17, 64),
            nn.GELU(),
            nn.Linear(64, self.latent_profile // 4),
        )
        self.transport_encoder = nn.Sequential(
            nn.Linear(40, self.latent_profile // 2),
            nn.GELU(),
            nn.Linear(self.latent_profile // 2, self.latent_profile // 2),
        )
        self.source_encoder = nn.Sequential(
            nn.Linear(40, self.latent_profile // 2),
            nn.GELU(),
            nn.Linear(self.latent_profile // 2, self.latent_profile // 2),
        )

        context_width = max(192, self.latent_dim // 2)
        self.rmmd_context_proj = nn.Sequential(
            nn.Linear(self.latent_geom + self.machine_embedding_dim, context_width),
            nn.SiLU(),
            nn.Linear(context_width, context_width),
            nn.SiLU(),
            nn.Linear(context_width, context_width),
        )
        self.rmmd_context_norm = nn.LayerNorm(context_width)
        self.geom_context_norm = nn.LayerNorm(self.latent_geom)
        self.geom_to_latent = nn.Sequential(
            nn.Linear(self.latent_geom, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        # Learnable geometry->latent coupling scale (init 0.1 so geometry influences the density dynamics).
        self.geom_to_latent_scale = nn.Parameter(torch.tensor(0.1))
        # Geometry identity prior: gate the predicted geometry delta (init small) so geometry_pred
        # starts near the current geometry.
        self.geom_delta_gate = nn.Parameter(torch.tensor(0.05))
        self.profile_output_gain = nn.Parameter(torch.tensor(8.0))

        self.rmmd = RMMDBlock(
            state_dim=self.latent_dim,
            latent_dim=self.latent_dim,
            context_dim=context_width,
            n_harmonics=n_harmonics,
            use_time_kernel=True,
            n_taus=8,
            ablate_offdiag=self.ablate_dres,
        )
        self.rmmd_geom = RMMDBlock(
            state_dim=self.latent_geom,
            latent_dim=self.latent_geom,
            context_dim=context_width,
            n_harmonics=n_harmonics,
            use_time_kernel=True,
            n_taus=8,
        )

        # Conservative transport step: predicts NI(t+1) as a semi-implicit flux-form continuity step.
        # feat carries the resonance frequencies (omega_t, omega_d) so the diffusivity D is resonance-mediated.
        self.use_transport_step = bool(use_transport_step)
        if self.use_transport_step:
            self.transport_step = ConservativeTransportStep(
                n_radial=self.n_radial, feat_dim=self.latent_dim + 2 + self.n_drivers, dt=1.0,
                s_max=0.3,
            )

        self._reset_output_layers()

    def _small_init_linear(self, layer: nn.Linear, std: float = 1e-2) -> None:
        nn.init.normal_(layer.weight, mean=0.0, std=std)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _reset_output_layers(self) -> None:
        standard_output_layers = [
            self.state_legacy_encoder.net[-1],
            self.koopman_decoder[-1],
            self.profile_decoder.net[-1],
            self.geometry_decoder[-1],
            self.rmmd_context_proj[-1],
            self.rmmd.decoder[-1],
            self.rmmd.encoder[-1],
            self.rmmd_geom.decoder[-1],
            self.rmmd_geom.encoder[-1],
        ]
        for layer in standard_output_layers:
            if isinstance(layer, nn.Linear):
                self._small_init_linear(layer, std=1e-2)
        # Geometry delta decoder starts near-zero so geometry_pred ≈ geom_t0 at init.
        # NRMSE should then start low (reflecting natural flux surface drift)
        # rather than pinning near 1.0 from random predictions.
        geom_delta_final = self.geometry_decoder_from_geom[-1]
        if isinstance(geom_delta_final, nn.Linear):
            self._small_init_linear(geom_delta_final, std=1e-3)
        # NI delta decoder starts near-zero so x_next ≈ ni_current at init (identity
        # prior for short horizons). The model learns the *change* in NI from there.
        ni_delta_final = self.compact_ni_decoder.net[-1]
        if isinstance(ni_delta_final, nn.Linear):
            self._small_init_linear(ni_delta_final, std=1e-3)
        # dt encoder starts near-zero so it begins as a gentle modulation of the latent.
        dt_final = self.compact_dt_encoder[-1]
        if isinstance(dt_final, nn.Linear):
            self._small_init_linear(dt_final, std=1e-2)

    def _machine_indices(self, machine_names: Sequence[str]) -> torch.Tensor:
        indices: List[int] = []
        for name in machine_names:
            indices.append(self.machine_to_idx.get(str(name), self.machine_to_idx.get("UNKNOWN", 0)))
        return torch.tensor(indices, dtype=torch.long)

    def _build_geometry_context(self, batch_data: Dict[str, torch.Tensor], machine_names: Sequence[str]) -> torch.Tensor:
        geometry = batch_data.get("geometry_tensor")
        limiter_geometry = batch_data.get("limiter_geometry_tensor")
        if not isinstance(geometry, torch.Tensor):
            geometry = torch.zeros(len(machine_names), self.n_psi, self.n_fourier, device=self.machine_embedding.weight.device)
        if not isinstance(limiter_geometry, torch.Tensor):
            limiter_geometry = torch.zeros_like(geometry)

        geometry = geometry.to(self.machine_embedding.weight.device)
        limiter_geometry = limiter_geometry.to(self.machine_embedding.weight.device)
        geom_context = self.geometry_encoder(geometry)
        geom_context = geom_context + self.limiter_geometry_encoder(limiter_geometry)

        if self.machine_geometry_encoders:
            machine_idx = self._machine_indices(machine_names).to(geometry.device)
            machine_ctx = []
            for i, idx in enumerate(machine_idx.tolist()):
                machine_ctx.append(self.machine_geometry_encoders[idx](geometry[i : i + 1]).squeeze(0))
            geom_context = geom_context + torch.stack(machine_ctx, dim=0)

        return self.geom_context_norm(torch.nan_to_num(geom_context, nan=0.0, posinf=0.0, neginf=0.0))

    def _full_state_to_latent(self, batch_data: Dict[str, torch.Tensor], machine_names: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t = batch_data.get("x_t")
        if not isinstance(x_t, torch.Tensor):
            raise TypeError("batch_data must contain x_t tensor")
        x_t = x_t.to(self.machine_embedding.weight.device)
        z = self.koopman_encoder(self.state_legacy_encoder(x_t))
        machine_idx = self._machine_indices(machine_names).to(x_t.device)
        machine_emb = self.machine_embedding(machine_idx)
        geom_context = self._build_geometry_context(batch_data, machine_names)
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), machine_emb, geom_context

    def forward(
        self,
        x_t: torch.Tensor | Dict[str, torch.Tensor],
        machine_names: Sequence[str],
        omega_t: torch.Tensor,
        omega_d: torch.Tensor,
        batch_data: Dict[str, torch.Tensor] | None = None,
        apply_physics_projection: bool = False,
    ) -> MultiMachineOutput:
        if isinstance(x_t, dict):
            batch_data = x_t
            x_t = batch_data.get("x_t")  # type: ignore[assignment]

        compact_mode = bool(batch_data is not None and batch_data.get("compact_mode"))
        relax_gate = None   # DRIVER-GATE: per-batch contraction-relax for the NI block (compact path)

        if compact_mode:
            if batch_data is None:
                raise ValueError("compact_mode requires batch_data")
            pre_shot_context = batch_data.get("pre_shot_context")
            limiter_geometry = batch_data.get("limiter_geometry_tensor")
            if not isinstance(pre_shot_context, torch.Tensor) or not isinstance(limiter_geometry, torch.Tensor):
                raise TypeError("compact_mode requires pre_shot_context and limiter_geometry_tensor tensors")

            device = pre_shot_context.device
            machine_idx = self._machine_indices(machine_names).to(device)
            machine_emb = self.machine_embedding(machine_idx)

            # Latent initialization: fuse pre-shot context (fixed per shot) with the
            # CURRENT NI profile (ni_t0 = ni_current in autoregressive rollout).
            # At step 1: ni_t0 = true initial condition.
            # At step T>1: ni_t0 = model's predicted NI from the previous step.
            z_preshot = self.pre_shot_initializer(pre_shot_context.to(device))
            ni_current = batch_data.get("ni_t0")
            compact_ni_base = None
            if isinstance(ni_current, torch.Tensor):
                ni_current = ni_current.to(device)
                compact_ni_base = ni_current.reshape(ni_current.shape[0], -1)[:, : self.n_radial]
                z_preshot = z_preshot + self.state_legacy_encoder(ni_current.reshape(ni_current.shape[0], -1))
            z = self.koopman_encoder(z_preshot)

            # Condition the dynamics on the step size (dt to the next checkpoint).
            # step_dt is expected pre-normalized (e.g. log1p(dt)/log1p(1000)) in [0, 1].
            step_dt = batch_data.get("step_dt")
            if isinstance(step_dt, torch.Tensor):
                z = z + self.compact_dt_encoder(step_dt.to(device).view(-1, 1).to(z.dtype))

            # Time-resolved exogenous drivers at this step: injected into the latent, and the raw vector
            # also feeds the transport coefficients. Absent -> zeros.
            drivers = batch_data.get("drivers")
            if self.ablate_drivers:
                # ABLATION (no-drivers): ignore time-resolved drivers entirely (static context only).
                drivers_vec = torch.zeros(z.shape[0], self.n_drivers, device=device, dtype=z.dtype)
            elif isinstance(drivers, torch.Tensor):
                drivers_vec = drivers.to(device).to(z.dtype).view(z.shape[0], -1)
                k = drivers_vec.shape[-1]
                if k < self.n_drivers:
                    drivers_vec = torch.cat(
                        [drivers_vec, torch.zeros(z.shape[0], self.n_drivers - k, device=device, dtype=z.dtype)],
                        dim=-1,
                    )
                elif k > self.n_drivers:
                    drivers_vec = drivers_vec[:, : self.n_drivers]
                z = z + self.driver_encoder(drivers_vec)
            else:
                drivers_vec = torch.zeros(z.shape[0], self.n_drivers, device=device, dtype=z.dtype)

            if self.drivergate:                          # DRIVER-GATE: open on transients (learned)
                relax_gate = torch.sigmoid(self.driver_gate_head(drivers_vec))
                self.last_drivergate = float(relax_gate.mean().item())

            # Geometry context: limiter (fixed) + CURRENT flux surface geometry.
            # At step T>1 this reflects the previously predicted geometry, so the RMMD
            # dynamics know the current plasma shape as it evolves.
            geom_current = batch_data.get("geometry_tensor")
            if isinstance(geom_current, torch.Tensor):
                geom_context = self.geom_context_norm(
                    self.limiter_geometry_encoder(limiter_geometry.to(device))
                    + self.geometry_encoder(geom_current.to(device))
                )
            else:
                geom_context = self.geom_context_norm(self.limiter_geometry_encoder(limiter_geometry.to(device)))
            if self.ablate_geometry:
                # ABLATION (no-geometry): remove flux-surface geometry information from the model
                # (context + geom->latent coupling both see zeros).
                geom_context = torch.zeros_like(geom_context)

            context = self.rmmd_context_norm(self.rmmd_context_proj(torch.cat([machine_emb, geom_context], dim=-1)))
            source_context = torch.zeros(z.shape[0], self.latent_profile // 2, device=device)
            phys_params = self.pre_shot_phys_projector(pre_shot_context.to(device))
            latent_to_decode = z
        elif batch_data is not None and isinstance(batch_data.get("kinetic_profiles"), torch.Tensor):
            z, machine_emb, geom_context = self._full_state_to_latent(batch_data, machine_names)
            context = self.rmmd_context_norm(self.rmmd_context_proj(torch.cat([machine_emb, geom_context], dim=-1)))
            source_context = torch.zeros(z.shape[0], self.latent_profile // 2, device=z.device)
            phys_params = batch_data.get("pre_shot_context")
            if isinstance(phys_params, torch.Tensor):
                phys_params = self.pre_shot_phys_projector(phys_params.to(z.device))
            else:
                phys_params = torch.ones(z.shape[0], 4, device=z.device)
            latent_to_decode = z
        else:
            if not isinstance(x_t, torch.Tensor):
                raise TypeError("x_t must be a tensor when batch_data is not provided")
            if x_t.shape[0] != len(machine_names):
                raise ValueError("x_t batch size must match number of machine names")
            x_t = x_t.to(self.machine_embedding.weight.device)
            machine_idx = self._machine_indices(machine_names).to(x_t.device)
            machine_emb = self.machine_embedding(machine_idx)
            z = self.koopman_encoder(self.state_legacy_encoder(x_t))
            geom_context = torch.zeros(z.shape[0], self.latent_geom, device=z.device, dtype=z.dtype)
            context = self.rmmd_context_norm(self.rmmd_context_proj(torch.cat([machine_emb, geom_context], dim=-1)))
            source_context = torch.zeros(z.shape[0], self.latent_profile // 2, device=z.device)
            phys_params = torch.ones(z.shape[0], 4, device=z.device)
            latent_to_decode = z

        rmmd_out = self.rmmd(
            x_t=latent_to_decode,
            omega_t=omega_t,
            omega_d=omega_d,
            context=context,
            z_t=latent_to_decode,
            relax_gate=relax_gate,
        )

        z_shifted = rmmd_out.z_next + self.s_universal.unsqueeze(0)
        # Learnable coupling scale keeps geom→latent influence well-conditioned.
        geom_coupling = torch.clamp(self.geom_to_latent_scale, min=0.0, max=1.0)
        z_shifted = z_shifted + geom_coupling * torch.tanh(self.geom_to_latent(geom_context))

        if len(self.delta_s_machines) > 0:
            machine_idx = self._machine_indices(machine_names).to(z_shifted.device)
            delta_stack = torch.stack([self.delta_s_machines[idx] for idx in machine_idx.tolist()], dim=0)
            delta_asym = 0.05 * (delta_stack - delta_stack.transpose(-1, -2))
            z_shifted = z_shifted + torch.einsum("bij,bj->bi", delta_asym, z_shifted)

        geom_rmmd_out = self.rmmd_geom(
            x_t=geom_context,
            omega_t=omega_t,
            omega_d=omega_d,
            context=context,
            z_t=geom_context,
        )
        geom_shifted = geom_rmmd_out.z_next
        # Decode a geometry delta and add it to the provided t=0 geometry (if available).
        geom_delta = self.geometry_decoder_from_geom(geom_shifted).view(geom_shifted.shape[0], self.n_psi, self.n_fourier)
        geom_delta = torch.nan_to_num(geom_delta, nan=0.0, posinf=0.0, neginf=0.0)
        base_geom = None
        if isinstance(batch_data, dict):
            base_geom = batch_data.get("geometry_tensor")
        if not isinstance(base_geom, torch.Tensor):
            base_geom = torch.zeros_like(geom_delta)
        else:
            base_geom = base_geom.to(geom_delta.device)
        geometry_pred = base_geom + torch.clamp(self.geom_delta_gate, 0.0, 1.0) * geom_delta

        z_decoded = self.koopman_decoder(z_shifted)
        if compact_mode:
            geom_feedback = self.compact_geom_feedback(geom_shifted)
            ni_context = self.compact_ni_norm(self.compact_ni_fusion(torch.cat([z_decoded, geom_feedback], dim=-1)))
            profile_pred = self.compact_ni_decoder(ni_context)
            latent_next = 0.1 * self.compact_latent_norm(z_shifted)
        else:
            profile_pred = self.profile_decoder(z_decoded)
            latent_next = z_shifted

        transport_coeffs = None
        if compact_mode:
            # NI is predicted as a residual (delta) from the current NI; the decoder final layer is
            # small-init so ni_delta ~= 0 at start and x_next ~= ni_current (identity at short horizon).
            ni_delta = profile_pred[:, : self.n_radial]
            if self.use_transport_step and compact_ni_base is not None:
                # C: NI(t+1) = conservative semi-implicit transport step on NI(t).
                # feat carries the resonance frequencies so D is resonance-mediated.
                feat = torch.cat(
                    [ni_context,
                     omega_t.reshape(-1, 1).to(ni_context.dtype),
                     omega_d.reshape(-1, 1).to(ni_context.dtype),
                     drivers_vec.to(ni_context.dtype)],
                    dim=-1,
                )
                ni_next, transport_coeffs = self.transport_step(compact_ni_base, feat, relax_gate=relax_gate)
                x_next = torch.clamp(ni_next, min=-8.0, max=8.0)
            elif compact_ni_base is not None:
                x_next = torch.clamp(compact_ni_base + ni_delta, min=-8.0, max=8.0)
            else:
                x_next = torch.clamp(ni_delta, min=-8.0, max=8.0)
        else:
            profile_gain = torch.clamp(torch.nn.functional.softplus(self.profile_output_gain), min=1.0, max=64.0)
            x_next = torch.clamp(profile_gain * profile_pred[:, : self.state_dim], min=-64.0, max=64.0)

        if apply_physics_projection:
            x_next = torch.nan_to_num(x_next, nan=0.0, posinf=0.0, neginf=0.0)

        machine_emb_out = self.machine_embedding(self._machine_indices(machine_names).to(z.device))
        return MultiMachineOutput(
            x_next=torch.nan_to_num(x_next, nan=0.0, posinf=0.0, neginf=0.0),
            machine_embedding=machine_emb_out,
            rmmd=rmmd_out,
            profile_pred=profile_pred,
            geometry_pred=geometry_pred,
            latent_next=torch.nan_to_num(latent_next, nan=0.0, posinf=0.0, neginf=0.0),
            transport=transport_coeffs,
        )

    def spectral_alignment_penalty(self) -> torch.Tensor:
        return torch.mean(self.s_universal ** 2)

    def shared_private_penalty(self) -> torch.Tensor:
        delta_penalty = torch.stack([torch.mean(delta ** 2) for delta in self.delta_s_machines]).mean()
        universal_penalty = torch.mean(self.s_universal ** 2)
        return universal_penalty + 0.1 * delta_penalty


def build_multi_machine_model(machine_names: Sequence[str] | None = None, **kwargs: object) -> MultiMachineRMMD:
    return MultiMachineRMMD(machine_names=machine_names, **kwargs)


__all__ = ["MultiMachineRMMD", "MultiMachineOutput", "build_multi_machine_model"]