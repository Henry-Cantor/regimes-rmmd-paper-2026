#!/usr/bin/env python3
"""SUT (Spectral Universality Theorem) test.

Loads a trained RMMD checkpoint, groups t=0 shots by machine, computes each machine's gyro-Bohm normalized
resonance frequencies, and measures whether the learned latent Koopman/dissipation spectrum is consistent
across machines. Run with --help.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PHASE3 = ROOT / "STRONG_RMMD" / "training"
sys.path.insert(0, str(PHASE3))
sys.path.insert(0, str(ROOT / "STRONG_RMMD"))

import rmmd_train_eval_impl as R  # noqa: E402
from strong_rmmd.resonance_frequencies import compute_resonance_frequencies  # noqa: E402
from strong_rmmd.sut_analysis import run_sut_test  # noqa: E402


def _load_model(checkpoint: Path, device: str, fallback_machines: List[str]):
    ck = R._torch_load_checkpoint_any(checkpoint, map_location="cpu")
    if isinstance(ck, dict) and "model_state" in ck:
        state_dict = ck["model_state"]
        normalization_stats = ck.get("normalization_stats", {})
        config = ck.get("config", {})
        machine_names = config.get("machine_names") or fallback_machines
        dims = R._infer_model_dimensions(state_dict)
        state_dim = int(config.get("state_dim", dims["state_dim"]))
        latent_dim = int(config.get("latent_dim", dims["latent_dim"]))
        latent_profile = int(config.get("latent_profile", dims["latent_profile"]))
        latent_geom = int(config.get("latent_geom", dims["latent_geom"]))
        machine_embedding_dim = int(config.get("machine_embedding_dim", dims["machine_embedding_dim"]))
        n_harmonics = int(config.get("n_harmonics", dims["n_harmonics"]))
    else:
        state_dict = ck
        normalization_stats = {}
        machine_names = fallback_machines
        dims = R._infer_model_dimensions(state_dict)
        state_dim = dims["state_dim"]
        latent_dim = dims["latent_dim"]
        latent_profile = dims["latent_profile"]
        latent_geom = dims["latent_geom"]
        machine_embedding_dim = dims["machine_embedding_dim"]
        n_harmonics = dims["n_harmonics"]

    model = R._make_model(
        sorted(set(machine_names)),
        state_dim=state_dim,
        latent_dim=latent_dim,
        latent_profile=latent_profile,
        latent_geom=latent_geom,
        machine_embedding_dim=machine_embedding_dim,
        n_harmonics=n_harmonics,
    ).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        R._load_compatible_state_dict(model, state_dict)
    model.eval()
    return model, normalization_stats


def _group_samples_by_machine(payloads: List[Path], normalization_stats, max_per_machine: int):
    by_machine: Dict[str, List[dict]] = defaultdict(list)
    for path in payloads:
        payload = R._load_phase0_dataset(path)
        view = R.CompactRolloutDataset(
            payload, max(R.DIRECT_COMPACT_HORIZONS), normalization_stats=normalization_stats
        )
        for idx in range(len(view)):
            sample = view.get_sample(idx)
            m = str(sample.get("machine", "UNKNOWN"))
            if len(by_machine[m]) >= max_per_machine:
                continue
            by_machine[m].append(sample)
    return by_machine


def _machine_omegas(samples, normalization_stats):
    """Mean GB-normalized (omega_t, omega_d) and extras for a machine's shots."""
    ots, ods, gbs, rstars = [], [], [], []
    for s in samples:
        ni_t0 = s.get("ni_t0")
        if not isinstance(ni_t0, torch.Tensor):
            ni_t0 = torch.zeros(40)
        ni_denorm = R._denormalize_ni_batch(ni_t0.reshape(1, -1).cpu(), normalization_stats)[0].numpy()
        scalars = s.get("pre_shot_scalars", {}) or {}
        freq = compute_resonance_frequencies({"NI": ni_denorm}, str(s.get("machine", "UNKNOWN")), scalars)
        ots.append(freq["omega_t"]); ods.append(freq["omega_d"])
        gbs.append(freq.get("omega_gb", 0.0)); rstars.append(freq.get("rho_star", 0.0))
    return (
        (float(np.mean(ots)) if ots else 1.0, float(np.mean(ods)) if ods else 1.0),
        {"omega_gb": float(np.mean(gbs)) if gbs else 0.0, "rho_star": float(np.mean(rstars)) if rstars else 0.0},
    )


def _plot(result, out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot skipped: {exc}]")
        return

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: eigenvalues in the complex plane, per machine (unit circle = neutral).
    theta = np.linspace(0, 2 * np.pi, 200)
    ax0.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.5)
    cmap = plt.get_cmap("tab10")
    for i, (m, spec) in enumerate(result.spectra.items()):
        eig = spec.eigenvalues
        ax0.scatter(eig.real, eig.imag, s=10, alpha=0.6, color=cmap(i % 10), label=m)
    ax0.set_title("Learned latent spectrum per machine")
    ax0.set_xlabel("Re(λ)"); ax0.set_ylabel("Im(λ)")
    ax0.set_aspect("equal", "box"); ax0.legend(fontsize=8, loc="upper right")

    # Right: per-mode sigma/mu, real vs null, with the 0.20 SUT gate.
    som = result.sigma_over_mu
    nsom = result.null_sigma_over_mu
    k = np.arange(1, len(som) + 1)
    ax1.bar(k - 0.2, som, width=0.4, label="GB-normalized (real)", color="#2c7fb8")
    if nsom.size:
        kk = np.arange(1, len(nsom) + 1)
        ax1.bar(kk + 0.2, nsom, width=0.4, label="permuted-ω null", color="#d95f0e", alpha=0.8)
    ax1.axhline(result.threshold, color="k", ls="--", lw=1.0, label=f"SUT gate ={result.threshold}")
    ax1.set_title(
        f"Cross-machine mode dispersion  (real {result.frac_below_threshold:.0%} < gate, "
        f"null {result.null_frac_below_threshold:.0%})"
    )
    ax1.set_xlabel("mode index (sorted by |frequency|)"); ax1.set_ylabel("σ/μ across machines")
    ax1.legend(fontsize=8)

    fig.suptitle(
        f"SUT universality — {'PASS' if result.passed else 'NOT PASSED'} "
        f"({len(result.machines)} machines, top-{result.n_modes} modes)",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=200)
    print(f"wrote: {out_png}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default=None, help="Combined test payload (multi-machine).")
    p.add_argument("--machine-data", action="append", default=[],
                   help="NAME=PATH for a per-machine payload (repeatable).")
    p.add_argument("--out", default="sut_report")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-per-machine", type=int, default=32)
    p.add_argument("--n-modes", type=int, default=None, help="Top-N0 modes (default latent_dim//4).")
    p.add_argument("--threshold", type=float, default=0.20)
    args = p.parse_args()

    payload_paths: List[Path] = []
    if args.data:
        payload_paths.append(Path(args.data))
    for spec in args.machine_data:
        if "=" in spec:
            payload_paths.append(Path(spec.split("=", 1)[1]))
    if not payload_paths:
        print("ERROR: provide --data and/or --machine-data NAME=PATH")
        return 1

    # Peek machine names for model construction fallback.
    fallback = []
    for path in payload_paths:
        try:
            fallback += R._collect_machine_names(R._load_phase0_dataset(path))
        except Exception:
            pass
    fallback = sorted(set(fallback)) or ["UNKNOWN"]

    model, normalization_stats = _load_model(Path(args.checkpoint), args.device, fallback)

    by_machine = _group_samples_by_machine(payload_paths, normalization_stats, args.max_per_machine)
    machines = [m for m in sorted(by_machine) if len(by_machine[m]) >= 2]
    if len(machines) < 2:
        print(f"ERROR: need >=2 machines with >=2 shots; got {[ (m, len(by_machine[m])) for m in by_machine ]}")
        return 1
    print(f"Machines: {[(m, len(by_machine[m])) for m in machines]}")

    machine_batches: Dict[str, dict] = {}
    machine_omegas: Dict[str, tuple] = {}
    machine_extras: Dict[str, dict] = {}
    for m in machines:
        samples = by_machine[m]
        batch = R._compact_rollout_collate(samples)
        batch["geometry_tensor"] = batch.get("geom_t0")  # current geom = t0 geom for the spectrum probe
        machine_batches[m] = batch
        (ot, od), extras = _machine_omegas(samples, normalization_stats)
        machine_omegas[m] = (ot, od)
        machine_extras[m] = extras

    result = run_sut_test(
        model,
        machine_batches,
        machine_omegas,
        device=args.device,
        n_modes=args.n_modes,
        threshold=args.threshold,
        machine_extras=machine_extras,
    )

    out_json = Path(str(args.out) + ".json")
    out_png = Path(str(args.out) + ".png")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")
    print(f"wrote: {out_json}")
    _plot(result, out_png)

    s = result.summary()
    print("\n==== SUT SUMMARY ====")
    print(f"machines           : {s['machines']}")
    print(f"top-N0 modes       : {s['n_modes']}")
    print(f"median σ/μ (real)  : {s['median_sigma_over_mu']:.3f}")
    print(f"median σ/μ (null)  : {s['null_median_sigma_over_mu']:.3f}")
    print(f"modes < {result.threshold:.2f} (real): {s['frac_modes_below_threshold']:.0%}")
    print(f"modes < {result.threshold:.2f} (null): {s['null_frac_modes_below_threshold']:.0%}")
    print(f"SUT PASS           : {s['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
