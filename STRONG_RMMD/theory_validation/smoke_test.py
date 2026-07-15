#!/usr/bin/env python3
"""CPU smoke test for the phase6 scripts: synthetic dataset + tiny checkpoints, then run
all three scripts end-to-end and assert the report JSONs exist with expected keys.
Validates schema/flow only (not physics). Run:  python STRONG_RMMD/theory_validation/smoke_test.py
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_sample(rng, machine: str, T: int = 30):
    ni_traj = rng.normal(0, 1, (T, 40)).astype(np.float32).cumsum(axis=0) * 0.02
    ni_t0 = rng.normal(0, 1, 40).astype(np.float32)
    geom = rng.normal(0, 1, (40, 66)).astype(np.float32) * 0.1
    steps = np.array([1, 10, 20], dtype=np.int64)
    return {
        "machine": machine,
        "y_seq": ni_traj[steps - 1].copy(),
        "geom_seq": np.stack([geom] * len(steps)),
        "target_steps": steps,
        "target_mask": np.ones(len(steps), dtype=bool),
        "pre_shot_context": rng.normal(0, 1, 1280).astype(np.float32),
        "limiter_geometry_tensor": geom + 0.01,
        "ni_t0": ni_t0,
        "geom_t0": geom,
        "ni_traj": ni_traj,
        "geom_traj": np.stack([geom] * T) + rng.normal(0, 0.01, (T, 40, 66)).astype(np.float32),
        "drivers_traj": np.abs(rng.normal(0, 1, (T, 8))).astype(np.float32),
        "state_t0": {"pre_shot_inputs": {"scalars": {
            "BTDIA": 2.0 + rng.uniform(-0.5, 0.5), "RAXIS": 1.7, "RMAJB": 0.6,
            "TE": 2.0, "Q95": 3.5, "PINJ": float(np.abs(rng.normal(2e6, 1e6)))}}},
    }


def main() -> int:
    strong = REPO / "STRONG_RMMD"
    for p in (str(strong), str(strong / "data_io"), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    rc = _import_module(strong / "training" / "rmmd_train_eval_impl.py", "rmmd_train_eval_impl")

    tmp = Path(tempfile.mkdtemp(prefix="phase6_smoke_"))
    rng = np.random.default_rng(0)
    machines = ["D3D", "KSTR", "HL2A", "NSTX", "CMOD"]
    norm = {"kinetic_profiles.NI": {"mean": 0.0, "std": 1.0},
            "geometry_tensor": {"mean": 0.0, "std": 1.0}}
    payload = {"data": [make_sample(rng, m) for m in machines for _ in range(4)],
               "normalization_stats": norm}
    payload_east = {"data": [make_sample(rng, "EAST") for _ in range(6)],
                    "normalization_stats": norm}
    ind = tmp / "synth_test.pt"; east = tmp / "synth_east.pt"
    torch.save(payload, ind); torch.save(payload_east, east)

    cfg = dict(state_dim=40, latent_dim=64, latent_profile=32, latent_geom=32,
               machine_embedding_dim=8, n_harmonics=4, use_transport_step=True,
               model_type="rmmd", machine_names=machines)
    for label, abl in (("full", False), ("abl_dres", True)):
        model = rc._make_model(machines, state_dim=40, latent_dim=64, latent_profile=32,
                               latent_geom=32, machine_embedding_dim=8, n_harmonics=4,
                               use_transport_step=True, ablate_dres=abl)
        d = tmp / label; d.mkdir()
        rc._torch_save_checkpoint(
            {"model_state": model.state_dict(),
             "config": {**cfg, "ablate_dres": abl},
             "normalization_stats": norm},
            d / "checkpoint_best.pt")

    env_py = sys.executable
    runs = [
        ("extrap", [env_py, str(HERE / "extrap_strong.py"),
                    "--indist-data", str(ind), "--east-data", str(east),
                    "--models", f"full={tmp/'full'}", f"abl_dres={tmp/'abl_dres'}",
                    "--reference", "full", "--horizons", "1", "5", "10", "20",
                    "--device", "cpu", "--out", str(tmp / "extrap.json")],
         tmp / "extrap.json", ["models", "strong_fit", "zero_shot_ablation_table"]),
        ("sut", [env_py, str(HERE / "sut_confirmation.py"),
                 "--checkpoint", str(tmp / "full"), "--indist-data", str(ind),
                 "--east-data", str(east), "--device", "cpu",
                 "--compare-checkpoint", str(tmp / "abl_dres"),
                 "--max-shots-per-machine", "3", "--n-perm", "50", "--jacobian",
                 "--out", str(tmp / "sut.json")],
         tmp / "sut.json", ["families", "verdict", "shared_private", "resonance_landscape",
                            "resonance_landscape_universality", "sut_loss_effect",
                            "sut_extrapolation_link"]),
        ("theorems", [env_py, str(HERE / "theorems_validation.py"),
                      "--checkpoint", str(tmp / "full"),
                      "--abl-dres-checkpoint", str(tmp / "abl_dres"),
                      "--test-data", str(ind), "--device", "cpu",
                      "--max-shots", "12", "--max-horizon", "20",
                      "--out", str(tmp / "theorems.json")],
         tmp / "theorems.json", ["EDT", "GIT", "RODEA", "PCT", "EBK", "gates_summary"]),
    ]
    failures = []
    for name, cmd, outp, keys in runs:
        print(f"\n===== smoke: {name} =====", flush=True)
        r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=1800)
        tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-25:])
        print(tail)
        if r.returncode != 0 or not outp.exists():
            failures.append(name); continue
        rep = json.loads(outp.read_text())
        missing = [k for k in keys if k not in rep]
        if missing:
            failures.append(f"{name} (missing keys {missing})")
        else:
            print(f"[smoke] {name}: OK ({outp})")
    print("\n===== SMOKE RESULT =====")
    print("ALL PASS" if not failures else f"FAILURES: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
