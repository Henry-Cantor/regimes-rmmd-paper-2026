#!/usr/bin/env python3
"""Entry point — delegates to the single RMMD model implementation.

There is ONE model: autoregressive, NI-only, compact.  Run:
  python rmmd_train_eval.py train
  python rmmd_train_eval.py eval  --checkpoint <ckpt>
  python rmmd_train_eval.py test  --train-data <data>

Architecture: per-shot unit-step autoregressive rollout from T=0 (dt=1 every step).
- Loss at EVERY integer step 1..T_frontier using dense ni_traj targets.
- Eval/report horizons: 1, 2, 3, 5, 8, 12, 16, 20, 32, 50 (rebuild data_build after change).
- Curriculum starts at T=1 and advances when NI NRMSE < threshold.
- No leakage: model only ever sees T=0 initial conditions at step 1,
  then its own predictions at subsequent steps.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "STRONG_RMMD" / "training" / "rmmd_train_eval_impl.py"

spec = importlib.util.spec_from_file_location("rmmd_corrected", MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load corrected module from {MODULE_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rmmd_eval_wrapper")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RMMD compact NI+geometry training/eval (delegates to corrected implementation)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--train-data",
        default="/scratch/gpfs/USER/strong_rmmd/data_build/dataset_train_compact.pt",
    )
    train_parser.add_argument(
        "--val-data",
        default="/scratch/gpfs/USER/strong_rmmd/data_build/dataset_val_compact.pt",
    )
    train_parser.add_argument(
        "--compact-train-data",
        default="/scratch/gpfs/USER/strong_rmmd/data_build/dataset_train_compact.pt",
    )
    train_parser.add_argument(
        "--compact-val-data",
        default="/scratch/gpfs/USER/strong_rmmd/data_build/dataset_val_compact.pt",
    )
    train_parser.add_argument("--epochs", type=int, default=120)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=5e-5)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--num-workers", type=int, default=8)
    train_parser.add_argument(
        "--checkpoint-dir", default="/scratch/gpfs/USER/models/rmmd_final"
    )
    train_parser.add_argument("--latent-dim", type=int, default=384)
    train_parser.add_argument("--latent-profile", type=int, default=160)
    train_parser.add_argument("--latent-geom", type=int, default=160)
    train_parser.add_argument("--machine-embedding-dim", type=int, default=48)
    train_parser.add_argument("--patience", type=int, default=15)
    train_parser.add_argument("--max-train-shots", type=int, default=None)
    train_parser.add_argument("--max-val-shots", type=int, default=None)
    train_parser.add_argument("--rmmd-warmup-epochs", type=int, default=0)

    # Loss weights (data signal first, physics ramped in later). kinetics_weight dominates. Training and eval
    # both run in normalized space so gradients are balanced and NRMSE is a bounded relative error.
    train_parser.add_argument("--loss-kinetics-weight", type=float, default=0.75)
    train_parser.add_argument("--loss-geometry-weight", type=float, default=0.20)
    train_parser.add_argument("--loss-kinetics-cons-weight", type=float, default=0.05)
    # Non-negativity bound is meaningless in normalized space (NI is legitimately
    # negative there); disabled so it does not bias predictions upward.
    train_parser.add_argument("--loss-profile-bounds-weight", type=float, default=0.0)
    train_parser.add_argument("--loss-hard-kinetics-weight", type=float, default=0.08)
    train_parser.add_argument("--disable-dres-hard-kinetics", action="store_true")
    # Latent alignment DISABLED (0.0). Using future NI to guide the latent is
    # training-time leakage. Primary NRMSE loss provides all gradient needed.
    train_parser.add_argument("--loss-latent-align-weight", type=float, default=0.0)
    # Physics terms ramp in gradually via linear_ramp inside losses.py;
    # these base weights are intentionally small relative to kinetics.
    train_parser.add_argument("--loss-energy-weight-base", type=float, default=0.04)
    train_parser.add_argument("--loss-dissip-weight-base", type=float, default=0.04)
    train_parser.add_argument("--loss-d-res-time-weight", type=float, default=0.005)
    train_parser.add_argument("--loss-time-freq-weight", type=float, default=0.005)
    train_parser.add_argument("--loss-physics-weight-base", type=float, default=0.03)
    train_parser.add_argument("--loss-sut-weight-base", type=float, default=0.01)
    # SUT alignment ramp window (epochs). Default 60->220; --fast-protocol pulls it to ~15->35 so a
    # short run still enforces universality and the extrapolation ablations are in a SUT-on regime.
    train_parser.add_argument("--loss-sut-ramp-start", type=int, default=60)
    train_parser.add_argument("--loss-sut-ramp-end", type=int, default=220)
    train_parser.add_argument("--loss-snt-weight-base", type=float, default=0.01)
    train_parser.add_argument("--loss-d-res-sparse-weight", type=float, default=0.005)
    train_parser.add_argument("--loss-delta-s-weight", type=float, default=0.005)
    train_parser.add_argument("--loss-jarzy-weight-base", type=float, default=0.01)

    # Curriculum advancement gate: advance the horizon only when val NI NRMSE is below this threshold, so a
    # frontier is mastered before moving to a harder one. Default is the T20 target (0.05).
    train_parser.add_argument("--curriculum-advance-threshold", type=float, default=0.05)
    train_parser.add_argument("--curriculum-min-hold-epochs", type=int, default=2)
    train_parser.add_argument("--curriculum-max-hold-epochs", type=int, default=25)
    # Curriculum frontier cap + one-flag fast protocol (use the SAME for every ablation/baseline).
    train_parser.add_argument("--max-frontier", type=int, default=0,
                              help="cap the curriculum at T_frontier<=this (0=full to 1000)")
    train_parser.add_argument("--fast-protocol", action="store_true",
                              help="fast uniform run for the ablation/baseline table: caps frontier "
                                   "at T100, loosens gating, trims epochs (~70). Apply identically to all models.")

    # Rollout-stability controls. Early-step anchoring w_k = 1 + a*exp(-(k-1)/tau) weights near-term steps
    # so the one-step map stays sharp while the curriculum supervises longer horizons.
    train_parser.add_argument("--rollout-anchor-weight", type=float, default=12.0)
    train_parser.add_argument("--rollout-anchor-tau", type=float, default=1.5)
    # Truncated BPTT window: gradient flows through this many steps before detaching, so the model is
    # penalized for compounding error and learns to self-correct.
    train_parser.add_argument("--rollout-tbptt-steps", type=int, default=4)
    # Gaussian drift noise on the fed-forward state (normalized units), hardening the map against its own
    # error distribution; never applied to the true t0 input.
    train_parser.add_argument("--rollout-state-noise", type=float, default=0.01)

    # Contractivity / Jacobian penalty: penalizes expansive directions of J = d x_next/d x_t so per-step
    # error cannot grow. The observable-space realization of the D_res dissipation. Set 0.0 to disable.
    train_parser.add_argument("--loss-jacobian-weight", type=float, default=0.5)
    train_parser.add_argument("--jacobian-target-gain", type=float, default=1.0)
    train_parser.add_argument("--jacobian-max-steps", type=int, default=6)
    train_parser.add_argument("--jacobian-probes", type=int, default=1)
    train_parser.add_argument("--jacobian-ramp-start", type=int, default=2)
    train_parser.add_argument("--jacobian-ramp-end", type=int, default=20)

    # ---- Metriplectic physics ramp windows (latent energy/dissipation) ----
    train_parser.add_argument("--loss-energy-ramp-start", type=int, default=3)
    train_parser.add_argument("--loss-energy-ramp-end", type=int, default=30)
    train_parser.add_argument("--loss-dissip-ramp-start", type=int, default=3)
    train_parser.add_argument("--loss-dissip-ramp-end", type=int, default=30)

    # Autoregressive vs direct prediction. --direct-prediction predicts each horizon directly from the
    # initial condition with absolute-time conditioning, avoiding error accumulation.
    train_parser.add_argument("--model", default="rmmd",
                              choices=["rmmd", "hybrid", "dgknet-hybrid", "fused", "mlp", "lstm", "node", "dgknet", "fno"],
                              help="model architecture: rmmd (full), hybrid (RMMD + gated MLP skip), "
                                   "dgknet-hybrid (RMMD on quiet + a genuine dgknet operator on q4, "
                                   "dead-zone per-shot gate), or a baseline (mlp/lstm/node/dgknet/fno). "
                                   "Baselines run the SAME harness/drivers/curriculum, data-loss only.")
    # dgknet-hybrid gate supervision (default 1.0 when --model dgknet-hybrid; 0 disables).
    train_parser.add_argument("--gate-sup-weight", type=float, default=0.0,
                              help="dgknet-hybrid: weight on supervising the per-shot gate to OPEN on "
                                   "dynamic (high per-step |dNI|) steps and stay closed on quiet")
    train_parser.add_argument("--skip-competence-weight", type=float, default=0.0,
                              help="dgknet-hybrid: weight on training the dgknet skip to predict next-NI "
                                   "EVERYWHERE (so it's good when the gate opens)")
    train_parser.add_argument("--gate-target-scale", type=float, default=0.1,
                              help="dgknet-hybrid: relative |dNI| at which the gate target ~=0.76 "
                                   "(smaller = gate opens on subtler dynamics)")
    train_parser.add_argument("--baseline-latent-dim", type=int, default=128,
                              help="latent width for baseline models (128 default ~1.5-4M params; "
                                   "node@1536 gives a CAPACITY-MATCHED baseline 26.9M ≈ full RMMD 26.2M)")
    train_parser.add_argument("--direct-prediction", action="store_true")
    train_parser.add_argument("--no-transport-step", dest="use_transport_step", action="store_false",
                              help="use free-residual decode instead of the conservative transport step C")
    train_parser.set_defaults(use_transport_step=True)
    # ABLATIONS (each removes ONE novel component -> a row in the ablation table). Train each
    # under the SAME protocol as full RMMD for a fair comparison.
    train_parser.add_argument("--ablate-drivers", action="store_true",
                              help="ablation: ignore time-resolved drivers (static context only)")
    train_parser.add_argument("--ablate-geometry", action="store_true",
                              help="ablation: remove flux-surface geometry information")
    train_parser.add_argument("--ablate-dres", action="store_true",
                              help="ablation: diagonal-only dissipation (no resonant off-diagonal D_res)")
    train_parser.add_argument("--ablate-transport", action="store_true",
                              help="ablation: free-residual decode (no conservative transport step)")
    train_parser.add_argument("--drivergate", action="store_true",
                              help="enable the learned DRIVER-GATE: relax the NI block's resonant "
                                   "contraction on driver-signalled transients (targets q4 dynamics) "
                                   "while staying contractive on quiet shots. Off => ungated RMMD.")
    # HYBRID-only (model=hybrid): the gated MLP skip that bypasses the operator bottleneck.
    train_parser.add_argument("--ablate-skip", action="store_true",
                              help="hybrid ablation: remove the MLP skip -> recovers the exact RMMD")
    train_parser.add_argument("--hybrid-skip-hidden", type=int, default=512,
                              help="hidden width of the hybrid MLP skip (model=hybrid)")

    # Off-diagonal dissipation guardrail: keeps cross-mode resonant coupling as a real share of the
    # dissipation operator. target_frac is a floor on ||D_res||_F / ||D_psd||_F.
    train_parser.add_argument("--loss-offdiag-dissip-weight", type=float, default=0.02)
    train_parser.add_argument("--offdiag-target-frac", type=float, default=0.3)

    # Tendency + conservation losses. Tendency = relative increment error
    # ||dNI_pred - dNI_true||^2 / ||dNI_true||^2, giving the per-step increment a gradient. Set 0.0 to disable.
    train_parser.add_argument("--loss-tendency-weight", type=float, default=0.05)
    train_parser.add_argument("--tendency-ramp-start", type=int, default=2)
    train_parser.add_argument("--tendency-ramp-end", type=int, default=12)
    # Conservation: relative drift of the volume-weighted particle integral sum_i NI_i V'_i.
    # 'radial' uses a rho-weight PROXY for V'(rho); true V' from geometry is a C deliverable.
    train_parser.add_argument("--loss-conservation-weight", type=float, default=0.05)
    train_parser.add_argument("--conservation-volume-mode", default="radial", choices=["radial", "uniform"])

    train_parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    train_parser.add_argument("--seed", type=int, default=None,
                              help="set torch/numpy/random seeds for reproducible replicates (default None = "
                                   "current nondeterministic behavior). Used by decisive EXP-1 seed replicates.")
    # compact_ni_geom is always True (legacy mode removed)
    train_parser.set_defaults(compact_ni_geom=True)

    # ------------------------------------------------------------------ eval
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument(
        "--checkpoint",
        default="/scratch/gpfs/USER/models/rmmd_final/checkpoint_best.pt",
    )
    eval_parser.add_argument(
        "--test-data",
        default="/scratch/gpfs/USER/strong_rmmd/data_build/dataset_test_compact.pt",
    )
    eval_parser.add_argument(
        "--compact-test-data",
        default="/scratch/gpfs/USER/strong_rmmd/data_build/dataset_test_compact.pt",
    )
    eval_parser.add_argument(
        "--output-dir",
        default="/scratch/gpfs/USER/models/rmmd_final/eval",
    )
    eval_parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    eval_parser.add_argument("--max-shots", type=int, default=1000)
    eval_parser.add_argument("--log-every", type=int, default=25)
    eval_parser.set_defaults(compact_ni_geom=True)

    # ------------------------------------------------------------------ test
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument(
        "--train-data",
        default="/scratch/gpfs/USER/strong_rmmd/data_build/dataset_train_compact.pt",
    )
    test_parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()
    if args.command == "train":
        return mod.train_command(args)
    if args.command == "eval":
        return mod.eval_command(args)
    if args.command == "test":
        return mod.test_command(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
