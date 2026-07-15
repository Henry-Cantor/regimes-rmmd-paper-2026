# comparison

Evaluates trained checkpoints (RMMD, the structural ablations, and the baselines) on the same
in-distribution test set, per horizon, and emits one comparison-table JSON. The in-distribution
counterpart to the zero-shot suite in `../theory_validation`.

| File | What it is |
|---|---|
| `run_comparison.py` | The runner: one comparison-table JSON over all checkpoints, per horizon. |
| `run_all.sh`, `run_headline.sh`, `run_baseline_lr_grid.sh`, `run_rmmd_sensitivity.sh` | Launch scripts for the full suite, the headline model, the per-baseline LR grid, and RMMD hyperparameter sweeps. |
| `select_lr_winners.py` | Selects each baseline's best learning rate, so comparisons are best-config vs. best-config. |
| `eval_sut_universality.py`, `eval_sut_zeroshot.py` | SUT evaluations. |
| `convergence_check.py`, `uncertainty_summary.py` | Convergence and uncertainty summaries. |
| `results/` | Emitted comparison-table JSONs. |

Every model is selected at its own best learning rate.
