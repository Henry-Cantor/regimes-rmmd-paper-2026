# decisive_experiments

Pre-registered, control-based experiments that test the paper's claims. Each writes a result JSON to
`results/`, which the paper notebook reads.

| Script | Question | Result JSON |
|---|---|---|
| `exp1_universality_emergent.py` | Is operator universality emergent or an artifact of parameter sharing? | `universality_emergent.json` |
| `exp2_dres_regime.py` | Does `D_res` help the regime GIT predicts? | `dres_regime.json` |
| `exp3_git_synthetic.py` | Is the GIT KL law exact on systems with known coupling? | `git_synthetic.json` |
| `exp4_predictability_wall.py` | Is the per-shot winner predictable from inputs? | `predictability_wall.json` |
| `exp4b_router_sota.py` | Driver-keyed router vs. the baselines | `router_rmmd_dgknet.json` |
| `exp5_lomo_analysis.py` | Leave-one-machine-out transfer | `universality_predictive.json` |
| `buttress.py` | Crossover mechanism, router, nonlocality diagnostics | `buttress.json` |
| `fno_extension.py` | 1-D Fourier Neural Operator baseline | `fno.json` |

## Leave-one-machine-out (exp5)

Hold out one training machine; the model receives the unknown-machine embedding and adapts only through
geometry, resonance frequencies, and drivers. Trained on the four remaining pool machines under the fast
protocol.

## buttress

One per-shot pass that computes: (A) `||D_res||^2` vs. `omega_d` per machine; (B) the driver-keyed router
and its per-cell comparison against the baselines; (C) partial correlation of `||D_res||` with a
profile-nonlocality proxy, controlling for drive.

Run each script with `--help` for arguments and data paths.
