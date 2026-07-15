# RMMD

A physics-informed machine-learning surrogate for tokamak ion-density (NI) transport with zero-shot
transfer to held-out machines. The model is a physics-keyed metriplectic–Koopman latent operator
`L = S − D_diag − D_res` with a resonance-mediated off-diagonal term `D_res`, decoded through a
conservative, contraction-stable transport step so autoregressive rollouts do not drift. Novelty comes in the model structure via physics-keying and the use of computed frequencies, and several discoveries made in the experiment.

This repository reproduces every figure and table in the paper. The name "STRONG" in some of the folder names refers to an older acronym of the project (Spectral Thermodynamic and Resonance Operator for New Geometries).

## Layout

```
STRONG_RMMD/
  strong_rmmd/            Model package
    multi_machine_rmmd.py   MultiMachineRMMD: encoders, constrained operator, decoder, transport step
    rmmd_block.py           Metriplectic operator with resonance off-diagonal D_res
    transport.py            Conservative, contraction-stable transport step
    losses.py               Training losses
    resonance_frequencies.py  Gyro-Bohm resonance frequency features
  data_io/                Dataset loader
  data_build/             Build the compact NI + geometry dataset from TRANSP CDFs
  training/               Train and evaluate (rmmd_train_eval.py)
  comparison/             Comparison and ablation suite; results/ holds the committed figure inputs
  theory_validation/      SUT confirmation and zero-shot extrapolation; results/ holds committed JSONs
  hybrid_analysis/        RMMD/DGKNet hybrid threshold analysis
  notebooks_paper/        paper_figures.ipynb builds figures/ and tables/ from the committed JSONs
  tests/                  Unit tests

dgknet_baseline/          DGKNet baseline and the TRANSP CDF data pipeline
```

## Reproduce

The result JSONs under `comparison/results/` and `theory_validation/results/` are committed, so a fresh
clone can build every figure and table without a GPU (step 4). Steps 2–3 require a GPU.

```bash
# 0. Environment
pip install -r requirements.txt          # or: docker build -t strong-rmmd .
export PYTHONPATH="$PWD/STRONG_RMMD:$PWD"

# 1. (optional) Rebuild the dataset from TRANSP CDFs
python STRONG_RMMD/data_build/build_phase0new_from_cdfs.py --help

# 2. Train the model, ablations, and baselines under the fast protocol
bash STRONG_RMMD/comparison/run_all.sh <CKPT_ROOT> <indist_test_compact.pt>
#    single model:
python STRONG_RMMD/training/rmmd_train_eval.py train --model full \
       --epochs 35 --lr 2e-4 --latent-dim 192 --batch-size 32 --max-frontier 50

# 3. SUT confirmation and zero-shot extrapolation on both holdouts
bash STRONG_RMMD/theory_validation/run_validation.sh \
       <CKPT_ROOT> <indist_test.pt> <EAST_holdout.pt> <AUGD_holdout.pt>

# 4. Build every figure and table
jupyter nbconvert --to notebook --execute --inplace \
       STRONG_RMMD/notebooks_paper/paper_figures.ipynb
```

## Notes

- Cluster paths in script defaults are placeholders (`/scratch/gpfs/USER/...`); override them with the
  CLI arguments shown by each script's `--help`.
- Run scripts are resumable and set `CUDA_MODULE_LOADING=EAGER` for compatibility with sm_70 GPUs.
- `STRONG_RMMD/strong_rmmd` and `STRONG_RMMD` differ only in case; on case-insensitive filesystems, do
  not delete by the uppercase path.
