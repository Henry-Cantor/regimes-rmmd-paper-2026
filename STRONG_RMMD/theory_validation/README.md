# theory_validation

SUT confirmation, zero-shot extrapolation, and theorem validation. These scripts import the training and
comparison implementations, so evaluation semantics match the comparison suite.

| File | Produces |
|---|---|
| `sut_confirmation.py` | Cross-machine operator spectra, between/within-machine universality statistics with permutation p-values, EAST zero-shot containment → `results/sut_report.json` |
| `extrap_strong.py` | In-distribution vs. zero-shot NRMSE for all models with CIs and paired Wilcoxon tests, extrapolation-gap and ablation tables, per-machine breakdown → `results/extrap_strong_report.json` |
| `theorems_validation.py` | Growth-law model selection, GIT correlations and KL-linearity, full-vs-diagonal divergence, RODEA fits → `results/theorems_report.json` |

## Run

```bash
export CUDA_MODULE_LOADING=EAGER
CKPT=<checkpoint root>
IND=<indist_test_compact.pt>
EAST=<EAST_test_compact.pt>

python STRONG_RMMD/theory_validation/extrap_strong.py \
    --indist-data $IND --east-data $EAST --ckpt-root $CKPT --reference full --device cuda

python STRONG_RMMD/theory_validation/sut_confirmation.py \
    --checkpoint $CKPT/full --indist-data $IND --east-data $EAST --device cuda

python STRONG_RMMD/theory_validation/theorems_validation.py \
    --checkpoint $CKPT/full --abl-dres-checkpoint $CKPT/abl_dres --test-data $IND --device cuda
```

Every input except one checkpoint and the in-distribution dataset is optional; missing pieces are recorded
as null with a reason in the JSON. The PINJ driver channel defaults to index 0; override with
`--pinj-channel`.
