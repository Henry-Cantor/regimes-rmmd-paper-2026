# hybrid_analysis

This entire folder lays the groundwork for the later Ladder/Router test, expanded to persistence/RMMD/NODE. It tests whether a per-shot switch between RMMD and the DGKNet baseline (DGKNet on dynamic shots, RMMD on quiet ones) beats either model alone. The deployed result is the driver-keyed router in
`../decisive_experiments` (`exp4b_router_sota.py`, `buttress.py`); this folder is the threshold analysis
that preceded it.

| File | What it does |
|---|---|
| `collect_per_shot.py` | Collect per-shot RMMD vs. DGKNet errors across val/test/holdout shots. |
| `find_threshold.py` | Fit an activity threshold on val+test, evaluate on the held-out machines. |
| `diagnose_complementarity.py` | Test whether the RMMD/DGKNet complementarity has spatial/temporal structure. |

`find_threshold.py` reports both an oracle switch (on true per-shot activity, the ceiling) and a
deployable switch (on inference-available features). The gap between them is the cost of predicting
activity from inputs.

## Run

```bash
python STRONG_RMMD/hybrid_analysis/collect_per_shot.py \
    --ckpt-root <CKPT> --horizon 50 --device cuda \
    --dataset val:<val.pt> --dataset test:<test.pt> \
    --dataset east:<EAST.pt> --dataset augd:<AUGD.pt> \
    --out STRONG_RMMD/hybrid_analysis/results/per_shot.json

python STRONG_RMMD/hybrid_analysis/find_threshold.py \
    STRONG_RMMD/hybrid_analysis/results/per_shot.json
```
