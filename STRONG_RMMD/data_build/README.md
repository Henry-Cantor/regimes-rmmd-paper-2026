# data_build

Builds the compact datasets that every experiment uses from raw TRANSP CDFs.

| File | What it is |
|---|---|
| `build_phase0new_from_cdfs.py` | The builder. Reads raw TRANSP CDFs, extracts pre-shot context, builds NI, geometry, and drivers, and writes `dataset_{train,val,test}_compact.pt`. Normalizes per machine (each sample uses its own machine's stats) and stores `normalization_stats_by_machine`. |
| `build_limiters.py`, `find_limiter_reference.py` | Limiter-geometry reference construction. |

The pool is the five training machines; EAST and AUGD are the zero-shot holdouts. Held-out evaluation data
is already per-machine normalized — attach the machine's stats rather than re-normalizing. Driver set: PINJ,
PCUR, gas.
