# data_io

I/O layer between the compact `.pt` datasets (built in `../data_build`) and the training and evaluation
code.

| File | What it is |
|---|---|
| `dataset_loader.py` | Loads the compact datasets and applies the per-machine normalization stats attached at build time. |
| `scripts/` | Data-staging and conversion helpers. |

The datasets are already per-machine normalized at build time; attach a machine's stats rather than
re-normalizing.
