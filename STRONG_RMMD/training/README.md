# training

Train and evaluate the model. Experiments and figures use the checkpoints produced here.

| File | What it is |
|---|---|
| `rmmd_train_eval.py` | CLI entry point: `train` / `eval`. Key flags: `--model {rmmd,mlp,lstm,node,dgknet,fno}`, `--epochs`, `--latent-dim`, `--lr`, `--batch-size`, `--max-frontier`, `--compact-train-data`, `--compact-val-data`, `--checkpoint-dir`. |
| `rmmd_train_eval_impl.py` | Implementation: `CompactRolloutDataset`, curriculum scheduler, fast protocol (35 epochs), rollout, per-step drivers. |
| `scripts/` | Launch helpers. |

The fast protocol (latent 192, 35 epochs) is the standard configuration; one uniform configuration is used across all experiments.
