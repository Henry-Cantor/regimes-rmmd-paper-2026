# strong_rmmd

The core model package. PLEASE NOTE that many of the files included here may be slightly outdated, or correspond to failed experiments. Files that run the paper experiments themselves are found in other folders, with this one being primarily composed of helper files. Training and evaluation drivers are in `../training`; experiments are in
`../decisive_experiments`, `../theory_validation`, and `../comparison`.

| File | What it is |
|---|---|
| `multi_machine_rmmd.py` | `MultiMachineRMMD` — the full operator: per-machine embedding (with an unknown-machine token for zero-shot holdouts), geometry and driver encoders, resonance kernel, transport step, autoregressive rollout. |
| `rmmd_block.py` | `RMMDBlock` and the off-diagonal `D_res` term. |
| `resonance_kernel.py` | `LorentzianResonanceKernel`: `D_psd = Σ_k a_k(z)·L_k(ω_t, ω_d, γ_k)·v_k v_k^T`, with harmonic centers keyed to the physical drift frequency. |
| `resonance_frequencies.py` | Computes `ω_t`, `ω_d` from the state and geometry. |
| `transport.py` | The conservative transport step. |
| `geometry.py` | Geometry featurization. |
| `baselines.py` | Comparison baselines: MLP, LSTM, NODE, DGKNet, FNO. |
| `losses.py` | Loss terms (rollout, conservation, SUT regularizers). |
| `sut_analysis.py` | SUT diagnostics. |
| `theorems.py` | The theorems as code, where implemented. |
| `config.py`, `dataset.py`, `data_loader.py` | Config and dataset plumbing. |

Support modules: `memory_kernel.py`, `transfer.py`, `diagnostics.py`, `visualization.py`, `utils.py`.
Earlier hybrid/fused variants (`hybrid_rmmd.py`, `fused_rmmd.py`, `dgknet_hybrid_rmmd.py`) are retained for
the ablation record; the deployed hybrid is the per-shot router in `../decisive_experiments`.
