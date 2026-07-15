# notebooks_paper

Renders the paper's figures and tables from the committed result JSONs. No compute here.

| Item | What it is |
|---|---|
| `paper_figures.ipynb` | The figure notebook. Each cell reads a result JSON and writes to `figures/` or `tables/`. |
| `figures/` | Figure outputs (PDF and PNG). |
| `tables/` | LaTeX table outputs. |

Rebuild everything with:

```bash
jupyter nbconvert --to notebook --execute --inplace STRONG_RMMD/notebooks_paper/paper_figures.ipynb
```
