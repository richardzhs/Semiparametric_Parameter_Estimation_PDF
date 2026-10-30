# Reproducing the paper results

- [BCDMS demo: T3 closure test](BCDMS_demo.ipynb)
- [HERA demo: V closure test](HERA_demo.ipynb)

Each notebook displays the saved simulation results used in the paper. Run all cells to regenerate the summary tables and plots from the included archives. New fits and simulations are optional and disabled by default.

## Environment and execution

Use Python 3.11 and install the tested dependencies:

```sh
python -m pip install -r requirements-paper-demos.txt
```

Open either notebook in a Jupyter-compatible editor, select that Python environment, and run all cells from the repository directory. If using JupyterLab, install it separately. Keep `paper_demo_utils.py` and `paper_results/` alongside the notebooks. LHAPDF is not required to read the results or refit the saved design matrices.

## Reading the results

The notebooks show experiment settings, estimator bias and spread, finite-result counts, 95% interval coverage and widths, sampling distributions, and coverage calibration. The default comparison includes separate MLE, its one-step correction, and joint MLE with asymptotic intervals. Set `joint_ci="bootstrap"` to display the archived joint-MLE bootstrap intervals instead.

Coverage uses saved interval endpoints. Calibration curves use normal intervals reconstructed from saved standard errors; they do not recompute percentile-bootstrap intervals at each confidence level. Counts in the summary table identify the replicates included in each calculation.

## Saved experiments

Both archives contain 1,000 simulations and 500 bootstrap samples per fit. `paper_results/manifest.json` records the archive paths, SHA-256 checksums, and source-notebook hashes. The included snapshots are byte-for-byte copies of the research archives. The research-archive paths below record provenance and are not required files on this branch:

| Experiment | Research archive | Included snapshot |
|---|---|---|
| BCDMS | `simulation_results/BCDMS/T3_simulation_results_c=500_Lambda=500.npz` | `paper_results/T3_simulation_results_c=500_Lambda=500.npz` |
| HERA | `simulation_results/HERA/V_simulation_results_c=2_Lambda=500.npz` | `paper_results/V_simulation_results_c=2_Lambda=500.npz` |

The HERA archive does not record the Hessian method used to compute its joint-MLE intervals. Those saved intervals are displayed directly; new runs explicitly use numerical Hessians.

## Rerunning the inference

In Section 6, set `run_reproduction=True`. The helper reads the design matrix, generating mean, covariance, EFT coefficients, fixed contributions, constraints, and optimizer choices from the archive, then uses `utils.py` to refit simulated datasets. BCDMS uses full covariance in the one-step calculation; HERA uses its diagonal.

A small rerun defaults to 10 replicates without bootstrap. To use the paper's sample counts, set:

```python
rerun_count = int(paper["N_sim"])
rerun_bootstrap_count = int(paper["n_boot_sim"])
```

Nested bootstrap runs can take a long time. New results remain separate from the archived `paper` results. Set `save_new_results=True` to save a uniquely named NPZ under `simulation_results/paper_reruns/`. Fit failures and optimizer warnings are recorded in `rerun_diagnostics`.

This workflow reproduces the inference stage from the saved model. The research notebooks and upstream data-generation work are maintained on the [real-model-experiment branch](https://github.com/richardzhs/Semiparametric_Parameter_Estimation_PDF/tree/real-model-experiment). Library versions, optimizer initialization, and historical random-number handling can prevent exact numerical reproduction. In particular, the historical HERA bootstrap calls did not set a seed; new bootstrap draws are seeded. Use the checked archives to recover the exact saved results.

## Required files

Keep the two demo notebooks, `paper_demo_utils.py`, `requirements-paper-demos.txt`, and the contents of `paper_results/` together. Optional refitting also requires the repository's `utils.py`. The notebooks retain their tables and plots for reading without execution.
