# Semiparametric Estimation of Physics Parameters and PDFs

[![arXiv](https://img.shields.io/badge/arXiv-2611.04267-b31b1b.svg)](https://arxiv.org/abs/2611.04267)

This repository contains the code for the paper [*Semiparametric Estimation of Physics Parameters and Parton Distribution Functions*](https://arxiv.org/abs/2611.04267).

The results can be reproduced with two kinds of notebook: demos, which show the published results, and experiments, which let you change the settings and rerun the simulation.

| | Notebooks | Use these to |
|---|---|---|
| **Demos** | [BCDMS demo](BCDMS_demo.ipynb), [HERA demo](HERA_demo.ipynb) | Reproduce the paper's tables and plots from the saved experiments. No extra setup and nothing to configure. Model settings cannot be changed here. |
| **Experiments** | [BCDMS experiment](BCDMS_experiment.ipynb), [HERA experiment](HERA_experiment.ipynb) | Change any setting (basis, design matrix, fitting method, simulation settings) and rerun the simulation end to end. Use these whenever you want different settings from the paper. |

Start with a demo if you want the published numbers. Each demo notebook displays the saved simulation results used in the paper: run all cells to regenerate the summary tables and plots from the included saved experiments. New fits and simulations are optional and disabled by default.

## Installation

Use Python 3.11 and install the dependencies:

```sh
python -m pip install -r requirements-paper-demos.txt
```

Rerunning the experiment notebooks additionally requires LHAPDF with the `NNPDF40_nnlo_as_01180` set.

### Required files

Keep the two demo notebooks, `paper_demo_utils.py`, `requirements-paper-demos.txt`, and the contents of `paper_results/` together. Optional refitting also requires the repository's `utils.py`. The experiment notebooks additionally need `Dataset/data_208.npz` and `Dataset/data_HERA_F3_aZ_PID200.npz`.

## Saved experiments

Both saved experiments contain 1,000 simulations and 500 bootstrap samples per fit.

| Experiment | Path                                                       |
|---|------------------------------------------------------------|
| BCDMS | `paper_results/T3_simulation_results_c=500_Lambda=500.npz` |
| HERA | `paper_results/V_simulation_results_c=2_Lambda=500.npz`    |

The numbers in each file name are the true EFT coefficient used to generate the pseudo-data (`c_true`) and the new-physics scale (`Lambda`). Each file holds three kinds of content:

- **Settings:** the dataset and PDF set used, `c_true`, `Lambda`, the number of simulations and bootstrap samples, the random seed, the spline basis (degree and number of basis functions), the optimizers, and the fit constraints. The HERA file also records its sum rule and the fixed NNPDF background.
- **Model:** the data points in $(x, Q^2)$, the covariance matrix, the design matrix, the mean used to generate the pseudo-data, and the linear and quadratic EFT terms. 
- **Results:** for each of the 1,000 simulated datasets, and for both the separate and the joint fit, the MLE and one-step estimates of `c`, their standard errors, the asymptotic, one-step and bootstrap 95% intervals, and all 500 bootstrap estimates.

The demo notebooks only use the results: their tables and plots are made from the stored estimates and intervals, and no fitting is done. If you start a new run (see below), the settings and model are used to generate fresh pseudo-data and fit it again.

## Reading the results

The demo notebooks show experiment settings, estimator bias and spread, finite-result counts, 95% interval coverage and widths, sampling distributions, and coverage calibration. The default comparison includes separate MLE, its one-step correction, and joint MLE with asymptotic intervals. Set `joint_ci="bootstrap"` to display the joint-MLE bootstrap intervals from the saved experiments instead.

Coverage uses saved interval endpoints. Calibration curves use normal intervals reconstructed from saved standard errors; they do not recompute percentile-bootstrap intervals at each confidence level. Counts in the summary table identify the replicates included in each calculation.

## Rerunning the inference

First, in section 6 of either demo notebook ("Optional reproduction from the saved model"), set `run_reproduction=True`. 

The helper reads the design matrix, generating mean, covariance, EFT coefficients, fixed contributions, constraints, and optimizer choices from the saved experiment, then uses `utils.py` to refit simulated datasets. 

A small rerun defaults to 10 replicates without bootstrap. To use the paper's sample counts, set:

```python
rerun_count = int(paper["N_sim"])
rerun_bootstrap_count = int(paper["n_boot_sim"])
```

Nested bootstrap runs can take a long time. New results remain separate from the saved experiment loaded as `paper`. Set `save_new_results=True` to save a uniquely named NPZ under `simulation_results/paper_reruns/`. Fit failures and optimizer warnings are recorded in `rerun_diagnostics`.

## Experiment notebooks

[BCDMS experiment](BCDMS_experiment.ipynb) and [HERA experiment](HERA_experiment.ipynb) are the notebooks behind the two demos, covering basis construction, design-matrix assembly, fitting, and the simulation studies. 

They read the saved experiments in `paper_results/` together with the design-matrix inputs in `Dataset/`, and they retain their tables and plots for reading without execution. When their simulation cells are enabled, new output is written under `simulation_results/` and the saved experiments are left untouched.
