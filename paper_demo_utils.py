"""Read paper snapshots and optionally refit their frozen design matrices.

Viewing results requires only NumPy and pandas. Fitting imports the repository's
utils lazily. No function writes into paper_results or the original notebooks.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def load_results(path, expected_sha256=None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Paper snapshot checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        saved = {k: (archive[k].item() if archive[k].ndim == 0 else archive[k].copy())
                 for k in archive.files}
    n = int(saved['N_sim'])
    for method in saved['simulation_fit_methods']:
        for suffix, shape in [('plug_in_estimates', (n,)), ('one_step_estimates', (n,)),
                              ('CIs_one_step', (n, 2)), ('one_step_SEs', (n,))]:
            if saved[f'{method}_{suffix}'].shape != shape:
                raise ValueError(f"Unexpected shape: {method}_{suffix}")
    return saved


def make_series(saved, selection, joint_ci='asymptotic'):
    """Match the original notebooks' finite-value masks and interval selection."""
    series = []
    for method, estimator in selection:
        if estimator not in {'plug_in', 'one_step'}:
            raise ValueError(f'Unknown estimator: {estimator}')
        ci_method = 'one_step' if estimator == 'one_step' else (
            joint_ci if method == 'joint' else 'bootstrap')
        prefix = f'{method}_'
        estimates = saved[prefix + ('one_step_estimates' if estimator == 'one_step'
                                    else 'plug_in_estimates')]
        cis = saved[prefix + ('CIs_one_step' if estimator == 'one_step'
                              else f'CIs_mle_{ci_method}')]
        ses = saved[prefix + ('one_step_SEs' if estimator == 'one_step'
                              else f'mle_{ci_method}_SEs')]
        valid = np.isfinite(estimates) & np.isfinite(ses) & np.all(np.isfinite(cis), axis=1)
        series.append(dict(method=method, estimator=estimator, ci_method=ci_method,
                           label=f"{method} {'MLE' if estimator == 'plug_in' else 'one-step'}",
                           estimates=estimates, cis=cis, ses=ses, valid=valid,
                           color='C3' if method == 'joint' and estimator == 'plug_in'
                           else ('C1' if estimator == 'plug_in' else 'C4')))
    return series


def summarize(saved, series):
    rows = []
    truth = saved['c_true']
    for s in series:
        values = s['estimates'][np.isfinite(s['estimates'])]
        cis = s['cis'][s['valid']]
        rows.append(dict(estimator=s['label'], interval=s['ci_method'],
                         valid_estimates=len(values), valid_intervals=len(cis), total=saved['N_sim'],
                         mean=values.mean() if len(values) else np.nan,
                         bias=values.mean()-truth if len(values) else np.nan,
                         std=values.std(ddof=1) if len(values)>1 else np.nan,
                         coverage=np.mean((cis[:, 0]<=truth)&(cis[:, 1]>=truth)) if len(cis) else np.nan,
                         mean_width=np.mean(cis[:, 1]-cis[:, 0]) if len(cis) else np.nan))
    return pd.DataFrame(rows).set_index('estimator')


def rerun(saved, n_sim=10, n_boot=0, seed=1, hessian_method='numerical'):
    """Refit the saved statistical model; bootstrap and paper-scale runs are opt-in.

    Uses the archived A, covariance, EFT kernels, offsets, and constraints directly.
    This reproduces the inference stage, not FK generation or LHAPDF integration.
    Current utils/optimizer versions and historical RNG differences can change results.
    """
    from utils import (fit_beta_c, mle_beta, mle_c, one_step_c,
                       one_step_confidence_interval, bootstrap_confidence_interval,
                       joint_mle_asymptotic_confidence_interval)
    for name, value, minimum in [('n_sim', n_sim, 1), ('n_boot', n_boot, 0)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if n_boot == 1:
        raise ValueError('Use n_boot=0 to disable bootstrap or at least 2 draws')
    if hessian_method not in {'analytical', 'numerical'}:
        raise ValueError('Unknown Hessian method')
    is_v = 'eft_offset' in saved
    A, Sigma, xQ2 = (saved[k] for k in ('design_matrix', 'Sigma', 'xQ2'))
    linear, quadratic = saved['klin_values'], saved['kquad_values']
    # All calls below use the archived kinematic row order.
    klin = lambda z: linear
    kquad = lambda z: quadratic
    v = lambda c, xQ2: 1 + linear*c + quadratic*c*c
    vp = lambda c, xQ2: linear + 2*quadratic*c
    vpp = lambda c, xQ2: 2*quadratic
    offset = saved.get('eft_offset')
    w = saved.get('w_norm') if saved.get('enforce_sum_rule', saved.get('mle_constrained', False)) else None
    os_w = saved.get('w_norm') if saved.get('enforce_sum_rule', saved.get('one_step_constrained', False)) else None
    nonneg = saved.get('nonnegative_pdf', saved.get('nonnegative_beta', True))
    pin = saved['pin_endpoints']
    c0, scale = saved['c0'], saved['loss_scale']
    # T3 currently splits the data and bootstrap streams; V uses the direct data stream.
    data_seed, boot_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(seed if is_v else data_seed)
    boot_rng = np.random.default_rng(boot_seed)
    result_suffixes = ('plug_in_estimates', 'one_step_estimates', 'CIs_one_step',
                       'one_step_SEs', 'CIs_mle_asymptotic', 'mle_asymptotic_SEs',
                       'CIs_mle_bootstrap', 'mle_bootstrap_SEs', 'all_boot_samples')
    methods = tuple(saved['simulation_fit_methods'])
    result_keys = {f'{m}_{s}' for m in methods for s in result_suffixes}
    output = {k: v for k, v in saved.items() if k not in result_keys}
    output.update(N_sim=n_sim, n_boot_sim=n_boot, simulation_seed=seed,
                  compute_bootstrap_coverage=bool(n_boot),
                  simulation_joint_mle_ci_method='both' if n_boot else 'asymptotic',
                  simulation_joint_mle_hessian_method=hessian_method,
                  result_origin='fresh rerun from frozen paper design', numpy_version=np.__version__)
    for method in methods:
        for suffix in result_suffixes:
            shape = (n_sim, n_boot) if suffix == 'all_boot_samples' else (
                (n_sim, 2) if suffix.startswith('CIs_') else (n_sim,))
            output[f'{method}_{suffix}'] = np.full(shape, np.nan)
    failures = []
    for i in range(n_sim):
        draw_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        D = np.random.default_rng(draw_seed).multivariate_normal(saved['generating_mean'], Sigma)
        for method in methods:
            prefix = method + '_'
            beta_opt = saved['joint_optimizer'] if method == 'joint' else saved['separate_beta_optimizer']
            try:
                if method == 'joint':
                    fit = fit_beta_c(D=D, A=A, v=v, v_prime=vp, c0=c0,
                                     w_norm=w, xQ2=xQ2, Sigma=Sigma, eft_offset=offset,
                                     nonneg_beta=nonneg, pin_endpoints=pin,
                                     opt_method=beta_opt, loss_scale=scale)
                    if not fit.success:
                        failures.append(f'{method} replicate {i}: optimizer reported {fit.message}')
                    beta, c = fit.beta_hat, fit.c_hat
                else:
                    beta = mle_beta(D=D, A=A, c=c0, v=v, xQ2=xQ2, Sigma=Sigma,
                                    w_norm=w, eft_offset=offset, nonneg_beta=nonneg,
                                    pin_endpoints=pin, beta_opt=beta_opt, loss_scale=scale)
                    c = mle_c(D=D, A=A, v=v, beta=beta, xQ2=xQ2, Sigma=Sigma,
                              eft_offset=offset, c_opt=saved['separate_c_optimizer'],
                              c_range=tuple(saved['c_range']))
                output[prefix+'plug_in_estimates'][i] = c
                K, Kp = v(c, xQ2)[:, None]*A, vp(c, xQ2)[:, None]*A
                os, info = one_step_c(D, K, Kp, beta, c, w_norm=os_w,
                                     Sigma=None if is_v else Sigma,
                                     s2_diag=np.diag(Sigma) if is_v else None, pin_endpoints=pin,
                                     fixed_mean=v(c, xQ2)*offset if is_v else None,
                                     fixed_mean_prime=vp(c, xQ2)*offset if is_v else None)
                lo, hi, se = one_step_confidence_interval(info)
                output[prefix+'one_step_estimates'][i] = os
                output[prefix+'CIs_one_step'][i] = lo, hi
                output[prefix+'one_step_SEs'][i] = se
                if method == 'joint':
                    lo, hi, se, _ = joint_mle_asymptotic_confidence_interval(
                        A=A, xQ2=xQ2, Sigma=Sigma, c_hat=c, beta_hat=beta,
                        v=v, v_prime=vp, D=D, v_second=vpp, w_norm=w,
                        eft_offset=offset, pin_endpoints=pin, hessian_method=hessian_method)
                    output[prefix+'CIs_mle_asymptotic'][i] = lo, hi
                    output[prefix+'mle_asymptotic_SEs'][i] = se
                if n_boot:
                    lo, hi, draws = bootstrap_confidence_interval(
                        A=A, xQ2=xQ2, Sigma=Sigma, w_norm=w, eft_offset=offset,
                        c0=c0, c_hat=c, beta_hat=beta, n_boot=n_boot, estimator='mle',
                        klin=klin, kquad=kquad, nonneg_beta=nonneg, pin_endpoints=pin,
                        opt=method, c_opt=saved['separate_c_optimizer'],
                        c_range=tuple(saved['c_range']), beta_opt=beta_opt, loss_scale=scale,
                        seed=int(boot_rng.integers(0, np.iinfo(np.uint32).max)), verbose=False)
                    output[prefix+'CIs_mle_bootstrap'][i] = lo, hi
                    output[prefix+'mle_bootstrap_SEs'][i] = np.nanstd(draws, ddof=1)
                    output[prefix+'all_boot_samples'][i] = draws
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                failures.append(f'{method} replicate {i}: {exc}')
        if (i+1) % 20 == 0:
            print(f'{i+1}/{n_sim} replicates complete')
    output['rerun_diagnostics'] = np.asarray(failures, dtype=str)
    for message in failures:
        print(message)
    return output


def save_rerun(results, name):
    """Create a unique rerun file, never overwrite a paper snapshot or prior run."""
    from datetime import datetime, timezone
    if name not in {'T3', 'V'}:
        raise ValueError('name must be T3 or V')
    folder = Path('simulation_results/paper_reruns')
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = folder / f'{name}_{stamp}.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, **results)
    return path
