"""
utils.py — Utility functions for the demos.

Sections
--------
1. Basis Functions
2. Wilson Coefficients & EFT
3. Design Matrices & Theory Predictions
4. Likelihood & Fitting
5. Plotting
6. Experiment
"""

from functools import partial

import numpy as np
from scipy.interpolate import BSpline
from scipy.integrate import quad, fixed_quad
from scipy.differentiate import hessian as scipy_hessian
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# 1. Basis Functions
# =============================================================================

def bspline_basis(degree, M, a, b):
    """
    Build a B-spline basis on the interval [a, b].

    Parameters
    ----------
    degree : int
        Degree of the B-splines.
    M : int
        Number of basis functions.
    a, b : float
        Interval endpoints.

    Returns
    -------
    B : list of M BSpline objects, each callable as B_k(x).
    """
    n_internal = M - degree - 1
    if n_internal > 0:
        inner_knots = np.linspace(a, b, n_internal + 2)[1:-1]
    else:
        inner_knots = np.array([])
    knots = np.r_[np.repeat(a, degree + 1), inner_knots, np.repeat(b, degree + 1)]

    I = np.eye(M)
    B = [BSpline(knots, I[m], degree, extrapolate=False) for m in range(M)]
    return B


# =============================================================================
# 2. Wilson Coefficients & EFT
# =============================================================================

def _as_pairs(xQ2):
    """
    Normalize input into an (n, 2) float array of (x, Q2) pairs.

    Accepts:
      - array-like of shape (n, 2)
      - tuple/list (x_array, Q2_array) with the same length
    """
    if isinstance(xQ2, (tuple, list)) and len(xQ2) == 2 and np.ndim(xQ2[0]) > 0:
        x = np.asarray(xQ2[0], dtype=float)
        Q2 = np.asarray(xQ2[1], dtype=float)
        if x.shape != Q2.shape:
            raise ValueError(f"x and Q2 must have same shape, got {x.shape} vs {Q2.shape}")
        return np.column_stack([x.ravel(), Q2.ravel()]).astype(float)

    arr = np.asarray(xQ2, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"xQ2 must have shape (n,2), got {arr.shape}")
    return arr


def _validate_x(x, idx=None):
    if x <= 0.0:
        where = f" at index {idx}" if idx is not None else ""
        raise ValueError(f"x must be in (0,1], got x={x}{where}.")

def klin(z, Lambda=100):
    """Linear Wilson coefficient: Q^2 / Lambda^2."""
    z = np.asarray(z, dtype=float)
    return z / (Lambda ** 2)


def kquad(z, Lambda=100):
    """Quadratic Wilson coefficient: Q^4 / Lambda^4."""
    z = np.asarray(z, dtype=float)
    return (z ** 2) / (Lambda ** 4)


def v_c(c, xQ2, klin=klin, kquad=kquad):
    """
    Polynomial factor v(c; Q^2) = 1 + klin(Q^2) * c + kquad(Q^2) * c^2.

    Parameters
    ----------
    c : float
        Wilson coefficient.
    xQ2 : (n, 2) array
        Array of (x, Q^2) pairs.

    Returns
    -------
    v : (n,) array
    """
    xQ2 = _as_pairs(xQ2)
    Q2 = xQ2[:, 1]
    return 1.0 + klin(Q2) * c + kquad(Q2) * (c ** 2)


def v_prime_c(c, xQ2, klin=klin, kquad=kquad):
    """
    Derivative of v(c; Q^2) with respect to c.

    Returns
    -------
    dv/dc : (n,) array
    """
    xQ2 = _as_pairs(xQ2)
    Q2 = xQ2[:, 1]
    return klin(Q2) + 2.0 * c * kquad(Q2)


def v_second_c(c, xQ2, kquad=kquad):
    """Second derivative of ``v(c; Q^2)`` with respect to ``c``."""
    xQ2 = _as_pairs(xQ2)
    Q2 = xQ2[:, 1]
    return 2.0 * kquad(Q2)


# =============================================================================
# 3. Design Matrices & Theory Predictions
# =============================================================================

def build_f_design_matrix_FK(B, W, grid):
    """
    Build the design matrix A for parametrization f(y) = sum_k beta_k B_k(y),
    using FK tables.

    Parameters
    ----------
    B : list of M basis functions, each callable B_k(y).
    W : (N_data, N_grid) FK table matrix.
    grid : (N_grid,) array of y values corresponding to columns of W.

    Returns
    -------
    A : (N_data, M) array where A[r,k] = (W @ B_k(grid)).
    """
    B_grid = np.vstack([Bk(grid) for Bk in B])   # (M, N_grid)
    return W @ B_grid.T                            # (N_data, M)


def build_yf_design_matrix_FK(B, W, grid):
    """
    Build the design matrix A for parametrization g(y) = y*f(y) = sum_k theta_k B_k(y),
    using FK tables.  Here f(y) = g(y)/y, so the FK convolution gains a 1/y factor.

    Parameters
    ----------
    B : list of M basis functions, each callable B_k(y).
    W : (N_data, N_grid) FK table matrix.
    grid : (N_grid,) array of y values.

    Returns
    -------
    A : (N_data, M) array where A[r,k] = (W @ (B_k(grid) / grid)).
    """
    B_grid = np.vstack([Bk(grid) for Bk in B])           # (M, N_grid)
    return W @ (B_grid.T / grid[:, np.newaxis])            # (N_data, M)


def build_K_matrix(c, A, xQ2, klin=klin, kquad=kquad):
    """K(c) = diag(v_c(c, xQ2)) @ A  —  so that mu = K @ beta."""
    v = v_c(c, xQ2, klin=klin, kquad=kquad)
    return np.diag(v) @ A


def build_K_prime_matrix(c, A, xQ2, klin=klin, kquad=kquad):
    """dK/dc = diag(v_prime_c(c, xQ2)) @ A."""
    v_prime = v_prime_c(c, xQ2, klin=klin, kquad=kquad)
    return np.diag(v_prime) @ A


def T_sm_FK(f, W, grid):
    """
    Standard-model theory predictions using FK table convolution.

    Parameters
    ----------
    f : callable
        PDF function f(y).
    W : (N_data, N_grid) FK table matrix.
    grid : (N_grid,) array of y values.

    Returns
    -------
    T_vec : (N_data,) array of SM predictions.
    """
    f_grid = f(grid)       # (N_grid,)
    return W @ f_grid      # (N_data,)


def T_sm_basis(A, beta):
    """SM predictions from basis expansion: T = A @ beta."""
    return A @ beta


def true_mean(c, T, xQ2, klin=klin, kquad=kquad):
    """
    Full mean mu = v_c(c, xQ2) * T, where T is the SM prediction vector.
    """
    v = v_c(c, xQ2, klin=klin, kquad=kquad)
    return v * T


def true_mean_basis(c, beta, A, xQ2, klin=klin, kquad=kquad,
                    eft_offset=None):
    """
    Full mean mu = v_c(c, xQ2) * (A @ beta + eft_offset).

    ``eft_offset`` is a fixed contribution inside the EFT factor.  
    Unlike an ordinary additive background, it is multiplied by the EFT factor.
    """
    v = v_c(c, xQ2, klin=klin, kquad=kquad)
    T = T_sm_basis(A, beta)
    if eft_offset is not None:
        eft_offset = np.asarray(eft_offset, dtype=float).reshape(-1)
        if eft_offset.shape != T.shape:
            raise ValueError(
                f"eft_offset must have shape {T.shape}, got {eft_offset.shape}."
            )
        T = T + eft_offset
    return v * T


# =============================================================================
# 4. Likelihood & Fitting
# =============================================================================

def make_nll_and_grad(D, A, v, v_prime, xQ2, Sigma=None,
                      eft_offset=None):
    """
    Build the negative log-likelihood and its gradient as callables.

    Parameters
    ----------
    D : (N,) data vector.
    A : (N, M) design matrix.
    v : callable, v(c, xQ2) -> (N,) vector.
    v_prime : callable, v_prime(c, xQ2) -> (N,) derivative vector.
    xQ2 : (N, 2) array of (x, Q^2) pairs.
    Sigma : (N, N) covariance matrix, or None for identity.

    Returns
    -------
    nll : callable(theta) -> float,  theta = [beta (M,), c (1,)]
    grad : callable(theta) -> (M+1,) gradient array
    """
    N, M = A.shape
    if eft_offset is None:
        eft_offset = np.zeros(N)
    else:
        eft_offset = np.asarray(eft_offset, dtype=float).reshape(-1)
        if eft_offset.shape != (N,):
            raise ValueError(
                f"eft_offset must have shape {(N,)}, got {eft_offset.shape}."
            )

    if Sigma is not None:
        cF, lower = cho_factor(Sigma, check_finite=False)
        def Sigma_inv(v):
            return cho_solve((cF, lower), v, check_finite=False)
    else:
        def Sigma_inv(v):
            return v

    def nll(theta):
        beta = theta[:M]
        c = theta[-1]
        vc = v(c, xQ2=xQ2)
        target = A @ beta + eft_offset
        mu = vc * target
        r = D - mu
        return 0.5 * r @ Sigma_inv(r)

    def grad(theta):
        beta = theta[:M]
        c = theta[-1]
        vc = v(c, xQ2=xQ2)
        dv = v_prime(c, xQ2=xQ2)
        target = A @ beta + eft_offset
        mu = vc * target
        r = D - mu
        Sigma_inv_r = Sigma_inv(r)
        g_beta = -((vc[:, None] * A).T @ Sigma_inv_r)
        g_c = -(dv * target).T @ Sigma_inv_r
        return np.r_[g_beta, g_c]

    return nll, grad


class _BetaLayout:
    """Manages the free/pinned beta index layout and normalization helpers."""

    def __init__(self, M, w_norm, pin_endpoints):
        if w_norm is not None and w_norm.shape[0] != M:
            raise ValueError(f"w_norm must have length M={M}, got {w_norm.shape[0]}")
        if pin_endpoints:
            if M < 3:
                raise ValueError("Need at least M >= 3 when pin_endpoints=True.")
            self.s = slice(1, M - 1)
        else:
            self.s = slice(0, M)
        self.M = M
        self.M_free = self.s.stop - self.s.start
        self.w_free = w_norm[self.s] if w_norm is not None else None
        if self.M_free < 1:
            raise ValueError("No free beta coefficients to optimize.")

    def build_full(self, beta_free):
        beta_full = np.zeros(self.M)
        beta_full[self.s] = beta_free
        return beta_full

    def extract_free(self, beta_full):
        return np.asarray(beta_full)[self.s].copy()

    def normalize_free(self, beta_free):
        denom = float(self.w_free @ beta_free)
        if abs(denom) < 1e-15:
            raise ValueError("Cannot normalize beta: w_free @ beta_free is zero or too close to zero.")
        return beta_free / denom

    def random_free(self, nonneg_beta):
        b = np.random.rand(self.M_free)
        if nonneg_beta:
            b = np.abs(b)
        return self.normalize_free(b) if self.w_free is not None else b

    def init_free(self, beta0, nonneg_beta, pin_endpoints, verbose=False):
        if beta0 is None:
            return self.random_free(nonneg_beta)
        beta0 = np.asarray(beta0, dtype=float).copy()
        if beta0.shape[0] != self.M:
            raise ValueError(f"beta0 must have length M={self.M}, got {beta0.shape[0]}")
        if pin_endpoints:
            beta0[0] = 0.0
            beta0[-1] = 0.0
        beta0_free = self.extract_free(beta0)
        if self.w_free is not None and abs(self.w_free @ beta0_free - 1.0) > 1e-10:
            if verbose:
                print("Initial beta0 does not satisfy the normalization constraint on free coefficients:",
                      "w_free @ beta0_free =", self.w_free @ beta0_free)
            beta0_free = self.normalize_free(beta0_free)
        if verbose and nonneg_beta and np.any(beta0_free < 0):
            print("Initial beta0 has negative free entries.")
        return beta0_free


def _run_with_restarts(minimize_fn, theta0, gen_theta, num_try, history=None):
    """Run minimize_fn(theta0); on failure retry with gen_theta() up to num_try times."""
    res = minimize_fn(theta0)
    while not res.success and num_try > 0:
        if history is not None:
            history.clear()
        res = minimize_fn(gen_theta())
        num_try -= 1
    return res


def fit_beta_c(D, A, v, v_prime, w_norm=None, xQ2=None, Sigma=None, eft_offset=None,
               beta0=None, c0=0.0, nonneg_beta=False, pin_endpoints=False, 
               return_history=False, opt_method="SLSQP", loss_scale=1, num_try=10):
    """
    Joint MLE of (beta, c) via constrained minimization of the negative log-likelihood.

    Parameters
    ----------
    D : (N,) data vector.
    A : (N, M) design matrix.
    v : callable, v(c, xQ2) -> (N,) vector.
    v_prime : callable, v_prime(c, xQ2) -> (N,) derivative vector.
    w_norm : (M,) normalization weights enforcing w_norm @ beta = 1, or None for
        unconstrained optimization.
    xQ2 : (N, 2) array of (x, Q^2) pairs.
    Sigma : (N, N) covariance, or None for identity.
    eft_offset : (N,) fixed contribution inside the EFT factor, or None.
    beta0 : (M,) initial beta, or None for random initialization.
    c0 : float, initial c.
    nonneg_beta : bool, enforce beta >= 0.
    pin_endpoints : bool, fix beta[0] = beta[-1] = 0.
    return_history : bool, if True return (res, history).
    opt_method : "SLSQP" or "trust-constr".
    loss_scale : float, scale factor for the loss (useful for numerical stability).
    num_try : int, number of random initializations to try for the optimization.

    Returns
    -------
    res : OptimizeResult with attributes beta_hat and c_hat.
    history : list of per-iteration dicts (only if return_history=True).
    """
    N, M = A.shape
    layout = _BetaLayout(M, w_norm, pin_endpoints)
    M_free, w_free = layout.M_free, layout.w_free

    beta0_free = layout.init_free(beta0, nonneg_beta, pin_endpoints)
    theta0 = np.r_[beta0_free, c0]

    nll, grad = make_nll_and_grad(
        D, A, v, v_prime, xQ2=xQ2, Sigma=Sigma,
        eft_offset=eft_offset,
    )

    def unpack_theta(theta):
        beta_free = theta[:M_free]
        c = theta[-1]
        beta_full = layout.build_full(beta_free)
        return beta_full, c

    def nll_scaled(theta):
        beta_full, c = unpack_theta(theta)
        return nll(np.r_[beta_full, c]) / loss_scale

    def grad_scaled(theta):
        beta_full, c = unpack_theta(theta)
        g_full = grad(np.r_[beta_full, c]) / loss_scale
        return np.r_[g_full[:M][layout.s], g_full[-1]]

    if w_free is not None:
        cons = [{
            "type": "eq",
            "fun": lambda x: float(w_free @ x[:M_free] - 1.0),
            "jac": lambda x: np.r_[w_free, 0.0]
        }]
    else:
        cons = []

    history = [] if return_history else None

    def callback(theta, state=None):
        if not return_history:
            return
        beta_full, c = unpack_theta(theta)
        constr_viol = (float(state.constr_violation) if state is not None
                       else (float(abs(w_free @ theta[:M_free] - 1.0)) if w_free is not None else 0.0))
        history.append({
            "loss": float(state.fun) if state is not None else float(nll_scaled(theta)),
            "constr_viol": constr_viol,
            "grad_norm": (float(state.optimality)
                          if state is not None else float(np.linalg.norm(grad_scaled(theta)))),
            "beta_norm": float(np.linalg.norm(beta_full)),
            "c": float(c),
        })

    if nonneg_beta:
        bounds = [(0.0, None)] * M_free + [(None, None)]
    else:
        bounds = [(None, None)] * M_free + [(None, None)]

    if opt_method == "SLSQP":
        options = dict(maxiter=5000, ftol=1e-10)
    elif opt_method == "trust-constr":
        options = dict(maxiter=5000, gtol=1e-10)
    else:
        raise ValueError(f"Unsupported optimization method: {opt_method}")

    def do_minimize(t0):
        return minimize(nll_scaled, t0, jac=grad_scaled, method=opt_method,
                        constraints=cons, bounds=bounds, options=options,
                        callback=callback if return_history else None)

    res = _run_with_restarts(
        do_minimize, theta0,
        gen_theta=lambda: np.r_[layout.random_free(nonneg_beta), c0],
        num_try=num_try, history=history,
    )

    if not res.success:
        print(f"Optimization failed: {res.message}")

    res.beta_hat = layout.build_full(res.x[:M_free])
    res.c_hat = res.x[-1]

    if return_history:
        return res, history
    return res


def mle_beta(D, A, c, v, xQ2, w_norm=None, Sigma=None, eft_offset=None, 
             beta0=None, nonneg_beta=True, pin_endpoints=False, beta_opt="SLSQP", 
             return_history=False, loss_scale=1, num_try=10):
    """
    MLE of beta given a fixed c.

    Parameters
    ----------
    D : (N,) data vector.
    A : (N, M) design matrix.
    c : float, fixed Wilson coefficient.
    v : callable, v(c, xQ2) -> (N,) vector.
    xQ2 : (N, 2) array of (x, Q^2) pairs.
    w_norm : (M,) normalization weights enforcing w_norm @ beta = 1, or None for
        unconstrained optimization.
    Sigma : (N, N) covariance, or None for identity.
    eft_offset : (N,) fixed contribution inside the EFT factor, or None.
    beta0 : (M,) initial guess, or None for random initialization.
    nonneg_beta : bool, enforce beta >= 0.
    pin_endpoints : bool, fix beta[0] = beta[-1] = 0.
    beta_opt : "SLSQP", "trust-constr", or "closed_form".
    return_history : bool, if True return (beta_hat, history).
    loss_scale : float, scale factor for the loss.
    num_try : int, number of random initializations to try for the optimization.

    Returns
    -------
    beta_hat : (M,) array.
    history : list of per-iteration dicts (only if return_history=True).
    """
    N, M = A.shape
    if eft_offset is None:
        eft_offset = np.zeros(N)
    else:
        eft_offset = np.asarray(eft_offset, dtype=float).reshape(-1)
        if eft_offset.shape != (N,):
            raise ValueError(
                f"eft_offset must have shape {(N,)}, got {eft_offset.shape}."
            )
    layout = _BetaLayout(M, w_norm, pin_endpoints)
    M_free, w_free = layout.M_free, layout.w_free

    if Sigma is not None:
        cF, lower = cho_factor(Sigma, check_finite=False)
        Sigma_inv = lambda z: cho_solve((cF, lower), z, check_finite=False)
    else:
        Sigma_inv = lambda z: z

    vc = v(c, xQ2=xQ2)
    vcA = vc[:, None] * A  # precomputed; fixed since c is fixed
    beta0_free = layout.init_free(beta0, nonneg_beta, pin_endpoints, verbose=True)

    if beta_opt in ("SLSQP", "trust-constr"):
        def loss_and_grad(beta_free):
            r = D - vc * (
                A @ layout.build_full(beta_free) + eft_offset
            )
            Sinv_r = Sigma_inv(r)
            f = 0.5 * r @ Sinv_r
            g = -(vcA.T @ Sinv_r)[layout.s]
            return f / loss_scale, g / loss_scale

        if w_free is not None:
            cons = {
                'type': 'eq',
                'fun': lambda beta_free: float(w_free @ beta_free - 1.0),
                'jac': lambda beta_free: w_free
            }
        else:
            cons = []

        if nonneg_beta:
            bounds = [(0.0, None)] * M_free
        else:
            bounds = [(None, None)] * M_free

        history = [] if return_history else None

        def callback(beta_free, state=None):
            if not return_history:
                return
            constr_viol = (float(state.constr_violation) if state is not None
                           else (float(abs(w_free @ beta_free - 1.0)) if w_free is not None else 0.0))
            if state is not None:
                loss_val, grad_norm = float(state.fun), float(state.optimality)
            else:
                f, g = loss_and_grad(beta_free)
                loss_val, grad_norm = float(f), float(np.linalg.norm(g))
            history.append({
                'loss': loss_val,
                'constr_viol': constr_viol,
                'grad_norm': grad_norm,
                'beta_norm': float(np.linalg.norm(layout.build_full(beta_free))),
                'beta_free_norm': float(np.linalg.norm(beta_free)),
            })

        if beta_opt == "SLSQP":
            options = dict(maxiter=5000, ftol=1e-10)
        elif beta_opt == "trust-constr":
            options = dict(maxiter=5000, gtol=1e-10)

        def do_minimize(t0):
            return minimize(loss_and_grad, t0, jac=True,
                            method=beta_opt, constraints=cons, bounds=bounds,
                            callback=callback if return_history else None, options=options)

        res = _run_with_restarts(
            do_minimize, beta0_free,
            gen_theta=lambda: layout.random_free(nonneg_beta),
            num_try=num_try, history=history,
        )

        if not res.success:
            print(f"Optimization for beta failed: {res.message}")

        beta_hat = layout.build_full(res.x)
        if return_history:
            return beta_hat, history
        return beta_hat

    elif beta_opt == "closed_form":
        # Solve in the same free-coordinate layout as the numerical
        # optimizers so pin_endpoints=True is respected.
        Wmat_free = vcA[:, layout.s]                                  # (N, M_free)
        SinvW = np.column_stack(
            [Sigma_inv(Wmat_free[:, j]) for j in range(M_free)]
        )
        U = Wmat_free.T @ SinvW                                      # (M_free, M_free)
        D_adjusted = D - vc * eft_offset
        b = Wmat_free.T @ Sigma_inv(D_adjusted)                      # (M_free,)

        if w_free is not None:
            KKT = np.zeros((M_free + 1, M_free + 1), dtype=float)
            KKT[:M_free, :M_free] = U
            KKT[:M_free, M_free] = w_free
            KKT[M_free, :M_free] = w_free
            rhs = np.zeros(M_free + 1, dtype=float)
            rhs[:M_free] = b
            rhs[M_free] = 1.0
            sol = np.linalg.solve(KKT, rhs)
            beta_free = sol[:M_free]
        else:
            beta_free = np.linalg.solve(U, b)

        beta_hat = layout.build_full(beta_free)

        if nonneg_beta and np.any(beta_hat < 0):
            raise ValueError("Closed-form beta has negative entries; try 'SLSQP' or 'trust-constr'.")
        return beta_hat
    else:
        raise ValueError("Invalid beta_opt method.")



def mle_c(D, A, v, beta, xQ2, Sigma=None, eft_offset=None, 
          c_opt="scipy_optimize", c_range=(-5.0, 5.0)):
    """
    MLE of c given a fixed beta.

    Parameters
    ----------
    D : (N,) data vector.
    A : (N, M) design matrix.
    v : callable, v(c, xQ2) -> (N,) vector.
    beta : (M,) fixed coefficient vector.
    xQ2 : (N, 2) array of (x, Q^2) pairs.
    Sigma : (N, N) covariance, or None for identity.
    eft_offset : (N,) fixed contribution inside the EFT factor, or None.
    c_opt : "scipy_optimize" or "grid_search".
    c_range: tuple (c_min, c_max) for grid search, default (-5.0, 5.0).

    Returns
    -------
    c_hat : float
    """

    N = A.shape[0]
    if eft_offset is None:
        eft_offset = np.zeros(N)
    else:
        eft_offset = np.asarray(eft_offset, dtype=float).reshape(-1)
        if eft_offset.shape != (N,):
            raise ValueError(
                f"eft_offset must have shape {(N,)}, got {eft_offset.shape}."
            )

    if Sigma is not None:
        cF, lower = cho_factor(Sigma, check_finite=False)
        Sigma_inv = lambda vec: cho_solve((cF, lower), vec, check_finite=False)
    else:
        Sigma_inv = lambda vec: vec

    def nll(c):
        vc = v(c, xQ2=xQ2)
        mu = vc * (A @ beta + eft_offset)
        r = D - mu
        return 0.5 * r @ Sigma_inv(r)

    if c_opt == "L-BFGS-B":
        res = minimize(lambda c: nll(c), x0=0.0, method='L-BFGS-B')
    elif c_opt == "Nelder-Mead":
        res = minimize(lambda c: nll(c), x0=0.0, method='Nelder-Mead')
    elif c_opt == "grid_search":
        c_min, c_max = c_range
        c_grid = np.linspace(c_min, c_max, 1000)
        nll_values = np.array([nll(c) for c in c_grid])
        return float(c_grid[np.argmin(nll_values)])
    else:
        raise ValueError("Invalid c_opt method.")
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return float(res.x[0])


def one_step_c(D, K, K_prime, beta_hat, c_hat, w_norm=None, Sigma=None,
               s2_diag=None, pin_endpoints=False, fixed_mean=None,
               fixed_mean_prime=None):
    """
    One-step estimator for c using the efficient influence function (EIF).

    The nuisance parameter beta is constrained to the tangent space
    {h : w_norm @ h = 0}, and the EIF is projected accordingly. If
    w_norm is None, no constraint is enforced and h is projected onto
    the unconstrained tangent space (plain least-squares projection).

    Parameters
    ----------
    D : (N,) data vector.
    K : (N, M) matrix such that mu = K @ beta.
    K_prime : (N, M) derivative dmu/dc = K_prime @ beta.
    beta_hat : (M,) plug-in estimate of beta.
    c_hat : float, plug-in estimate of c.
    w_norm : (M,) normalization weights, or None to skip the constraint.
    Sigma : (N, N) full covariance, or None.
    s2_diag : (N,) diagonal variance vector (preferred over full Sigma when diagonal).
    pin_endpoints : bool
        If True, beta[0] and beta[-1] are fixed and excluded from the nuisance
        tangent space. The fitted mean and its c derivative still use the full
        coefficient vector.
    fixed_mean : (N,) fixed target contribution after multiplication by v(c).
    fixed_mean_prime : (N,) c derivative of ``fixed_mean``. Both default to
        zero. These terms affect the mean and c score but not the beta nuisance
        tangent space.

    Returns
    -------
    c_one_step : float
    info : dict with diagnostics (eif, num, den, lambda, w_dot_beta, w_dot_h).
    """
    D = np.asarray(D).reshape(-1)
    beta_hat = np.asarray(beta_hat).reshape(-1)
    if w_norm is not None:
        w_norm = np.asarray(w_norm).reshape(-1)
    K = np.asarray(K)
    K_prime = np.asarray(K_prime)

    N, M = K.shape

    if fixed_mean is None:
        fixed_mean = np.zeros(N)
    else:
        fixed_mean = np.asarray(fixed_mean, dtype=float).reshape(-1)
        if fixed_mean.shape != (N,):
            raise ValueError(
                f"fixed_mean must have shape {(N,)}, got {fixed_mean.shape}."
            )
    if fixed_mean_prime is None:
        fixed_mean_prime = np.zeros(N)
    else:
        fixed_mean_prime = np.asarray(fixed_mean_prime, dtype=float).reshape(-1)
        if fixed_mean_prime.shape != (N,):
            raise ValueError(
                "fixed_mean_prime must have shape "
                f"{(N,)}, got {fixed_mean_prime.shape}."
            )

    if D.shape[0] != N:
        raise ValueError(f"D has length {D.shape[0]} but K has {N} rows.")
    if K_prime.shape != (N, M):
        raise ValueError(f"K_prime must have shape {(N, M)}, got {K_prime.shape}.")
    if beta_hat.shape[0] != M:
        raise ValueError(f"beta_hat has length {beta_hat.shape[0]} but K has {M} cols.")
    if w_norm is not None and w_norm.shape[0] != M:
        raise ValueError(f"w_norm has length {w_norm.shape[0]} but K has {M} cols.")

    layout = _BetaLayout(M, w_norm, pin_endpoints)
    if pin_endpoints and not np.allclose(
        beta_hat[[0, -1]], 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "pin_endpoints=True, but beta_hat has nonzero endpoint coefficients."
        )

    K_free = K[:, layout.s]
    w_free = layout.w_free
    M_free = layout.M_free

    if s2_diag is not None:
        s2_diag = np.asarray(s2_diag).reshape(-1)
        if s2_diag.shape[0] != N:
            raise ValueError(f"s2_diag has length {s2_diag.shape[0]} but N={N}.")
        if np.any(s2_diag <= 0):
            raise ValueError("All entries of s2_diag must be positive.")
        def Sigma_inv(v):
            v = np.asarray(v)
            return v / s2_diag[:, None] if v.ndim == 2 else v / s2_diag
    else:
        if Sigma is None:
            def Sigma_inv(v):
                return np.asarray(v)
        else:
            Sigma = np.asarray(Sigma)
            if Sigma.shape != (N, N):
                raise ValueError(f"Sigma must have shape {(N, N)}, got {Sigma.shape}.")
            cF, lower = cho_factor(Sigma, check_finite=False)
            def Sigma_inv(v):
                return cho_solve((cF, lower), v, check_finite=False)

    mu_hat = K @ beta_hat + fixed_mean
    r_hat = D - mu_hat
    mu_prime_hat = K_prime @ beta_hat + fixed_mean_prime

    SinvK_free = Sigma_inv(K_free)                       # (N, M_free)
    U = 2 * K_free.T @ SinvK_free                        # (M_free, M_free)
    b = 2 * K_free.T @ Sigma_inv(mu_prime_hat)           # (M_free,)

    if w_free is None:
        # Unconstrained projection: minimize ||mu' - K h||^2_{Sigma^{-1}}
        try:
            h_free = np.linalg.solve(U, b)
        except np.linalg.LinAlgError:
            h_free = np.linalg.lstsq(U, b, rcond=None)[0]
        lam = None
    else:
        # Constrained projection over the free coordinates:
        # minimize ||mu' - K_free h_free||^2_{Sigma^{-1}}
        # subject to w_free^T h_free = 0.
        KKT = np.block([
            [U,                    w_free[:, None]],
            [w_free[None, :],      np.zeros((1, 1))]
        ])
        rhs = np.concatenate([b, np.array([0.0])])

        try:
            sol = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

        h_free = sol[:M_free]
        lam = float(sol[M_free])

    h_star = layout.build_full(h_free)
    delta_hat = mu_prime_hat - K @ h_star

    num = float(delta_hat.T @ Sigma_inv(r_hat))
    den = float(delta_hat.T @ Sigma_inv(delta_hat))

    if not np.isfinite(den) or den <= 0:
        print(f"Invalid EIF denominator (information): den={den}. "
              "This suggests non-identifiability or numerical issues.")

    eif = num / den
    c_one_step = float(c_hat + eif)

    info = {
        "c_hat": c_hat,
        "num": num,
        "den": den,
        "eif": eif,
        "lambda": lam,
        "h_star": h_star,
        "free_beta_indices": np.arange(M)[layout.s],
        "pin_endpoints": bool(pin_endpoints),
        "w_dot_beta": float(w_norm @ beta_hat) if w_norm is not None else None,
        "w_dot_h": float(w_norm @ h_star) if w_norm is not None else None,
    }

    return c_one_step, info

def one_step_confidence_interval(c_one_step_info, alpha=0.05):
    """
    Confidence interval for c based on the one-step estimator and its EIF.

    Parameters
    ----------
    c_one_step_info : dict, info returned by one_step_estimator.
    alpha : float, significance level (default 0.05 for 95% CI).

    Returns
    -------
    ci_lower : float, lower bound of the confidence interval.
    ci_upper : float, upper bound of the confidence interval.
    """

    c_one_step = c_one_step_info["c_hat"] + c_one_step_info["eif"]
    se = np.sqrt(1 / c_one_step_info["den"])
    z = norm.ppf(1 - alpha / 2)
    ci_lower = c_one_step - z * se
    ci_upper = c_one_step + z * se
    return ci_lower, ci_upper, se


def joint_mle_numerical_hessian(
    A, xQ2, Sigma, c_hat, beta_hat, v, D,
    w_norm=None, eft_offset=None, pin_endpoints=False,
    initial_step=1e-2, order=8, maxiter=10,
):
    """Numerically evaluate the reduced joint NLL Hessian at the MLE.

    The parameter order is ``theta = (c, gamma)``, where ``gamma`` spans the
    allowed tangent directions of the PDF coefficients. Differentiation is
    performed in scaled local coordinates and transformed back to the
    original ``(c, gamma)`` coordinates before returning the Hessian.

    Returns
    -------
    numerical_hessian, info
        The Hessian and finite-difference diagnostics, including the tangent
        basis, parameter scales, estimated errors, and convergence statuses.
    """
    A = np.asarray(A, dtype=float)
    xQ2 = np.asarray(xQ2, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float).reshape(-1)
    D = np.asarray(D, dtype=float).reshape(-1)

    if A.ndim != 2:
        raise ValueError(f"A must be a matrix, got shape {A.shape}.")
    N, M = A.shape
    if xQ2.shape != (N, 2):
        raise ValueError(f"xQ2 must have shape {(N, 2)}, got {xQ2.shape}.")
    if Sigma.shape != (N, N):
        raise ValueError(f"Sigma must have shape {(N, N)}, got {Sigma.shape}.")
    if beta_hat.shape != (M,):
        raise ValueError(f"beta_hat must have shape {(M,)}, got {beta_hat.shape}.")
    if D.shape != (N,):
        raise ValueError(f"D must have shape {(N,)}, got {D.shape}.")

    if w_norm is not None:
        w_norm = np.asarray(w_norm, dtype=float).reshape(-1)
        if w_norm.shape != (M,):
            raise ValueError(f"w_norm must have shape {(M,)}, got {w_norm.shape}.")

    if eft_offset is None:
        eft_offset = np.zeros(N)
    else:
        eft_offset = np.asarray(eft_offset, dtype=float).reshape(-1)
        if eft_offset.shape != (N,):
            raise ValueError(
                f"eft_offset must have shape {(N,)}, got {eft_offset.shape}."
            )

    layout = _BetaLayout(M, w_norm, pin_endpoints)
    if pin_endpoints and not np.allclose(
        beta_hat[[0, -1]], 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "pin_endpoints=True, but beta_hat has nonzero endpoint coefficients."
        )

    if layout.w_free is None:
        tangent_basis = np.eye(layout.M_free)
    else:
        _, _, vh = np.linalg.svd(
            layout.w_free[None, :], full_matrices=True
        )
        tangent_basis = vh[1:].T

    free_beta_indices = np.arange(M)[layout.s]
    beta_free = beta_hat[layout.s]
    n_gamma = tangent_basis.shape[1]
    gamma_scale = max(
        1.0,
        float(np.linalg.norm(beta_free)) / np.sqrt(max(layout.M_free, 1)),
    )
    parameter_scales = np.r_[
        max(1.0, abs(float(c_hat))),
        np.full(n_gamma, gamma_scale),
    ]

    cF, lower = cho_factor(Sigma, check_finite=False)

    def nll_at_scaled_local_point(z):
        local_theta = parameter_scales * np.asarray(z, dtype=float)
        c = float(c_hat) + local_theta[0]
        gamma = local_theta[1:]
        beta = beta_hat.copy()
        beta[layout.s] += tangent_basis @ gamma
        target = A @ beta + eft_offset
        mu = np.asarray(v(c, xQ2=xQ2), dtype=float).reshape(-1) * target
        residual = D - mu
        Sigma_inv_residual = cho_solve(
            (cF, lower), residual, check_finite=False
        )
        return float(0.5 * residual @ Sigma_inv_residual)

    def vectorized_nll(z):
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            return nll_at_scaled_local_point(z)
        return np.apply_along_axis(nll_at_scaled_local_point, 0, z)

    theta0_scaled = np.zeros(1 + n_gamma)
    numerical_result = scipy_hessian(
        vectorized_nll, theta0_scaled,
        initial_step=initial_step, order=order, maxiter=maxiter,
    )
    hessian_scaled = np.asarray(numerical_result.ddf, dtype=float)
    numerical_hessian = (
        hessian_scaled
        / parameter_scales[:, None]
        / parameter_scales[None, :]
    )
    numerical_hessian = 0.5 * (
        numerical_hessian + numerical_hessian.T
    )
    numerical_error = (
        np.asarray(numerical_result.error, dtype=float)
        / parameter_scales[:, None]
        / parameter_scales[None, :]
    )
    info = {
        "parameter_order": ("c", "f_tangent"),
        "parameter_scales": parameter_scales,
        "tangent_basis": tangent_basis,
        "free_beta_indices": free_beta_indices,
        "success": np.asarray(numerical_result.success).copy(),
        "status": np.asarray(numerical_result.status).copy(),
        "error": numerical_error,
        "nfev": np.asarray(numerical_result.nfev).copy(),
        "all_elements_converged": bool(np.all(numerical_result.success)),
        "condition_number": float(np.linalg.cond(numerical_hessian)),
        "eigenvalues": np.linalg.eigvalsh(numerical_hessian),
    }
    return numerical_hessian, info


def joint_mle_analytical_hessian(
    A, xQ2, Sigma, c_hat, beta_hat, v, v_prime, D, v_second,
    w_norm=None, eft_offset=None, pin_endpoints=False,
):
    """Analytically evaluate the reduced joint observed NLL Hessian.

    The parameter order is ``theta = (c, gamma)``, where ``gamma`` represents
    the allowed PDF-coefficient directions. Pinned endpoints are excluded, and
    a normalization constraint is incorporated through its tangent space.

    Returns
    -------
    analytical_hessian, info
        The exact observed Hessian and its component blocks and diagnostics.
    """
    A = np.asarray(A, dtype=float)
    xQ2 = np.asarray(xQ2, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float).reshape(-1)

    if A.ndim != 2:
        raise ValueError(f"A must be a matrix, got shape {A.shape}.")
    N, M = A.shape
    if xQ2.shape != (N, 2):
        raise ValueError(f"xQ2 must have shape {(N, 2)}, got {xQ2.shape}.")
    if Sigma.shape != (N, N):
        raise ValueError(f"Sigma must have shape {(N, N)}, got {Sigma.shape}.")
    if beta_hat.shape != (M,):
        raise ValueError(f"beta_hat must have shape {(M,)}, got {beta_hat.shape}.")

    if w_norm is not None:
        w_norm = np.asarray(w_norm, dtype=float).reshape(-1)
        if w_norm.shape != (M,):
            raise ValueError(f"w_norm must have shape {(M,)}, got {w_norm.shape}.")

    if eft_offset is None:
        eft_offset = np.zeros(N)
    else:
        eft_offset = np.asarray(eft_offset, dtype=float).reshape(-1)
        if eft_offset.shape != (N,):
            raise ValueError(
                f"eft_offset must have shape {(N,)}, got {eft_offset.shape}."
            )

    layout = _BetaLayout(M, w_norm, pin_endpoints)
    if pin_endpoints and not np.allclose(
        beta_hat[[0, -1]], 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "pin_endpoints=True, but beta_hat has nonzero endpoint coefficients."
        )

    vc = np.asarray(v(c_hat, xQ2=xQ2), dtype=float).reshape(-1)
    dv = np.asarray(v_prime(c_hat, xQ2=xQ2), dtype=float).reshape(-1)
    if vc.shape != (N,) or dv.shape != (N,):
        raise ValueError("v and v_prime must each return a vector of length N.")

    target = A @ beta_hat + eft_offset
    K_free = (vc[:, None] * A)[:, layout.s]
    mu_prime = dv * target

    # Orthonormal basis for the allowed beta perturbations.
    if layout.w_free is None:
        tangent_basis = np.eye(layout.M_free)
    else:
        _, _, vh = np.linalg.svd(
            layout.w_free[None, :], full_matrices=True
        )
        tangent_basis = vh[1:].T
    K_tangent = K_free @ tangent_basis

    # Full Jacobian for theta = (c, gamma), where allowed beta perturbations
    # are parameterized as h = tangent_basis @ gamma. The first column is the
    # derivative with respect to c, matching [[H_cc, H_cf], [H_fc, H_ff]].
    joint_jacobian = np.column_stack([mu_prime, K_tangent])
    cF, lower = cho_factor(Sigma, check_finite=False)
    Sigma_inv_jacobian = cho_solve(
        (cF, lower), joint_jacobian, check_finite=False
    )
    expected_fisher_info = joint_jacobian.T @ Sigma_inv_jacobian
    D = np.asarray(D, dtype=float).reshape(-1)
    if D.shape != (N,):
        raise ValueError(f"D must have shape {(N,)}, got {D.shape}.")

    d2v = np.asarray(
        v_second(c_hat, xQ2=xQ2), dtype=float
    ).reshape(-1)
    if d2v.shape != (N,):
        raise ValueError("v_second must return a vector of length N.")

    mu_hat = vc * target
    residual = D - mu_hat
    Sigma_inv_residual = cho_solve(
        (cF, lower), residual, check_finite=False
    )

    # Second derivatives of mu with respect to theta=(c, gamma).
    A_free = A[:, layout.s]
    A_tangent = A_free @ tangent_basis
    mu_gamma_c = dv[:, None] * A_tangent
    mu_cc = d2v * target

    # mu_gamma_gamma=0
    H_ff = expected_fisher_info[1:, 1:]
    H_fc = expected_fisher_info[1:, 0] - mu_gamma_c.T @ Sigma_inv_residual
    H_cf = H_fc[None, :]
    H_cc = expected_fisher_info[0, 0] - mu_cc @ Sigma_inv_residual
    observed_hessian = np.block([
        [np.array([[H_cc]]), H_cf],
        [H_fc[:, None], H_ff],
    ])
    info = {
        "parameter_order": ("c", "f_tangent"),
        "H_cc": H_cc,
        "H_cf": H_cf,
        "H_fc": H_fc[:, None],
        "H_ff": H_ff,
        "residual": residual,
        "joint_jacobian": joint_jacobian,
        "tangent_basis": tangent_basis,
        "free_beta_indices": np.arange(M)[layout.s],
        "pin_endpoints": bool(pin_endpoints),
        "w_dot_beta": float(w_norm @ beta_hat) if w_norm is not None else None,
        "condition_number": float(np.linalg.cond(observed_hessian)),
        "eigenvalues": np.linalg.eigvalsh(observed_hessian),
    }
    return observed_hessian, info


def joint_mle_asymptotic_confidence_interval(
    A, xQ2, Sigma, c_hat, beta_hat, v, v_prime=None, D=None, v_second=None,
    w_norm=None, eft_offset=None, pin_endpoints=False, alpha=0.05,
    hessian_method="analytical", numerical_hessian_options=None,
):
    """Asymptotic joint-MLE confidence interval using a selected Hessian.

    Parameters
    ----------
    hessian_method : {"analytical", "numerical"}
        How to compute the full observed Hessian of the negative
        log-likelihood. Both methods use the same reduced parameter order
        ``theta = (c, gamma)``. ``v_prime`` and ``v_second`` are only required
        by the analytical method.
    numerical_hessian_options : dict or None
        Optional keyword arguments forwarded to
        ``joint_mle_numerical_hessian`` when ``hessian_method="numerical"``.

    Returns
    -------
    ci_lower, ci_upper, se, info
        The variance of ``c_hat`` is element ``[0, 0]`` of the inverse of the
        selected observed Hessian.
    """
    method_aliases = {
        "analytic": "analytical",
        "analytical": "analytical",
        "numeric": "numerical",
        "numerical": "numerical",
    }
    if not isinstance(hessian_method, str):
        raise TypeError("hessian_method must be 'analytical' or 'numerical'.")
    method = method_aliases.get(hessian_method.lower())
    if method is None:
        raise ValueError("hessian_method must be 'analytical' or 'numerical'.")
    if D is None:
        raise ValueError("D is required to compute the observed Hessian.")

    if method == "analytical":
        if v_prime is None or v_second is None:
            raise ValueError(
                "v_prime and v_second are required for the analytical Hessian."
            )
        observed_hessian, hessian_info = joint_mle_analytical_hessian(
            A=A, xQ2=xQ2, Sigma=Sigma, c_hat=c_hat,
            beta_hat=beta_hat, v=v, v_prime=v_prime, D=D,
            v_second=v_second, w_norm=w_norm, eft_offset=eft_offset,
            pin_endpoints=pin_endpoints,
        )
    else:
        if numerical_hessian_options is None:
            numerical_hessian_options = {}
        observed_hessian, hessian_info = joint_mle_numerical_hessian(
            A=A, xQ2=xQ2, Sigma=Sigma, c_hat=c_hat,
            beta_hat=beta_hat, v=v, D=D,
            w_norm=w_norm, eft_offset=eft_offset,
            pin_endpoints=pin_endpoints,
            **numerical_hessian_options,
        )
    observed_hessian_condition_number = float(
        np.linalg.cond(observed_hessian)
    )
    try:
        observed_hessian_inverse = np.linalg.inv(observed_hessian)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f"The {method} joint observed Hessian is singular. The joint "
            "model may not identify c separately from the PDF."
        ) from error

    variance_c = float(observed_hessian_inverse[0, 0])
    if not np.isfinite(variance_c) or variance_c <= 0:
        raise ValueError(
            "The top-left element of the inverse observed Hessian is not "
            f"positive and finite: {variance_c}."
        )

    se = float(np.sqrt(variance_c))
    z = norm.ppf(1.0 - alpha / 2.0)
    ci_lower = float(c_hat - z * se)
    ci_upper = float(c_hat + z * se)
    info = dict(hessian_info)
    info.update({
        "c_hat": float(c_hat),
        "se": se,
        "variance_c": variance_c,
        "hessian_method": method,
        "observed_hessian": observed_hessian,
        "observed_hessian_inverse": observed_hessian_inverse,
        "observed_hessian_condition_number": observed_hessian_condition_number,
        "hessian_computation_info": hessian_info,
    })
    info[f"{method}_hessian"] = observed_hessian
    info[f"{method}_hessian_info"] = hessian_info
    return ci_lower, ci_upper, se, info

def bootstrap_confidence_interval(
    A, xQ2, Sigma,
    c_hat, beta_hat,
    w_norm=None,
    eft_offset=None,
    c0=0, n_boot=500,
    alpha=0.05,
    ci_method="normal",
    estimator="one_step",
    opt="joint",
    c_opt="scipy_optimize",
    c_range=(-5.0, 5.0),
    beta_opt="SLSQP",
    nonneg_beta=True,
    pin_endpoints=False,
    loss_scale=1,
    klin=klin, kquad=kquad,
    seed=None,
    verbose=True,
):
    """
    Parametric bootstrap CI for c (plug-in MLE or one-step estimator).

    Resamples D* ~ N(mu_hat, Sigma), where
    mu_hat = v_c(c_hat) * (A @ beta_hat + eft_offset), refits the model on
    each replicate, and returns a confidence interval.

    Parameters
    ----------
    A : (N, M) design matrix.
    xQ2 : (N, 2) array of (x, Q^2) pairs.
    Sigma : (N, N) covariance matrix of the data.
    c_hat : float, plug-in MLE of c from the original fit.
    beta_hat : (M,) plug-in MLE of beta from the original fit.
    w_norm : (M,) normalization weights for the sum-rule constraint, or None
        to skip the constraint (default None).
    eft_offset : (N,) fixed contribution inside the EFT factor, or None.
    c0 : float, initial guess for c in each bootstrap fit (default 0).
    n_boot : int, number of bootstrap replicates (default 500).
    alpha : float, significance level (default 0.05 for 95% CI).
    estimator : {"one_step", "mle"}
        Which estimator to collect across replicates.
    opt : {"joint", "separate"}, fitting strategy (must match the original fit).
    c_opt : str, optimizer for the c step (when opt="separate").
    c_range : tuple (c_min, c_max)
        Search range passed to ``mle_c`` when ``opt="separate"`` and
        ``c_opt="grid_search"``.
    beta_opt : str, optimizer for the beta step.
    nonneg_beta : bool, enforce non-negativity of beta.
    pin_endpoints : bool, fix endpoint basis coefficients.
    loss_scale : float, scale factor for the loss.
    ci_method : {"normal", "percentile"}
        "normal"     — use bootstrap SE with a normal z-interval:
                       c_hat ± z * std(boot_samples).
        "percentile" — use the [alpha/2, 1-alpha/2] quantiles directly.
    klin, kquad : callables, Wilson coefficient functions.
    seed : int or None, random seed for reproducibility.
    verbose : bool, print progress every 50 replicates.

    Returns
    -------
    ci_lower : float
    ci_upper : float
    boot_samples : (n_boot,) array of per-replicate estimates (NaN for failures).
    """
    _klin, _kquad = klin, kquad
    _v_c       = lambda c, xQ2: v_c(c, xQ2, klin=_klin, kquad=_kquad)
    _v_prime_c = lambda c, xQ2: v_prime_c(c, xQ2, klin=_klin, kquad=_kquad)

    if eft_offset is None:
        eft_offset = np.zeros(A.shape[0])
    else:
        eft_offset = np.asarray(eft_offset, dtype=float).reshape(-1)
        if eft_offset.shape != (A.shape[0],):
            raise ValueError(
                "eft_offset must have shape "
                f"{(A.shape[0],)}, got {eft_offset.shape}."
            )

    mu_hat = true_mean_basis(
        c_hat, beta_hat, A, xQ2, klin=_klin, kquad=_kquad,
        eft_offset=eft_offset,
    )
    rng = np.random.default_rng(seed)
    boot_samples = np.full(n_boot, np.nan)

    # Sigma is fixed across replicates, so factor it once instead of letting
    # rng.multivariate_normal redo an SVD of Sigma on every draw.
    N = mu_hat.shape[0]
    L = np.linalg.cholesky(Sigma)

    for b in range(n_boot):
        if verbose and (b + 1) % 50 == 0:
            print(f"  Bootstrap replicate {b+1}/{n_boot}")
        D_star = mu_hat + L @ rng.standard_normal(N)
        try:
            if opt == "joint":
                res = fit_beta_c(D_star, A, _v_c, _v_prime_c, c0=c0, w_norm=w_norm, xQ2=xQ2,
                                 Sigma=Sigma, nonneg_beta=nonneg_beta, 
                                 pin_endpoints=pin_endpoints, opt_method=beta_opt,
                                 return_history=False, loss_scale=loss_scale,
                                 eft_offset=eft_offset)
                beta_b, c_b = res.beta_hat, res.c_hat
            else:  # separate
                beta_b = mle_beta(D=D_star, A=A, c=c0, v=_v_c, xQ2=xQ2,
                                  Sigma=Sigma, w_norm=w_norm,
                                  nonneg_beta=nonneg_beta, pin_endpoints=pin_endpoints,
                                  beta_opt=beta_opt, return_history=False,
                                  loss_scale=loss_scale,
                                  eft_offset=eft_offset)
                c_b = mle_c(D=D_star, A=A, v=_v_c, beta=beta_b, xQ2=xQ2,
                            Sigma=Sigma, c_opt=c_opt, c_range=c_range,
                            eft_offset=eft_offset)

            if estimator == "mle":
                boot_samples[b] = c_b
            else:
                K_b       = build_K_matrix(c_b, A, xQ2, klin=_klin, kquad=_kquad)
                K_prime_b = build_K_prime_matrix(c_b, A, xQ2, klin=_klin, kquad=_kquad)
                fixed_mean_b = _v_c(c_b, xQ2) * eft_offset
                fixed_mean_prime_b = _v_prime_c(c_b, xQ2) * eft_offset
                c_os_b, _ = one_step_c(D_star, K_b, K_prime_b, beta_b, c_b,
                                       w_norm=w_norm, Sigma=Sigma,
                                       pin_endpoints=pin_endpoints,
                                       fixed_mean=fixed_mean_b,
                                       fixed_mean_prime=fixed_mean_prime_b)
                boot_samples[b] = c_os_b
        except Exception as e:
            if verbose:
                print(f"  Replicate {b+1} failed: {e}")

    n_failed = np.sum(~np.isfinite(boot_samples))
    if n_failed:
        print(f"Warning: {n_failed}/{n_boot} replicates failed and were excluded.")

    if ci_method == "normal":
        se_boot = float(np.nanstd(boot_samples, ddof=1))
        z = norm.ppf(1 - alpha / 2)
        ci_lower = c_hat - z * se_boot
        ci_upper = c_hat + z * se_boot
    else:  # percentile
        ci_lower = float(np.nanpercentile(boot_samples, 100 * alpha / 2))
        ci_upper = float(np.nanpercentile(boot_samples, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper, boot_samples


# =============================================================================
# 5. Plotting
# =============================================================================

def plot_by_Q2(
    Y,
    xQ2,
    ncols=2,
    y_label="Value",
    x_label=r"$x$",
    x_log_scale=False,
    series_styles=None,
    default_style=None,
    show_legend=True,
    legend_fontsize=None,
    title_fontsize=None,
    label_fontsize=None,
    sort_x_within_panel=True,
    q2_round_decimals=12,
    q2_filter=None,
    alpha=0.05,
    pull=None,
    pull_label="Pull",
    pull_legend=False,
    y_label_first_column_only=False,
    legend_mode="first_panel",
    legend_loc="lower center",
    legend_ncols=None,
):
    """
    Plot one or multiple series grouped into Q^2 panels.

    Parameters
    ----------
    Y : array-like, list/tuple of array-like, or dict[str, array-like]
        Each series must have length n_pts, aligned with rows of xQ2.
    xQ2 : (n_pts, 2) array
        Columns are (x, Q^2).
    ncols : int
        Number of subplot columns.
    y_label : str
        Y-axis label for every panel.
    x_label : str
        X-axis label for every panel.
    x_log_scale : bool
        Use log scale on the x-axis.
    series_styles : dict, optional
        Per-series style dict; may include 'kind' in {"line", "scatter"} plus
        any matplotlib kwargs.  To add error bars for a series, include
        'cov': a (n_pts, n_pts) covariance matrix — the half-width is
        z * sqrt(diag(cov)) where z = norm.ppf(1 - alpha/2).
    default_style : dict, optional
        Fallback style dict.
    show_legend : bool
        Show legend in the first panel.
    legend_fontsize : int or float, optional
        Font size for the legend text. If None, uses matplotlib's default.
    title_fontsize : int or float, optional
        Font size for the panel titles. If None, uses matplotlib's default.
    label_fontsize : int or float, optional
        Font size for the axis labels. If None, uses matplotlib's default.
    sort_x_within_panel : bool
        Sort points by x within each Q^2 panel before plotting.
    q2_round_decimals : int
        Rounding precision for grouping Q^2 values (avoids float-equality issues).
    q2_filter : list/array or callable, optional
        Subset of Q^2 values to plot. If a list/array, only panels whose Q^2
        is in the list (matched within q2_round_decimals precision) are shown.
        If callable, it receives the Q^2 value and must return True to include it.
    alpha : float, optional
        Significance level for error bars (default 0.05, i.e. 95% CI).  Only
        used when a series provides a 'cov' key in series_styles.
    pull : dict or list of dict, optional
        If provided, draw a pull panel below each Q^2 panel.
        Each entry must have keys:
          "data"  : name of the data series (numerator: D - ref)
          "ref"   : name of the prediction series (subtracted from data)
        Optional keys per entry:
          "label" : legend label for this pull series (default: "data vs ref")
          "cov"   : (n_pts, n_pts) covariance for sigma; falls back to the
                    'cov' in series_styles[data] if omitted
          any other key is passed as a matplotlib scatter kwarg (color, s, etc.)
        A single dict is treated as a one-element list.
        Pull is defined as (data - ref) / sigma pointwise.
    pull_label : str, optional
        Y-axis label for the pull panels (default "Pull").
    pull_legend : bool, optional
        Whether to show a legend in the pull panels (default False).
    y_label_first_column_only : bool, optional
        If True, show main- and pull-panel y-axis labels only in the first
        subplot column. Tick labels are retained in every panel.
    legend_mode : {"first_panel", "figure"}, optional
        Place the main-series legend in the first panel (default) or as one
        figure-level legend below all panels.
    legend_loc : str, optional
        Matplotlib location for the figure-level legend.
    legend_ncols : int, optional
        Number of columns in the figure-level legend. Defaults to one column
        per plotted series.
    """
    xQ2 = np.asarray(xQ2, dtype=float)
    if xQ2.ndim != 2 or xQ2.shape[1] != 2:
        raise ValueError(f"xQ2 must have shape (n_pts, 2), got {xQ2.shape}")

    x = xQ2[:, 0]
    Q2 = xQ2[:, 1]
    n_pts = xQ2.shape[0]

    # Normalize Y into list of (label, vec)
    series = []
    if isinstance(Y, dict):
        for lab, vec in Y.items():
            series.append((lab, np.asarray(vec, dtype=float)))
    elif isinstance(Y, (list, tuple)):
        for i, vec in enumerate(Y):
            series.append((f"Y{i+1}", np.asarray(vec, dtype=float)))
    else:
        series = [("Y", np.asarray(Y, dtype=float))]

    for lab, vec in series:
        if vec.size != n_pts:
            raise ValueError(f"Series '{lab}' has length {vec.size}, expected {n_pts}.")

    if series_styles is None:
        series_styles = {}
    if default_style is None:
        default_style = dict(kind="line", linewidth=2)
    if legend_mode not in ("first_panel", "figure"):
        raise ValueError("legend_mode must be 'first_panel' or 'figure'.")

    Q2_key = np.round(Q2, q2_round_decimals)
    uniq_keys = np.unique(Q2_key)

    q2_values, idx_groups = [], []
    for k in uniq_keys:
        idx = np.where(Q2_key == k)[0]
        idx_groups.append(idx)
        q2_values.append(np.mean(Q2[idx]))
    order = np.argsort(q2_values)
    idx_groups = [idx_groups[i] for i in order]
    q2_values = [q2_values[i] for i in order]

    if q2_filter is not None:
        if callable(q2_filter):
            keep = [q2_filter(q) for q in q2_values]
        else:
            filter_keys = np.round(np.asarray(q2_filter, dtype=float), q2_round_decimals)
            keep = [np.round(q, q2_round_decimals) in filter_keys for q in q2_values]
        idx_groups = [g for g, k in zip(idx_groups, keep) if k]
        q2_values = [q for q, k in zip(q2_values, keep) if k]

    n_Q2 = len(idx_groups)
    nrows = int(np.ceil(float(n_Q2) / float(ncols)))

    # Normalize pull to a list of specs and resolve data/sigma for each
    pull_specs = None
    if pull is not None:
        raw = pull if isinstance(pull, list) else [pull]
        series_dict = dict(series)
        pull_specs = []
        for spec in raw:
            spec = dict(spec)
            data_name = spec.pop("data")
            ref_name = spec.pop("ref")
            label = spec.pop("label", f"{data_name} vs {ref_name}")
            if data_name not in series_dict:
                raise ValueError(f"pull 'data' series '{data_name}' not found in Y.")
            if ref_name not in series_dict:
                raise ValueError(f"pull 'ref' series '{ref_name}' not found in Y.")
            cov = spec.pop("cov", None)
            if cov is None:
                cov = series_styles.get(data_name, {}).get("cov", None)
            if cov is None:
                raise ValueError(
                    f"pull for '{data_name} vs {ref_name}' requires a 'cov' either "
                    f"in the pull spec or in series_styles['{data_name}']."
                )
            sigma = np.sqrt(np.diag(np.asarray(cov, dtype=float)))
            pull_specs.append({
                "data_vec": series_dict[data_name],
                "ref_vec": series_dict[ref_name],
                "sigma": sigma,
                "label": label,
                "scatter_kw": spec,  # remaining keys → scatter kwargs
            })

    # Build figure: with pull, each grid cell has a main + pull sub-panel
    if pull_specs is not None:
        
        fig = plt.figure(figsize=(6 * ncols, 4.5 * nrows))
        outer_gs = gridspec.GridSpec(nrows, ncols, hspace=0.4, wspace=0.3)
        main_axes, pull_axes_list = [], []
        for r in range(nrows):
            for c in range(ncols):
                inner_gs = gridspec.GridSpecFromSubplotSpec(
                    2, 1,
                    subplot_spec=outer_gs[r, c],
                    height_ratios=[3, 1],
                    hspace=0.05,
                )
                main_ax = fig.add_subplot(inner_gs[0])
                pull_ax = fig.add_subplot(inner_gs[1], sharex=main_ax)
                main_axes.append(main_ax)
                pull_axes_list.append(pull_ax)
        axes = np.array(main_axes)
        pull_axes_arr = np.array(pull_axes_list)
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        pull_axes_arr = None

    z = norm.ppf(1.0 - alpha / 2.0)

    for i, (idx, q2_val) in enumerate(zip(idx_groups, q2_values)):
        ax = axes[i]
        idxp = idx[np.argsort(x[idx])] if sort_x_within_panel else idx
        x_i = x[idxp]

        if x_log_scale:
            ax.set_xscale("log")

        for lab, vec in series:
            y_i = vec[idxp]
            style = default_style.copy()
            style.update(series_styles.get(lab, {}))
            kind = style.pop("kind", "line")
            cov = style.pop("cov", None)
            if cov is not None:
                cov = np.asarray(cov, dtype=float)
                err_i = z * np.sqrt(np.diag(cov)[idxp])
                fmt = style.pop("fmt", "o" if kind == "scatter" else "-o")
                s = style.pop("s", None)
                if s is not None and "markersize" not in style:
                    style["markersize"] = np.sqrt(s)
                ax.errorbar(x_i, y_i, yerr=err_i, fmt=fmt, label=lab, **style)
            elif kind == "scatter":
                ax.scatter(x_i, y_i, label=lab, **style)
            else:
                ax.plot(x_i, y_i, label=lab, **style)

        ax.set_title(r"$Q^2 = {:.6g}$".format(q2_val), fontsize=title_fontsize)
        if not y_label_first_column_only or i % ncols == 0:
            ax.set_ylabel(y_label, fontsize=label_fontsize)
        ax.grid(True)

        if show_legend and legend_mode == "first_panel" and i == 0:
            ax.legend(fontsize=legend_fontsize)

        # Pull panel
        if pull_specs is not None:
            pax = pull_axes_arr[i]
            for ps in pull_specs:
                pull_vals = (ps["data_vec"][idxp] - ps["ref_vec"][idxp]) / ps["sigma"][idxp]
                kw = {"s": 20}
                kw.update(ps["scatter_kw"])
                pax.scatter(x_i, pull_vals, label=ps["label"], **kw)
            pax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            if not y_label_first_column_only or i % ncols == 0:
                pax.set_ylabel(pull_label, fontsize=label_fontsize)
            pax.grid(True)
            if x_log_scale:
                pax.set_xscale("log")
            if pull_legend and len(pull_specs) > 1 and i == 0:
                pax.legend(fontsize=legend_fontsize)

    # Hide unused panels
    for ax in axes[n_Q2:]:
        ax.axis("off")
    if pull_specs is not None:
        for ax in pull_axes_arr[n_Q2:]:
            ax.axis("off")

    # X-axis labels on bottom-most visible panels
    if pull_specs is not None:
        for i in range(n_Q2):
            r = i // ncols
            is_bottom = (r == nrows - 1) or (i + ncols >= n_Q2)
            if is_bottom:
                pull_axes_arr[i].set_xlabel(x_label, fontsize=label_fontsize)
    else:
        start = max(0, n_Q2 - ncols)
        for ax in axes[start:n_Q2]:
            ax.set_xlabel(x_label, fontsize=label_fontsize)

    if pull_specs is not None:
        for ax in axes[:n_Q2]:
            ax.tick_params(bottom=False, labelbottom=False)

    if show_legend and legend_mode == "figure" and n_Q2:
        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        fig.legend(
            unique.values(), unique.keys(),
            loc=legend_loc,
            bbox_to_anchor=(0.5, 0.01),
            ncol=legend_ncols or max(1, len(unique)),
            fontsize=legend_fontsize,
        )

    if pull_specs is not None:
        if show_legend and legend_mode == "figure":
            fig.subplots_adjust(bottom=0.12)
    else:
        rect = (0, 0.10, 1, 1) if show_legend and legend_mode == "figure" else None
        fig.tight_layout(rect=rect)

    return fig


# =============================================================================
# 6. Experiment
# =============================================================================

def setup_experiment(pdf, c, f, xQ2, grid, degree, M, W,
                     data_var=None, data_Sigma=None, opt="joint",
                     c_opt="scipy_optimize", c_range=(-5.0, 5.0),
                     beta_opt="SLSQP", Lambda=100,
                     c0=0.0, beta0=None,
                     x_min=None, x_max=None, Q2_min=None, Q2_max=None,
                     klin_fn=None, kquad_fn=None,
                     nonneg_beta=True, pin_endpoints=False, ncol_plots=4,
                     x_log_scale=True, parametrize="f",
                     return_history=False, loss_scale=1, mle_constrained=True,
                     one_step_constrained=True, precomputed=None,
                     eft_offset=None, fit_validator=None,
                     one_step_diagonal_covariance=False):
    """
    One-time setup for a simulation experiment.

    Computes the basis, design matrix, and true mean once so they can be reused
    across many simulated datasets.  Call run_single_experiment() repeatedly
    with the returned dict.

    Parameters
    ----------
    pdf : str
        LHAPDF set name (stored for reference).
    c : float
        True Wilson coefficient.
    f : callable
        True PDF f(y).
    xQ2 : (N, 2) array
        (x, Q^2) pairs for the dataset.
    grid : (N_grid,) array
        x-grid for the FK table.
    degree : int
        B-spline degree.
    M : int
        Number of basis functions.
    W : (N, N_grid) array
        FK table matrix.
    data_var : float, optional
        Scalar noise variance (used if data_Sigma is None).
    data_Sigma : (N, N) array, optional
        Full noise covariance (takes precedence over data_var).
    opt : str
        Fitting strategy: "joint" or "separate".
    c_opt : str
        Optimizer for c: "scipy_optimize" or "grid_search".
    c_range : tuple (c_min, c_max)
        Search range passed to ``mle_c`` when ``opt="separate"`` and
        ``c_opt="grid_search"``.
    beta_opt : str
        Optimizer for beta: "SLSQP", "trust-constr", or "closed_form".
    Lambda : float
        New-physics suppression scale used to configure the default coefficient
        functions (default 100). Ignored for a coefficient whose custom callable
        is supplied.
    c0 : float
        Initial guess for c.
    beta0 : (M,) array, optional
        Initial guess for beta.
    x_min, x_max : float
        x range for the basis (typically pdf.xMin / pdf.xMax).
    Q2_min, Q2_max : float
        Q^2 range of the dataset.
    klin_fn, kquad_fn : callable, optional
        Fully configured coefficient functions with signature ``fn(Q2)``.
        When omitted, the corresponding module-level function is bound to
        ``Lambda``.
    nonneg_beta : bool
        Enforce beta >= 0.
    pin_endpoints : bool
        Fix beta[0] = beta[-1] = 0.
    ncol_plots : int
        Columns in diagnostic plots.
    x_log_scale : bool
        Log x-axis in diagnostic plots.
    parametrize : str
        "f"  — basis expands f(y) directly.
        "xf" — basis expands x*f(x); normalization weight uses 1/x factor.
    return_history : bool
        Pass through to fitting routines for convergence diagnostics.
    loss_scale : float
        Scale factor for the loss function.
    mle_constrained : bool
        If True (default), enforce the normalization constraint w_norm @ beta = 1
        during MLE fitting.  If False, fit beta without the constraint (w_norm is
        still computed).
    one_step_constrained : bool
        If True (default), the one-step EIF correction projects the nuisance
        direction h onto the tangent space {h : w_norm @ h = 0}.  If False,
        the one-step correction is unconstrained (w_norm=None is passed to
        one_step_c). Independent of `mle_constrained`.
    precomputed : dict, optional
        Precomputed specialized-model components. It must contain ``A``,
        ``B``, ``w_norm``, ``T``, and ``mu``.
    eft_offset : (N,) array, optional
        Fixed contribution inside the EFT factor, included in every fit.
    fit_validator : callable, optional
        Called as ``fit_validator(beta_hat)`` after each fit.
    one_step_diagonal_covariance : bool
        If True, use only ``diag(data_Sigma)`` for the one-step calculation.

    Returns
    -------
    dict with all setup quantities needed by run_single_experiment().
    """
    if parametrize not in ("f", "xf"):
        raise ValueError(f"parametrize must be 'f' or 'xf', got {parametrize!r}")
    if data_var is None and data_Sigma is None:
        raise ValueError("Must specify either data_var or data_Sigma.")

    klin_local = partial(klin, Lambda=Lambda) if klin_fn is None else klin_fn
    kquad_local = partial(kquad, Lambda=Lambda) if kquad_fn is None else kquad_fn

    if not callable(klin_local) or not callable(kquad_local):
        raise TypeError("klin_fn and kquad_fn must be callable or None.")

    if precomputed is None:
        T = T_sm_FK(f=f, W=W, grid=grid)
        mu = true_mean(c=c, T=T, xQ2=xQ2, klin=klin_local, kquad=kquad_local)

        B = bspline_basis(degree, M, x_min, x_max)

        if parametrize == "f":
            w_norm = np.array([Bk.integrate(x_min, x_max) for Bk in B])
            A = build_f_design_matrix_FK(B=B, W=W, grid=grid)
        else:  # "xf"
            w_norm = np.array([quad(lambda y, Bk=Bk: Bk(y) / y, x_min, x_max)[0] for Bk in B])
            A = build_yf_design_matrix_FK(B=B, W=W, grid=grid)
    else:
        required = {"A", "B", "w_norm", "T", "mu"}
        missing = required.difference(precomputed)
        if missing:
            raise ValueError(
                "precomputed is missing required entries: "
                + ", ".join(sorted(missing))
            )
        A = np.asarray(precomputed["A"], dtype=float)
        B = precomputed["B"]
        w_norm = precomputed["w_norm"]
        if w_norm is not None:
            w_norm = np.asarray(w_norm, dtype=float)
        T = np.asarray(precomputed["T"], dtype=float)
        mu = np.asarray(precomputed["mu"], dtype=float)
        expected_n = len(xQ2)
        if A.ndim != 2 or A.shape[0] != expected_n:
            raise ValueError(
                f"precomputed A must have {expected_n} rows, got {A.shape}."
            )
        if T.shape != (expected_n,) or mu.shape != (expected_n,):
            raise ValueError(
                "precomputed T and mu must both have shape "
                f"{(expected_n,)}; got {T.shape} and {mu.shape}."
            )
        if w_norm is not None and w_norm.shape != (A.shape[1],):
            raise ValueError(
                f"precomputed w_norm must have shape {(A.shape[1],)}, "
                f"got {w_norm.shape}."
            )

    if eft_offset is not None:
        eft_offset = np.asarray(eft_offset, dtype=float)
        if eft_offset.shape != (len(xQ2),):
            raise ValueError(
                f"eft_offset must have shape {(len(xQ2),)}, got {eft_offset.shape}."
            )

    return {
        'PDF': pdf, 'B': B, 'A': A, 'w_norm': w_norm,
        'T': T, 'mu': mu, 'c': c, 'f': f,
        'xQ2': xQ2, 'W': W, 'Lambda': Lambda, 'M': M, 'degree': degree,
        'data_var': data_var, 'data_Sigma': data_Sigma,
        'x_min': x_min, 'x_max': x_max, 'Q2_min': Q2_min, 'Q2_max': Q2_max,
        'klin': klin_local, 'kquad': kquad_local,
        'opt': opt, 'c_opt': c_opt, 'c_range': c_range, 'beta_opt': beta_opt,
        'c0': c0, 'beta0': beta0,
        'nonneg_beta': nonneg_beta, 'pin_endpoints': pin_endpoints,
        'ncol_plots': ncol_plots, 'x_log_scale': x_log_scale,
        'parametrize': parametrize, 'return_history': return_history,
        'loss_scale': loss_scale, 'mle_constrained': mle_constrained,
        'one_step_constrained': one_step_constrained,
        'eft_offset': eft_offset, 'fit_validator': fit_validator,
        'one_step_diagonal_covariance': one_step_diagonal_covariance,
    }


def run_single_experiment(setup_dict, diagnostics=False, show_plots=True, rng=None):
    """
    Run one experiment: generate data, fit (beta, c), apply one-step correction.

    Parameters
    ----------
    setup_dict : dict
        Returned by setup_experiment().
    diagnostics : bool
        If True, compute and return info dict plus print a results table.
    show_plots : bool
        If True (and diagnostics=True), render diagnostic figures. Set to
        False to get info without displaying any plots.
    rng : numpy.random.Generator, optional
        Random-number generator used for the simulated data draw. If omitted,
        a fresh generator is created. Passing one generator across calls makes
        a simulation sequence reproducible without repeating the same draw.

    Returns
    -------
    c_hat : float
        Plug-in MLE of c.
    c_one_step : float
        One-step corrected estimate of c.
    info : dict
        Only returned when diagnostics=True.
    """
    A          = setup_dict['A']
    B          = setup_dict['B']
    w_norm     = setup_dict['w_norm']
    mu         = setup_dict['mu']
    T          = setup_dict['T']
    c          = setup_dict['c']
    f          = setup_dict['f']
    xQ2        = setup_dict['xQ2']
    M          = setup_dict['M']
    data_var   = setup_dict['data_var']
    data_Sigma = setup_dict['data_Sigma']
    x_min      = setup_dict['x_min']
    x_max      = setup_dict['x_max']
    _klin      = setup_dict['klin']
    _kquad     = setup_dict['kquad']
    opt             = setup_dict['opt']
    c_opt           = setup_dict['c_opt']
    c_range         = setup_dict.get('c_range', (-5.0, 5.0))
    beta_opt        = setup_dict['beta_opt']
    c0              = setup_dict['c0']
    beta0           = setup_dict['beta0']
    nonneg_beta     = setup_dict['nonneg_beta']
    pin_endpoints        = setup_dict['pin_endpoints']
    ncol_plots      = setup_dict['ncol_plots']
    x_log_scale     = setup_dict['x_log_scale']
    parametrize     = setup_dict['parametrize']
    return_history  = setup_dict['return_history']
    loss_scale      = setup_dict['loss_scale']
    mle_constrained = setup_dict['mle_constrained']
    one_step_constrained = setup_dict['one_step_constrained']
    eft_offset = setup_dict.get('eft_offset')
    fit_validator = setup_dict.get('fit_validator')
    one_step_diagonal_covariance = setup_dict.get(
        'one_step_diagonal_covariance', False
    )

    fit_w_norm = w_norm if mle_constrained else None
    one_step_w_norm = w_norm if one_step_constrained else None
    #print(f"Fitting with w_norm constraint: {fit_w_norm}, "
    #      f"one-step constraint: {one_step_w_norm}")

    _v_c       = lambda c, xQ2: v_c(c, xQ2, klin=_klin, kquad=_kquad)
    _v_prime_c = lambda c, xQ2: v_prime_c(c, xQ2, klin=_klin, kquad=_kquad)

    # Generate data
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None.")

    if data_Sigma is not None:
        D = rng.multivariate_normal(mean=mu, cov=data_Sigma)
    else:
        D = mu + np.sqrt(data_var) * rng.standard_normal(len(mu))
        data_Sigma = np.eye(len(mu)) * data_var

    # MLE fitting
    if opt == "joint":
        if return_history:
            res, history = fit_beta_c(D, A, _v_c, _v_prime_c, fit_w_norm, xQ2,
                                      Sigma=data_Sigma, beta0=beta0, c0=c0,
                                      eft_offset=eft_offset,
                                      nonneg_beta=nonneg_beta,
                                      pin_endpoints=pin_endpoints, opt_method=beta_opt,
                                      return_history=True, loss_scale=loss_scale)
        else:
            res = fit_beta_c(D, A, _v_c, _v_prime_c, fit_w_norm, xQ2,
                             Sigma=data_Sigma, beta0=beta0, c0=c0,
                             eft_offset=eft_offset,
                             nonneg_beta=nonneg_beta,
                             pin_endpoints=pin_endpoints, opt_method=beta_opt,
                             return_history=False, loss_scale=loss_scale)
        beta_hat = res.beta_hat
        c_hat    = res.c_hat
    elif opt == "separate":
        if return_history:
            beta_hat, history = mle_beta(D=D, A=A, c=c0, v=_v_c, xQ2=xQ2,
                                         Sigma=data_Sigma, w_norm=fit_w_norm, beta0=beta0,
                                         eft_offset=eft_offset,
                                         nonneg_beta=nonneg_beta, pin_endpoints=pin_endpoints,
                                         beta_opt=beta_opt, return_history=True,
                                         loss_scale=loss_scale)
        else:
            beta_hat = mle_beta(D=D, A=A, c=c0, v=_v_c, xQ2=xQ2,
                                Sigma=data_Sigma, w_norm=fit_w_norm, beta0=beta0,
                                eft_offset=eft_offset,
                                nonneg_beta=nonneg_beta, pin_endpoints=pin_endpoints,
                                beta_opt=beta_opt, return_history=False,
                                loss_scale=loss_scale)
        c_hat = mle_c(D=D, A=A, v=_v_c, beta=beta_hat, xQ2=xQ2,
                      Sigma=data_Sigma, eft_offset=eft_offset,
                      c_opt=c_opt, c_range=c_range)
        res = {'beta_hat': beta_hat, 'c_hat': c_hat}
    else:
        raise ValueError(f"Invalid opt method: {opt!r}")

    if fit_validator is not None:
        fit_validator(beta_hat)

    mu_hat = true_mean_basis(
        c_hat, beta_hat, A, xQ2, klin=_klin, kquad=_kquad,
        eft_offset=eft_offset,
    )

    # One-step update
    K       = build_K_matrix(c_hat, A, xQ2, klin=_klin, kquad=_kquad)
    K_prime = build_K_prime_matrix(c_hat, A, xQ2, klin=_klin, kquad=_kquad)
    fixed_mean = (
        _v_c(c_hat, xQ2) * eft_offset if eft_offset is not None else None
    )
    fixed_mean_prime = (
        _v_prime_c(c_hat, xQ2) * eft_offset if eft_offset is not None else None
    )
    one_step_covariance = None if one_step_diagonal_covariance else data_Sigma
    one_step_variances = (
        np.diag(data_Sigma)
        if one_step_diagonal_covariance and data_Sigma is not None else None
    )
    c_one_step, one_step_info = one_step_c(
        D, K, K_prime, beta_hat, c_hat, one_step_w_norm,
        Sigma=one_step_covariance, s2_diag=one_step_variances,
        pin_endpoints=pin_endpoints, fixed_mean=fixed_mean,
        fixed_mean_prime=fixed_mean_prime,
    )
    mu_one_step = true_mean_basis(c_one_step, beta_hat, A, xQ2,
                                  klin=_klin, kquad=_kquad,
                                  eft_offset=eft_offset)

    if not diagnostics:
        return c_hat, c_one_step

    info = {}

    if show_plots:
        # PDF fit plot
        y_eval = np.linspace(x_min, x_max, 5000)
        basis_fit = sum(b * Bk(y_eval) for b, Bk in zip(beta_hat, B))
        fy_fit = basis_fit if parametrize == "f" else basis_fit / y_eval

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(y_eval, f(y_eval), lw=2.5, label="True $f(y)$")
        axes[0].plot(y_eval, fy_fit, lw=2.5, ls="--", label=r"Fitted $\hat f(y)$")
        axes[0].set_title("PDF fit")
        axes[0].set_ylim(0, 5)
        axes[0].set_xlabel("y")
        axes[0].set_ylabel("f(y)")
        axes[0].grid(True, color="0.85", linewidth=1.0)
        axes[0].legend()
        axes[1].plot(y_eval, y_eval * f(y_eval), lw=2.5, label="True $y f(y)$")
        axes[1].plot(y_eval, y_eval * fy_fit, lw=2.5, ls="--", label=r"Fitted $y \hat f(y)$")
        axes[1].set_title("PDF fit")
        axes[1].set_xlabel("y")
        axes[1].set_ylabel("$y f(y)$")
        axes[1].grid(True, color="0.85", linewidth=1.0)
        axes[1].legend()
        plt.tight_layout()
        plt.show()

        # Mean fit by Q^2
        plot_by_Q2(
            {"T": T, "mu": mu, "mu_hat": mu_hat, "Data": D}, #"mu_one_step": mu_one_step,
            xQ2=xQ2, ncols=ncol_plots, x_log_scale=x_log_scale,
            series_styles={
                "T":           dict(linestyle="-",  linewidth=3, color="C0"),
                "mu":          dict(linestyle="--", linewidth=3, color="C1"),
                "mu_hat":      dict(linestyle=":",  linewidth=3, color="C2"),
                #"mu_one_step": dict(linestyle="-.", linewidth=3, color="C3"),
                "Data":        dict(kind="scatter", color="gray", s=30, alpha=0.7),
            }
        )

        # Convergence history
        if return_history:
            history_df = pd.DataFrame(history)
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes[0, 0].plot(history_df['loss'], marker='o')
            axes[0, 0].set_title("Negative Log-Likelihood")
            axes[0, 0].set_xlabel("Iteration"); axes[0, 0].set_ylabel("NLL"); axes[0, 0].grid()
            axes[0, 1].plot(history_df['constr_viol'], marker='o')
            axes[0, 1].set_title("Constraint Violation (w^T beta - 1)")
            axes[0, 1].set_xlabel("Iteration"); axes[0, 1].set_ylabel("Violation"); axes[0, 1].grid()
            axes[1, 0].plot(history_df['grad_norm'], marker='o')
            axes[1, 0].set_title("Gradient Norm")
            axes[1, 0].set_xlabel("Iteration"); axes[1, 0].set_ylabel("||grad||"); axes[1, 0].grid()
            axes[1, 1].plot(history_df['beta_norm'], marker='o')
            axes[1, 1].set_title("Beta Norm")
            axes[1, 1].set_xlabel("Iteration"); axes[1, 1].set_ylabel("||beta||"); axes[1, 1].grid()

        # Numerical summary
        results_df = pd.DataFrame({
            "Parameter": ["c_hat", "c_one_step", "True c"] + [f"beta_{m}" for m in range(M)],
            "Value": np.concatenate([[c_hat, c_one_step, c], beta_hat])
        })
        print(results_df.round(6))
        print("Normalization constraint (w^T beta_hat):", float(w_norm @ beta_hat))
        fy_integral = fixed_quad(
            lambda y: sum(b * Bk(y) for b, Bk in zip(beta_hat, B)) / (y if parametrize == "xf" else 1.0),
            x_min, x_max, n=1000
        )[0]
        print(f"Fitted pdf integrates to ({parametrize} parametrize):", fy_integral)

    info['mle'] = res
    info['one_step'] = one_step_info
    info['D'] = D
    info['beta_hat'] = beta_hat
    return c_hat, c_one_step, info


def experiment(c, f, xQ2, grid, W, degree, M,
               data_var=None, data_Sigma=None, opt="joint",
               c_opt="scipy_optimize", c_range=(-5.0, 5.0),
               beta_opt="SLSQP", c0=0.0, beta0=None,
               Lambda=100, x_min=None, x_max=None, Q2_min=None, Q2_max=None,
               klin_fn=None, kquad_fn=None,
               nonneg_beta=True, diagnostics=False, ncol_plots=4,
               x_log_scale=True, parametrize="f", return_history=False,
               mle_constrained=True, one_step_constrained=True,
               pin_endpoints=False, precomputed=None, eft_offset=None,
               fit_validator=None, one_step_diagonal_covariance=False,
               loss_scale=1):
    """
    Convenience wrapper: runs setup_experiment() then run_single_experiment().

    For simulations prefer calling setup_experiment() once and then
    run_single_experiment() repeatedly to avoid recomputing the design matrix.

    Parameters mirror setup_experiment() and run_single_experiment().

    Returns
    -------
    c_hat, c_one_step[, info]  — same as run_single_experiment().
    """
    setup_dict = setup_experiment(
        c=c, f=f, xQ2=xQ2, grid=grid, W=W, degree=degree, M=M,
        data_var=data_var, data_Sigma=data_Sigma,
        opt=opt, c_opt=c_opt, c_range=c_range,
        beta_opt=beta_opt, c0=c0, beta0=beta0,
        Lambda=Lambda, x_min=x_min, x_max=x_max,
        Q2_min=Q2_min, Q2_max=Q2_max,
        klin_fn=klin_fn, kquad_fn=kquad_fn,
        nonneg_beta=nonneg_beta, pin_endpoints=pin_endpoints,
        ncol_plots=ncol_plots,
        x_log_scale=x_log_scale, parametrize=parametrize,
        mle_constrained=mle_constrained,
        one_step_constrained=one_step_constrained,
        return_history=return_history,
        loss_scale=loss_scale, precomputed=precomputed,
        eft_offset=eft_offset, fit_validator=fit_validator,
        one_step_diagonal_covariance=one_step_diagonal_covariance,
    )
    return run_single_experiment(setup_dict, diagnostics)
