"""
T3_utils.py — Utility functions for the T3 demo.

Sections
--------
1. Basis Functions
2. Wilson Coefficients & Model
3. Design Matrices & Theory Predictions
4. Likelihood & Fitting
5. Plotting
6. Experiment
"""

import numpy as np
from scipy.interpolate import BSpline
from scipy.integrate import quad, fixed_quad
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd
from matplotlib import pyplot as plt


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


# =============================================================================
# 2. Wilson Coefficients & Model
# =============================================================================

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


def true_mean_basis(c, beta, A, xQ2, klin=klin, kquad=kquad):
    """
    Full mean mu = v_c(c, xQ2) * (A @ beta).
    """
    v = v_c(c, xQ2, klin=klin, kquad=kquad)
    T = T_sm_basis(A, beta)
    return v * T


# =============================================================================
# 4. Likelihood & Fitting
# =============================================================================

def make_nll_and_grad(D, A, v, v_prime, xQ2, Sigma=None):
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
        mu = vc * (A @ beta)
        r = D - mu
        return 0.5 * r @ Sigma_inv(r)

    def grad(theta):
        beta = theta[:M]
        c = theta[-1]
        vc = v(c, xQ2=xQ2)
        dv = v_prime(c, xQ2=xQ2)
        mu = vc * (A @ beta)
        r = D - mu
        Sigma_inv_r = Sigma_inv(r)
        g_beta = -((vc[:, None] * A).T @ Sigma_inv_r)
        g_c = -(dv * (A @ beta)).T @ Sigma_inv_r
        return np.r_[g_beta, g_c]

    return nll, grad


def fit_beta_c(D, A, v, v_prime, w_norm, xQ2, Sigma=None, beta0=None, c0=0.0,
               nonneg_beta=False, fix_ends=False, return_history=False,
               opt_method="SLSQP", loss_scale=1):
    """
    Joint MLE of (beta, c) via constrained minimization of the negative log-likelihood.

    Parameters
    ----------
    D : (N,) data vector.
    A : (N, M) design matrix.
    v : callable, v(c, xQ2) -> (N,) vector.
    v_prime : callable, v_prime(c, xQ2) -> (N,) derivative vector.
    w_norm : (M,) normalization weights satisfying w_norm @ beta = 1.
    xQ2 : (N, 2) array of (x, Q^2) pairs.
    Sigma : (N, N) covariance, or None for identity.
    beta0 : (M,) initial beta, or None for random initialization.
    c0 : float, initial c.
    nonneg_beta : bool, enforce beta >= 0.
    fix_ends : bool, fix beta[0] = beta[-1] = 0.
    return_history : bool, if True return (res, history).
    opt_method : "SLSQP" or "trust-constr".
    loss_scale : float, scale factor for the loss (useful for numerical stability).

    Returns
    -------
    res : OptimizeResult with attributes beta_hat and c_hat.
    history : list of per-iteration dicts (only if return_history=True).
    """
    N, M = A.shape
    if w_norm.shape[0] != M:
        raise ValueError(f"w_norm must have length M={M}, got {w_norm.shape[0]}")

    if fix_ends:
        if M < 3:
            raise ValueError("Need at least M >= 3 when fix_ends=True.")
        beta_slice = slice(1, M - 1)
    else:
        beta_slice = slice(0, M)

    M_free = beta_slice.stop - beta_slice.start
    w_free = w_norm[beta_slice]

    if M_free < 1:
        raise ValueError("No free beta coefficients to optimize.")

    def build_full_beta(beta_free):
        beta_full = np.zeros(M)
        beta_full[beta_slice] = beta_free
        return beta_full

    def extract_free_beta(beta_full):
        return np.asarray(beta_full)[beta_slice].copy()

    def normalize_free_beta(beta_free):
        denom = float(w_free @ beta_free)
        if abs(denom) < 1e-15:
            raise ValueError("Cannot normalize beta: w_free @ beta_free is zero or too close to zero.")
        return beta_free / denom

    if beta0 is None:
        b = np.random.rand(M_free)
        if nonneg_beta:
            b = np.abs(b)
        beta0_free = normalize_free_beta(b)
    else:
        beta0 = np.asarray(beta0, dtype=float).copy()
        if beta0.shape[0] != M:
            raise ValueError(f"beta0 must have length M={M}, got {beta0.shape[0]}")
        if fix_ends:
            beta0[0] = 0.0
            beta0[-1] = 0.0
        beta0_free = extract_free_beta(beta0)
        if abs(w_free @ beta0_free - 1.0) > 1e-10:
            beta0_free = normalize_free_beta(beta0_free)

    theta0 = np.r_[beta0_free, c0]

    nll, grad = make_nll_and_grad(D, A, v, v_prime, xQ2=xQ2, Sigma=Sigma)

    def unpack_theta(theta):
        beta_free = theta[:M_free]
        c = theta[-1]
        beta_full = build_full_beta(beta_free)
        return beta_full, c

    def nll_scaled(theta):
        beta_full, c = unpack_theta(theta)
        return nll(np.r_[beta_full, c]) / loss_scale

    def grad_scaled(theta):
        beta_full, c = unpack_theta(theta)
        g_full = grad(np.r_[beta_full, c]) / loss_scale
        return np.r_[g_full[:M][beta_slice], g_full[-1]]

    cons = [{
        "type": "eq",
        "fun": lambda x: float(w_free @ x[:M_free] - 1.0),
        "jac": lambda x: np.r_[w_free, 0.0]
    }]

    history = [] if return_history else None

    def callback(theta, state=None):
        if not return_history:
            return
        beta_full, c = unpack_theta(theta)
        history.append({
            "loss": float(state.fun) if state is not None else float(nll_scaled(theta)),
            "constr_viol": (float(state.constr_violation)
                            if state is not None else float(abs(w_free @ theta[:M_free] - 1.0))),
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

    res = minimize(nll_scaled, theta0, jac=grad_scaled, method=opt_method,
                   constraints=cons, bounds=bounds, options=options,
                   callback=callback if return_history else None)

    num_try = 10
    while not res.success and num_try > 0:
        b = np.random.rand(M_free)
        if nonneg_beta:
            b = np.abs(b)
        beta0_free = normalize_free_beta(b)
        theta0 = np.r_[beta0_free, c0]
        if return_history:
            history.clear()
        res = minimize(nll_scaled, theta0, jac=grad_scaled, method=opt_method,
                       constraints=cons, bounds=bounds, options=options,
                       callback=callback if return_history else None)
        num_try -= 1

    if not res.success:
        print(f"Optimization failed: {res.message}")

    res.beta_hat = build_full_beta(res.x[:M_free])
    res.c_hat = res.x[-1]

    if return_history:
        return res, history
    return res


def mle_beta(D, A, c, v, xQ2, w_norm, Sigma=None, beta0=None, nonneg_beta=True,
             fix_ends=False, beta_opt="SLSQP", return_history=False, loss_scale=1):
    """
    MLE of beta given a fixed c.

    Parameters
    ----------
    D : (N,) data vector.
    A : (N, M) design matrix.
    c : float, fixed Wilson coefficient.
    v : callable, v(c, xQ2) -> (N,) vector.
    xQ2 : (N, 2) array of (x, Q^2) pairs.
    w_norm : (M,) normalization weights (w_norm @ beta = 1).
    Sigma : (N, N) covariance, or None for identity.
    beta0 : (M,) initial guess, or None for random initialization.
    nonneg_beta : bool, enforce beta >= 0.
    fix_ends : bool, fix beta[0] = beta[-1] = 0.
    beta_opt : "SLSQP", "trust-constr", or "closed_form".
    return_history : bool, if True return (beta_hat, history).
    loss_scale : float, scale factor for the loss.

    Returns
    -------
    beta_hat : (M,) array.
    history : list of per-iteration dicts (only if return_history=True).
    """
    N, M = A.shape
    if w_norm.shape[0] != M:
        raise ValueError(f"w_norm must have length M={M}, got {w_norm.shape[0]}")

    if fix_ends:
        if M < 3:
            raise ValueError("Need at least M >= 3 when fix_ends=True.")
        beta_slice = slice(1, M - 1)
    else:
        beta_slice = slice(0, M)

    M_free = beta_slice.stop - beta_slice.start
    w_free = w_norm[beta_slice]

    if M_free < 1:
        raise ValueError("No free beta coefficients to optimize.")

    def build_full_beta(beta_free):
        beta_full = np.zeros(M)
        beta_full[beta_slice] = beta_free
        return beta_full

    def extract_free_beta(beta_full):
        return np.asarray(beta_full)[beta_slice].copy()

    def normalize_free_beta(beta_free):
        denom = float(w_free @ beta_free)
        if abs(denom) < 1e-15:
            raise ValueError("Cannot normalize beta: w_free @ beta_free is zero or too close to zero.")
        return beta_free / denom

    if Sigma is not None:
        cF, lower = cho_factor(Sigma, check_finite=False)
        Sigma_inv = lambda z: cho_solve((cF, lower), z, check_finite=False)
    else:
        Sigma_inv = lambda z: z

    vc = v(c, xQ2=xQ2)

    if beta0 is None:
        b = np.random.rand(M_free)
        if nonneg_beta:
            b = np.abs(b)
        beta0_free = normalize_free_beta(b)
    else:
        beta0 = np.asarray(beta0, dtype=float).copy()
        if beta0.shape[0] != M:
            raise ValueError(f"beta0 must have length M={M}, got {beta0.shape[0]}")
        if fix_ends:
            beta0[0] = 0.0
            beta0[-1] = 0.0
        beta0_free = extract_free_beta(beta0)
        if abs(w_free @ beta0_free - 1.0) > 1e-10:
            print("Initial beta0 does not satisfy the normalization constraint on free coefficients:",
                  "w_free @ beta0_free =", w_free @ beta0_free)
            beta0_free = normalize_free_beta(beta0_free)
        if nonneg_beta and np.any(beta0_free < 0):
            print("Initial beta0 has negative free entries.")

    if beta_opt in ("SLSQP", "trust-constr"):
        def loss(beta_free):
            beta_full = build_full_beta(beta_free)
            mu = vc * (A @ beta_full)
            r = D - mu
            return 0.5 * r @ Sigma_inv(r)

        def grad(beta_free):
            beta_full = build_full_beta(beta_free)
            mu = vc * (A @ beta_full)
            r = D - mu
            g_full = -(vc[:, None] * A).T @ Sigma_inv(r)
            return g_full[beta_slice]

        cons = {
            'type': 'eq',
            'fun': lambda beta_free: float(w_free @ beta_free - 1.0),
            'jac': lambda beta_free: w_free
        }

        if nonneg_beta:
            bounds = [(0.0, None)] * M_free
        else:
            bounds = [(None, None)] * M_free

        history = [] if return_history else None

        def callback(beta_free, state=None):
            if not return_history:
                return
            beta_full = build_full_beta(beta_free)
            history.append({
                'loss': float(state.fun) if state is not None else float(loss(beta_free) / loss_scale),
                'constr_viol': (float(state.constr_violation)
                                if state is not None else float(abs(w_free @ beta_free - 1.0))),
                'grad_norm': (float(state.optimality)
                              if state is not None else float(np.linalg.norm(grad(beta_free) / loss_scale))),
                'beta_norm': float(np.linalg.norm(beta_full)),
                'beta_free_norm': float(np.linalg.norm(beta_free)),
            })

        if beta_opt == "SLSQP":
            options = dict(maxiter=5000, ftol=1e-10)
        elif beta_opt == "trust-constr":
            options = dict(maxiter=5000, gtol=1e-10)

        res = minimize(lambda bf: loss(bf) / loss_scale, beta0_free,
                       jac=lambda bf: grad(bf) / loss_scale,
                       method=beta_opt, constraints=cons, bounds=bounds,
                       callback=callback if return_history else None, options=options)

        num_try = 10
        while not res.success and num_try > 0:
            b = np.random.rand(M_free)
            if nonneg_beta:
                b = np.abs(b)
            beta0_free = normalize_free_beta(b)
            if return_history:
                history.clear()
            res = minimize(lambda bf: loss(bf) / loss_scale, beta0_free,
                           jac=lambda bf: grad(bf) / loss_scale,
                           method=beta_opt, constraints=cons, bounds=bounds,
                           callback=callback if return_history else None, options=options)
            num_try -= 1

        if not res.success:
            print(f"Optimization for beta failed: {res.message}")

        beta_hat = build_full_beta(res.x)
        if return_history:
            return beta_hat, history
        return beta_hat

    elif beta_opt == "closed_form":
        nu = np.asarray(v(c, xQ2=xQ2)).reshape(-1)
        if nu.shape[0] != N:
            raise ValueError(f"v(c) must return shape {(N,)}, got {nu.shape}.")
        Wmat = nu[:, None] * A                                        # (N, M)
        SinvW = np.column_stack([Sigma_inv(Wmat[:, j]) for j in range(M)])
        U = Wmat.T @ SinvW                                            # (M, M)
        b = Wmat.T @ Sigma_inv(D)                                     # (M,)

        KKT = np.zeros((M + 1, M + 1), dtype=float)
        KKT[:M, :M] = U
        KKT[:M, M] = w_norm
        KKT[M, :M] = w_norm
        rhs = np.zeros(M + 1, dtype=float)
        rhs[:M] = b
        rhs[M] = 1.0
        sol = np.linalg.solve(KKT, rhs)

        beta_hat = sol[:M]
        if nonneg_beta and np.any(beta_hat < 0):
            raise ValueError("Closed-form beta has negative entries; try 'SLSQP' or 'trust-constr'.")
        return beta_hat
    else:
        raise ValueError("Invalid beta_opt method.")


def mle_c(D, A, v, beta, xQ2, Sigma=None, c_opt="scipy_optimize"):
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
    c_opt : "scipy_optimize" or "grid_search".

    Returns
    -------
    c_hat : float
    """

    if Sigma is not None:
        cF, lower = cho_factor(Sigma, check_finite=False)
        Sigma_inv = lambda vec: cho_solve((cF, lower), vec, check_finite=False)
    else:
        Sigma_inv = lambda vec: vec

    def nll(c):
        vc = v(c, xQ2=xQ2)
        mu = vc * (A @ beta)
        r = D - mu
        return 0.5 * r @ Sigma_inv(r)

    if c_opt == "scipy_optimize":
        res = minimize(lambda c: nll(c), x0=0.0, method='L-BFGS-B')
        if not res.success:
            res = minimize(lambda c: nll(c), x0=0.0, method='Nelder-Mead')
            if not res.success:
                raise RuntimeError(f"Optimization failed: {res.message}")
        return res.x[0]
    elif c_opt == "grid_search":
        c_grid = np.linspace(-5.0, 5.0, 1000)
        nll_values = np.array([nll(c) for c in c_grid])
        return c_grid[np.argmin(nll_values)]
    else:
        raise ValueError("Invalid c_opt method.")


def one_step_c(D, K, K_prime, beta_hat, c_hat, w_norm, Sigma=None, s2_diag=None):
    """
    One-step estimator for c using the efficient influence function (EIF).

    The nuisance parameter beta is constrained to the tangent space
    {h : w_norm @ h = 0}, and the EIF is projected accordingly.

    Parameters
    ----------
    D : (N,) data vector.
    K : (N, M) matrix such that mu = K @ beta.
    K_prime : (N, M) derivative dmu/dc = K_prime @ beta.
    beta_hat : (M,) plug-in estimate of beta.
    c_hat : float, plug-in estimate of c.
    w_norm : (M,) normalization weights.
    Sigma : (N, N) full covariance, or None.
    s2_diag : (N,) diagonal variance vector (preferred over full Sigma when diagonal).

    Returns
    -------
    c_one_step : float
    info : dict with diagnostics (eif, num, den, lambda, w_dot_beta, w_dot_h).
    """
    D = np.asarray(D).reshape(-1)
    beta_hat = np.asarray(beta_hat).reshape(-1)
    w_norm = np.asarray(w_norm).reshape(-1)
    K = np.asarray(K)
    K_prime = np.asarray(K_prime)

    N, M = K.shape

    if D.shape[0] != N:
        raise ValueError(f"D has length {D.shape[0]} but K has {N} rows.")
    if K_prime.shape != (N, M):
        raise ValueError(f"K_prime must have shape {(N, M)}, got {K_prime.shape}.")
    if beta_hat.shape[0] != M:
        raise ValueError(f"beta_hat has length {beta_hat.shape[0]} but K has {M} cols.")
    if w_norm.shape[0] != M:
        raise ValueError(f"w_norm has length {w_norm.shape[0]} but K has {M} cols.")

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

    mu_hat = K @ beta_hat
    r_hat = D - mu_hat
    mu_prime_hat = K_prime @ beta_hat

    SinvK = Sigma_inv(K)                          # (N, M)
    U = 2 * K.T @ SinvK                           # (M, M)
    b = 2 * K.T @ Sigma_inv(mu_prime_hat)         # (M,)

    # Constrained projection: minimize ||mu' - K h||^2_{Sigma^{-1}} s.t. w^T h = 0
    KKT = np.block([
        [U,                    w_norm[:, None]],
        [w_norm[None, :],      np.zeros((1, 1))]
    ])
    rhs = np.concatenate([b, np.array([0.0])])

    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

    h_star = sol[:M]
    lam = float(sol[M])

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
        "w_dot_beta": float(w_norm @ beta_hat),
        "w_dot_h": float(w_norm @ h_star),
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

def bootstrap_confidence_interval(
    A, w_norm, xQ2, Sigma,
    c_hat, beta_hat,
    c0=0, n_boot=500,
    alpha=0.05,
    ci_method="normal",
    estimator="one_step",
    opt="joint",
    c_opt="scipy_optimize",
    beta_opt="SLSQP",
    nonneg_beta=True,
    fix_ends=False,
    loss_scale=1,
    klin=klin, kquad=kquad,
    seed=None,
    verbose=True,
):
    """
    Parametric bootstrap CI for c (plug-in MLE or one-step estimator).

    Resamples D* ~ N(mu_hat, Sigma) where mu_hat = v_c(c_hat) * A @ beta_hat,
    refits the model on each replicate, and returns a percentile CI.

    Parameters
    ----------
    A : (N, M) design matrix.
    w_norm : (M,) normalization weights for the sum-rule constraint.
    xQ2 : (N, 2) array of (x, Q^2) pairs.
    Sigma : (N, N) covariance matrix of the data.
    c_hat : float, plug-in MLE of c from the original fit.
    beta_hat : (M,) plug-in MLE of beta from the original fit.
    c0 : float, initial guess for c in each bootstrap fit (default 0).
    n_boot : int, number of bootstrap replicates (default 500).
    alpha : float, significance level (default 0.05 for 95% CI).
    estimator : {"one_step", "mle"}
        Which estimator to collect across replicates.
    opt : {"joint", "separate"}, fitting strategy (must match the original fit).
    c_opt : str, optimizer for the c step (when opt="separate").
    beta_opt : str, optimizer for the beta step.
    nonneg_beta : bool, enforce non-negativity of beta.
    fix_ends : bool, fix endpoint basis coefficients.
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

    mu_hat = true_mean_basis(c_hat, beta_hat, A, xQ2, klin=_klin, kquad=_kquad)
    rng = np.random.default_rng(seed)
    boot_samples = np.full(n_boot, np.nan)

    for b in range(n_boot):
        if verbose and (b + 1) % 50 == 0:
            print(f"  Bootstrap replicate {b+1}/{n_boot}")
        D_star = rng.multivariate_normal(mu_hat, Sigma)
        try:
            if opt == "joint":
                res = fit_beta_c(D_star, A, _v_c, _v_prime_c, w_norm, xQ2,
                                 Sigma=Sigma, nonneg_beta=nonneg_beta,
                                 fix_ends=fix_ends, opt_method=beta_opt,
                                 return_history=False, loss_scale=loss_scale)
                beta_b, c_b = res.beta_hat, res.c_hat
            else:  # separate
                beta_b = mle_beta(D=D_star, A=A, c=c0, v=_v_c, xQ2=xQ2,
                                  Sigma=Sigma, w_norm=w_norm,
                                  nonneg_beta=nonneg_beta, fix_ends=fix_ends,
                                  beta_opt=beta_opt, return_history=False,
                                  loss_scale=loss_scale)
                c_b = mle_c(D=D_star, A=A, v=_v_c, beta=beta_b, xQ2=xQ2,
                            Sigma=Sigma, c_opt=c_opt)

            if estimator == "mle":
                boot_samples[b] = c_b
            else:
                K_b       = build_K_matrix(c_b, A, xQ2, klin=_klin, kquad=_kquad)
                K_prime_b = build_K_prime_matrix(c_b, A, xQ2, klin=_klin, kquad=_kquad)
                c_os_b, _ = one_step_c(D_star, K_b, K_prime_b, beta_b, c_b,
                                       w_norm, Sigma=Sigma)
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
    x_log_scale=False,
    series_styles=None,
    default_style=None,
    show_legend=True,
    legend_fontsize=None,
    sort_x_within_panel=True,
    q2_round_decimals=12,
    q2_filter=None,
    alpha=0.05,
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

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

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

        ax.set_title(r"$Q^2 = {:.6g}$".format(q2_val))
        ax.set_ylabel(y_label)
        ax.grid(True)

        if show_legend and i == 0:
            ax.legend(fontsize=legend_fontsize)

    for ax in axes[n_Q2:]:
        ax.axis("off")

    start = max(0, n_Q2 - ncols)
    for ax in axes[start:n_Q2]:
        ax.set_xlabel(r"$x$")

    plt.tight_layout()
    plt.show()


# =============================================================================
# 6. Experiment
# =============================================================================

def setup_experiment(pdf, c, f, xQ2, grid, degree, M, W,
                     data_var=None, data_Sigma=None, opt="joint",
                     c_opt="scipy_optimize", beta_opt="SLSQP", Lambda=100,
                     c0=0.0, beta0=None,
                     x_min=None, x_max=None, Q2_min=None, Q2_max=None,
                     klin=klin, kquad=kquad,
                     nonneg_beta=True, fix_ends=False, ncol_plots=4,
                     x_log_scale=True, parametrize="f",
                     return_history=False, loss_scale=1):
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
    beta_opt : str
        Optimizer for beta: "SLSQP", "trust-constr", or "closed_form".
    Lambda : float
        New-physics suppression scale (default 100).
    c0 : float
        Initial guess for c.
    beta0 : (M,) array, optional
        Initial guess for beta.
    x_min, x_max : float
        x range for the basis (typically pdf.xMin / pdf.xMax).
    Q2_min, Q2_max : float
        Q^2 range of the dataset.
    klin, kquad : callable
        Wilson coefficient functions (default: module-level klin/kquad).
    nonneg_beta : bool
        Enforce beta >= 0.
    fix_ends : bool
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

    Returns
    -------
    dict with all setup quantities needed by run_single_experiment().
    """
    if parametrize not in ("f", "xf"):
        raise ValueError(f"parametrize must be 'f' or 'xf', got {parametrize!r}")
    if data_var is None and data_Sigma is None:
        raise ValueError("Must specify either data_var or data_Sigma.")

    klin_local = lambda z: klin(z, Lambda=Lambda)
    kquad_local = lambda z: kquad(z, Lambda=Lambda)

    T = T_sm_FK(f=f, W=W, grid=grid)
    mu = true_mean(c=c, T=T, xQ2=xQ2, klin=klin_local, kquad=kquad_local)

    B = bspline_basis(degree, M, x_min, x_max)

    if parametrize == "f":
        w_norm = np.array([Bk.integrate(x_min, x_max) for Bk in B])
        A = build_f_design_matrix_FK(B=B, W=W, grid=grid)
    else:  # "xf"
        w_norm = np.array([quad(lambda y, Bk=Bk: Bk(y) / y, x_min, x_max)[0] for Bk in B])
        A = build_yf_design_matrix_FK(B=B, W=W, grid=grid)

    return {
        'PDF': pdf, 'B': B, 'A': A, 'w_norm': w_norm,
        'T': T, 'mu': mu, 'c': c, 'f': f,
        'xQ2': xQ2, 'W': W, 'Lambda': Lambda, 'M': M, 'degree': degree,
        'data_var': data_var, 'data_Sigma': data_Sigma,
        'x_min': x_min, 'x_max': x_max, 'Q2_min': Q2_min, 'Q2_max': Q2_max,
        'klin': klin_local, 'kquad': kquad_local,
        'opt': opt, 'c_opt': c_opt, 'beta_opt': beta_opt,
        'c0': c0, 'beta0': beta0,
        'nonneg_beta': nonneg_beta, 'fix_ends': fix_ends,
        'ncol_plots': ncol_plots, 'x_log_scale': x_log_scale,
        'parametrize': parametrize, 'return_history': return_history,
        'loss_scale': loss_scale,
    }


def run_single_experiment(setup_dict, diagnostics=False, show_plots=True):
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
    beta_opt        = setup_dict['beta_opt']
    c0              = setup_dict['c0']
    beta0           = setup_dict['beta0']
    nonneg_beta     = setup_dict['nonneg_beta']
    fix_ends        = setup_dict['fix_ends']
    ncol_plots      = setup_dict['ncol_plots']
    x_log_scale     = setup_dict['x_log_scale']
    parametrize     = setup_dict['parametrize']
    return_history  = setup_dict['return_history']
    loss_scale      = setup_dict['loss_scale']

    _v_c       = lambda c, xQ2: v_c(c, xQ2, klin=_klin, kquad=_kquad)
    _v_prime_c = lambda c, xQ2: v_prime_c(c, xQ2, klin=_klin, kquad=_kquad)

    # Generate data
    if data_Sigma is not None:
        rng = np.random.default_rng(seed=None)
        D = rng.multivariate_normal(mean=mu, cov=data_Sigma)
    else:
        D = mu + np.sqrt(data_var) * np.random.randn(len(mu))
        data_Sigma = np.eye(len(mu)) * data_var

    # MLE fitting
    if opt == "joint":
        if return_history:
            res, history = fit_beta_c(D, A, _v_c, _v_prime_c, w_norm, xQ2,
                                      Sigma=data_Sigma, nonneg_beta=nonneg_beta,
                                      fix_ends=fix_ends, opt_method=beta_opt,
                                      return_history=True, loss_scale=loss_scale)
        else:
            res = fit_beta_c(D, A, _v_c, _v_prime_c, w_norm, xQ2,
                             Sigma=data_Sigma, nonneg_beta=nonneg_beta,
                             fix_ends=fix_ends, opt_method=beta_opt,
                             return_history=False, loss_scale=loss_scale)
        beta_hat = res.beta_hat
        c_hat    = res.c_hat
    elif opt == "separate":
        if return_history:
            beta_hat, history = mle_beta(D=D, A=A, c=c0, v=_v_c, xQ2=xQ2,
                                         Sigma=data_Sigma, w_norm=w_norm, beta0=None,
                                         nonneg_beta=nonneg_beta, fix_ends=fix_ends,
                                         beta_opt=beta_opt, return_history=True,
                                         loss_scale=loss_scale)
        else:
            beta_hat = mle_beta(D=D, A=A, c=c0, v=_v_c, xQ2=xQ2,
                                Sigma=data_Sigma, w_norm=w_norm, beta0=None,
                                nonneg_beta=nonneg_beta, fix_ends=fix_ends,
                                beta_opt=beta_opt, return_history=False,
                                loss_scale=loss_scale)
        c_hat = mle_c(D=D, A=A, v=_v_c, beta=beta_hat, xQ2=xQ2,
                      Sigma=data_Sigma, c_opt=c_opt)
        res = {'beta_hat': beta_hat, 'c_hat': c_hat}
    else:
        raise ValueError(f"Invalid opt method: {opt!r}")

    mu_hat = true_mean_basis(c_hat, beta_hat, A, xQ2, klin=_klin, kquad=_kquad)

    # One-step update
    K       = build_K_matrix(c_hat, A, xQ2, klin=_klin, kquad=_kquad)
    K_prime = build_K_prime_matrix(c_hat, A, xQ2, klin=_klin, kquad=_kquad)
    c_one_step, one_step_info = one_step_c(D, K, K_prime, beta_hat, c_hat,
                                           w_norm, Sigma=data_Sigma)
    mu_one_step = true_mean_basis(c_one_step, beta_hat, A, xQ2,
                                  klin=_klin, kquad=_kquad)

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
            {"T": T, "mu": mu, "mu_hat": mu_hat, "mu_one_step": mu_one_step, "Data": D},
            xQ2=xQ2, ncols=ncol_plots, x_log_scale=x_log_scale,
            series_styles={
                "T":           dict(linestyle="-",  linewidth=3, color="C0"),
                "mu":          dict(linestyle="--", linewidth=3, color="C1"),
                "mu_hat":      dict(linestyle=":",  linewidth=3, color="C2"),
                "mu_one_step": dict(linestyle="-.", linewidth=3, color="C3"),
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
               c_opt="scipy_optimize", beta_opt="SLSQP", c0=0.0, beta0=None,
               Lambda=100, x_min=None, x_max=None, Q2_min=None, Q2_max=None,
               klin=klin, kquad=kquad,
               nonneg_beta=True, diagnostics=False, ncol_plots=4,
               x_log_scale=True, parametrize="f", return_history=False):
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
        opt=opt, c_opt=c_opt, beta_opt=beta_opt, c0=c0, beta0=beta0,
        Lambda=Lambda, x_min=x_min, x_max=x_max,
        Q2_min=Q2_min, Q2_max=Q2_max, klin=klin, kquad=kquad,
        nonneg_beta=nonneg_beta, ncol_plots=ncol_plots,
        x_log_scale=x_log_scale, parametrize=parametrize,
        return_history=return_history,
    )
    return run_single_experiment(setup_dict, diagnostics)
