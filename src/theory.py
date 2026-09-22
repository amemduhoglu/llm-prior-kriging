"""Closed-form prior-versus-likelihood weight for a Normal prior on one regression-kriging
coefficient, conditional on the covariance parameters.

With GLS estimate b_hat ~ N(beta, V) and prior beta ~ N(mu0, tau2), the posterior mean is
w * mu0 + (1 - w) * b_hat with w = V / (V + tau2). Its MSE is (1-w)^2 V + w^2 delta^2
(delta = mu0 - beta), which is below V exactly when delta^2 < V + 2 tau2.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def matern52(d, ls):
    a = np.sqrt(5.0) * np.asarray(d, float) / ls
    return (1 + a + a * a / 3.0) * np.exp(-a)


def gls_coef_variance(X, Sigma, j):
    """Sampling variance of the j-th GLS coefficient, [(X' Sigma^-1 X)^-1]_jj."""
    L = np.linalg.cholesky(Sigma)
    Xw = np.linalg.solve(L, X)
    return float(np.linalg.inv(Xw.T @ Xw)[j, j])


def prior_weight(V, tau2):
    return V / (V + tau2)


def posterior_mean_mse(V, tau2, delta):
    w = prior_weight(V, tau2)
    return (1 - w) ** 2 * V + w ** 2 * delta ** 2


def prior_helps(V, tau2, delta):
    """The prior-centred posterior mean beats GLS in MSE iff delta^2 < V + 2 tau^2."""
    return delta ** 2 < V + 2 * tau2


def _profile(theta, d, X, y):
    ls, eta2, sig2 = np.exp(theta)
    S = eta2 * matern52(d, ls) + (sig2 + 1e-9) * np.eye(len(y))
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return np.inf, None
    Xw = np.linalg.solve(L, X)
    yw = np.linalg.solve(L, y)
    A = Xw.T @ Xw
    beta = np.linalg.solve(A, Xw.T @ yw)
    r = yw - Xw @ beta
    nll = 0.5 * r @ r + np.log(np.diag(L)).sum()
    return nll, (beta, np.diag(np.linalg.inv(A)), ls, eta2, sig2)


def fit_ml_covariance(coords, X, y, n_starts=3):
    """Maximum-likelihood Matern-5/2 + nugget fit with the trend profiled out by GLS.
    Returns beta, V (GLS sampling variances), ls, eta2, sigma2 at the optimum."""
    coords = np.asarray(coords, float)
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    d = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    span = float(np.hypot(*np.ptp(coords, axis=0))) or 1.0
    var = float(np.var(y)) or 1.0
    best = None
    nn = float(np.sort(d, axis=1)[:, 1].min()) or 1e-3 * span
    bounds = [(np.log(0.1 * nn), np.log(10 * span)), (np.log(1e-4 * var), np.log(100 * var)),
              (np.log(1e-6 * var), np.log(10 * var))]
    for frac in np.geomspace(0.02, 0.5, n_starts):
        x0 = np.log([frac * span, 0.5 * var, 0.5 * var])
        res = minimize(lambda t: _profile(t, d, X, y)[0], x0, method="L-BFGS-B",
                       bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    _, (beta, V, ls, eta2, sig2) = _profile(best.x, d, X, y)
    return dict(beta=beta, V=V, ls=float(ls), eta2=float(eta2), sigma2=float(sig2),
                nll=float(best.fun))
