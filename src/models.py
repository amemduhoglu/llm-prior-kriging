"""Predictive models. Each predict_* returns (mu, sd) on the test points, where sd is the
predictive standard deviation (kriging sd for classical models, posterior-predictive sd for
the Bayesian model). All consume standardized covariates X and raw coordinates.

The Bayesian-kriging model is IDENTICAL across prior conditions — only priors.build_priors
changes — which is the experiment's core control.
"""
from __future__ import annotations
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def _domain_scale(coords: np.ndarray) -> float:
    span = coords.max(0) - coords.min(0)
    return float(np.hypot(*span))


# ---------------------------------------------------------------- classical baselines

def _ok_fit_predict(cx, cy, cz, tx, ty):
    """Ordinary kriging with a robust variogram-model fallback chain."""
    from pykrige.ok import OrdinaryKriging
    last = None
    for vm in ("spherical", "exponential", "gaussian", "linear"):
        try:
            ok = OrdinaryKriging(cx, cy, cz, variogram_model=vm, verbose=False,
                                 enable_plotting=False)
            z, ss = ok.execute("points", tx, ty)
            return np.asarray(z, float), np.sqrt(np.clip(np.asarray(ss, float), 0, None))
        except Exception as e:
            last = e
    raise RuntimeError(f"OK failed all variogram models: {last}")


def predict_ordinary_kriging(coords_tr, X_tr, y_tr, coords_te, X_te, y_te, **kw):
    mu, sd = _ok_fit_predict(coords_tr[:, 0], coords_tr[:, 1], y_tr,
                             coords_te[:, 0], coords_te[:, 1])
    return mu, sd


def _trend_residual_kriging(trend_pred_tr, trend_pred_te, coords_tr, y_tr, coords_te):
    resid = y_tr - trend_pred_tr
    rmu, rsd = _ok_fit_predict(coords_tr[:, 0], coords_tr[:, 1], resid,
                               coords_te[:, 0], coords_te[:, 1])
    return trend_pred_te + rmu, rsd


def predict_regression_kriging(coords_tr, X_tr, y_tr, coords_te, X_te, y_te, **kw):
    from sklearn.linear_model import LinearRegression
    lin = LinearRegression().fit(X_tr, y_tr)
    return _trend_residual_kriging(lin.predict(X_tr), lin.predict(X_te),
                                   coords_tr, y_tr, coords_te)


def predict_rf_residual_kriging(coords_tr, X_tr, y_tr, coords_te, X_te, y_te, seed=0, **kw):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1).fit(X_tr, y_tr)
    return _trend_residual_kriging(rf.predict(X_tr), rf.predict(X_te),
                                   coords_tr, y_tr, coords_te)


# ---------------------------------------------------------------- Bayesian kriging

def predict_bayesian_kriging(coords_tr, X_tr, y_tr, coords_te, X_te, y_te, *,
                             condition, spec, covariates, mcmc, x_sd=None, seed=0,
                             width_scale=1.0, return_idata=False, **kw):
    """Latent GP marginalized analytically (pm.gp.Marginal, Gaussian likelihood) so it stays
    CPU-tractable. Covariate trend enters as a Linear mean over the covariate dims; the
    Matern52 covariance acts only on the coordinate dims (active_dims=[0,1])."""
    import pymc as pm
    import pytensor.tensor as pt
    from priors import build_priors

    p = X_tr.shape[1]
    dom = _domain_scale(coords_tr)
    y_scale = float(np.std(y_tr)) or 1.0

    Xtr_full = np.hstack([coords_tr, X_tr]).astype(float)
    Xte_full = np.hstack([coords_te, X_te]).astype(float)

    with pm.Model() as model:
        pr = build_priors(condition, spec, n_cov=p, covariates=covariates,
                          domain_scale=dom, y_scale=y_scale, x_sd=x_sd,
                          width_scale=width_scale)
        # full coeff vector: zeros over the 2 coord dims, betas over the p covariate dims
        coeffs_full = pt.concatenate([pt.zeros(2), pr["betas"]])
        mean_func = pm.gp.mean.Linear(coeffs=coeffs_full, intercept=pr["beta0"])
        cov = pr["eta"] ** 2 * pm.gp.cov.Matern52(input_dim=2 + p, ls=pr["ls"],
                                                  active_dims=[0, 1])
        gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov)
        gp.marginal_likelihood("y", X=Xtr_full, y=y_tr, sigma=pr["sigma"])

        idata = pm.sample(
            draws=int(mcmc.get("draws", 1000)), tune=int(mcmc.get("tune", 1000)),
            chains=int(mcmc.get("chains", 2)), cores=int(mcmc.get("cores", 1)),
            target_accept=float(mcmc.get("target_accept", 0.9)),
            random_seed=seed, progressbar=False,
            compute_convergence_checks=bool(return_idata),
        )

    with model:
        gp.conditional("y_pred", Xnew=Xte_full, pred_noise=True)
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["y_pred"], random_seed=seed, progressbar=False)

    arr = ppc.posterior_predictive["y_pred"].values  # (chain, draw, n_te)
    arr = arr.reshape(-1, arr.shape[-1])
    if return_idata:
        return arr.mean(0), arr.std(0), idata
    return arr.mean(0), arr.std(0)


PREDICTORS = {
    "ordinary_kriging": predict_ordinary_kriging,
    "regression_kriging": predict_regression_kriging,
    "rf_residual_kriging": predict_rf_residual_kriging,
    "bayesian_kriging": predict_bayesian_kriging,
}
