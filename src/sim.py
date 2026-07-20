"""Simulation study: how much does a misspecified variogram-range prior hurt, as a
function of (a) how wrong it is and (b) data density?

Ground truth is KNOWN here: synthetic Matern-5/2 GP fields on a 1000 km square domain.
For each replicate we fit the same Bayesian model under
  - the vague range prior (our standard control), and
  - a CONFIDENT LogNormal range prior centred at ratio r x true range,
    r in RATIOS (0.01x ... 100x), with log-sd 0.3 (the "high confidence" an LLM states).
Held-out RMSE/CRPS/coverage vs r gives the misspecification curve; in the analysis the
elicited prior of every LLM is then placed on that curve via its (elicited range / fitted
range) ratio. Converts the empirical "variogram priors hurt" finding into a quantitative
mechanism. Checkpointed per cell.

  python src/sim.py --config config.yaml
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np

import config as cfgmod
import metrics as metricsmod

# ---- design (fixed; logged into every cell) ----
DOMAIN_M = 1_000_000.0          # 1000 km square
TRUE_RANGE = 150_000.0          # 150 km — same order as the Himalaya residual field
TRUE_SILL = 4.0
TRUE_NUGGET = 0.5
N_OBS = [12, 30, 60]            # matches the d10/d25/d50 station counts of `main`
N_TEST = 60
RATIOS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]   # prior centre / true range
PRIOR_LOG_SD = 0.3              # the tight sd of a "high confidence" elicitation
N_REPS = 12
MCMC = dict(draws=500, tune=500, chains=2, cores=1, target_accept=0.9)


def _matern52(d, ls):
    a = np.sqrt(5.0) * d / ls
    return (1 + a + a * a / 3.0) * np.exp(-a)


def simulate_field(rng, n_total):
    """Sample coords + one realisation of the true GP (+ nugget noise)."""
    coords = rng.uniform(0, DOMAIN_M, size=(n_total, 2))
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    K = TRUE_SILL * _matern52(d, TRUE_RANGE) + np.eye(n_total) * 1e-8
    f = rng.multivariate_normal(np.zeros(n_total), K)
    y = f + rng.normal(0, np.sqrt(TRUE_NUGGET), n_total)
    return coords, y


def fit_predict(coords_tr, y_tr, coords_te, prior_kind, range_center, seed):
    """Same model family as the real experiment (constant mean + Matern52 GP, marginalized);
    only the range prior differs between conditions."""
    import pymc as pm
    y_scale = float(np.std(y_tr)) or 1.0
    with pm.Model() as model:
        beta0 = pm.Normal("beta0", 0.0, 10.0 * y_scale)
        if prior_kind == "vague":
            dom = float(np.hypot(DOMAIN_M, DOMAIN_M))
            ls = pm.LogNormal("ls", mu=np.log(dom / 3.0), sigma=1.0)
        else:  # confident, possibly misspecified
            ls = pm.LogNormal("ls", mu=np.log(range_center), sigma=PRIOR_LOG_SD)
        eta2 = pm.HalfNormal("eta2", sigma=y_scale)
        sigma2 = pm.HalfNormal("sigma2", sigma=y_scale)
        cov = eta2 * pm.gp.cov.Matern52(2, ls=ls)
        gp = pm.gp.Marginal(mean_func=pm.gp.mean.Constant(beta0), cov_func=cov)
        gp.marginal_likelihood("y", X=coords_tr, y=y_tr, sigma=pm.math.sqrt(sigma2))
        idata = pm.sample(random_seed=seed, progressbar=False,
                          compute_convergence_checks=False, **MCMC)
    with model:
        gp.conditional("y_pred", Xnew=coords_te, pred_noise=True)
        ppc = pm.sample_posterior_predictive(idata, var_names=["y_pred"],
                                             random_seed=seed, progressbar=False)
    arr = ppc.posterior_predictive["y_pred"].values.reshape(-1, len(coords_te))
    return arr.mean(0), arr.std(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = cfgmod.load(a.config)
    cells = cfg.output_dir / "sim" / "cells"
    cells.mkdir(parents=True, exist_ok=True)

    conditions = [("vague", None)] + [("ratio", r) for r in RATIOS]
    for n_obs in N_OBS:
        for rep in range(N_REPS):
            rng = np.random.default_rng(1000 * n_obs + rep)
            coords, y = simulate_field(rng, n_obs + N_TEST)
            tr = slice(0, n_obs); te = slice(n_obs, None)
            for kind, ratio in conditions:
                key = hashlib.md5(json.dumps(
                    dict(n=n_obs, rep=rep, kind=kind, ratio=ratio)).encode()).hexdigest()[:16]
                out = cells / f"{key}.json"
                if out.exists():
                    continue
                t0 = time.time()
                row = dict(n_obs=n_obs, rep=rep, condition=kind, ratio=ratio,
                           true_range=TRUE_RANGE, prior_log_sd=PRIOR_LOG_SD)
                try:
                    center = None if ratio is None else ratio * TRUE_RANGE
                    mu, sd = fit_predict(coords[tr], y[tr], coords[te], kind, center,
                                         seed=rep)
                    row.update(metricsmod.all_metrics(y[te], mu, sd))
                    row["status"] = "ok"
                except Exception as e:
                    row["status"] = f"error:{e}"
                    row["trace"] = traceback.format_exc()
                row["seconds"] = round(time.time() - t0, 2)
                out.write_text(json.dumps(row, indent=2))
                print(f"[sim] n{n_obs} rep{rep} {kind}{'' if ratio is None else f'x{ratio}'} "
                      f"{row['status']} ({row['seconds']}s)", flush=True)
    print("[sim] done", flush=True)


if __name__ == "__main__":
    main()
