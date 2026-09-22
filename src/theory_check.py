"""Apply the closed-form trend-prior help condition (src/theory.py) to the real networks.

For every (network, density, seed, fold) training set, a maximum-likelihood Matern-5/2 fit
gives the GLS sampling variance V of the standardized elevation coefficient. Each elicited
coefficient prior (v2 consensus, converted to the standardized scale exactly as priors.py
does) then yields the prior weight w = V / (V + tau2) and the MSE ratio of the posterior
mean against GLS. The miscentring delta is measured against the full-network ML slope, the
best available proxy for the true coefficient.

Writes results/paper/tbl_theory_coef.csv (per model and fold) and tbl_theory_coef_tier.csv
(tier aggregate with the predicted RMSE ratio and, where the bayes and eval stages have
run, the observed llm_coef RMSE ratio from results/summary.csv).

  python src/theory_check.py --config config.yaml
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json

import numpy as np
import pandas as pd

import config as cfgmod
import data as datamod
import theory
from run import _iter_folds

NETWORKS = ["main", "andes", "prcp", "urban"]
FRONTIER_IDS = ("claude-opus", "gpt-5.5", "gemini-3.1")
# The sub-threshold model whose log-target back-transform fails on precipitation; it is
# reported separately and kept out of the tier aggregates.
OUTLIER = "gemma4:e2b"


def tier(model) -> str:
    """local = Ollama tag, frontier = the three proprietary systems, openweight = other API models."""
    if not isinstance(model, str) or model == "none":
        return "none"
    if any(k in model for k in FRONTIER_IDS):
        return "frontier"
    return "openweight" if "/" in model else "local"


def _coef_priors(cfg, which, cov, x_sd):
    root = cfg.output_dir / "elicit" / which
    out = {}
    for d in sorted(root.iterdir()):
        f = d / "consensus.json"
        if "__" in d.name or not f.exists():
            continue                      # v2 protocol only: the coefficient prior of llm_coef
        js = json.loads(f.read_text())
        e = (js.get("consensus") or {}).get("coefficients", {}).get(cov)
        if not e:
            continue
        out[js["model"]] = (float(e.get("mean", 0.0)) * x_sd,
                            max(float(e.get("sd", 1.0)) * x_sd, 1e-3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="re-aggregate the saved per-model table without refitting")
    a = ap.parse_args()
    cfg = cfgmod.load(a.config)
    out = cfg.output_dir / "paper"
    if a.aggregate_only:
        aggregate(pd.read_csv(out / "tbl_theory_coef.csv"), cfg.output_dir, out)
        return
    rows = []
    for which in NETWORKS:
        b = datamod.load_dataset(cfg, which)
        coords, X, y = b["coords"], b["X"], b["y"]
        cov = b["covariates"][0]
        Xd = np.column_stack([np.ones(len(y)), X[:, 0]])
        beta_ref = theory.fit_ml_covariance(coords, Xd, y)["beta"][1]
        priors = _coef_priors(cfg, which, cov, float(b["X_sd"][0]))
        print(f"[theory] {which}: beta_ref={beta_ref:.4f}, {len(priors)} models", flush=True)
        for density in cfg.densities:
            for seed in cfg.seeds:
                for fold, tr, _ in _iter_folds(cfg, coords, density, seed):
                    fit = theory.fit_ml_covariance(coords[tr], Xd[tr], y[tr])
                    V = float(fit["V"][1])
                    for model, (mu0, tau) in priors.items():
                        delta = mu0 - beta_ref
                        rows.append(dict(
                            network=which, density=density, seed=seed, fold=fold,
                            n=len(tr), model=model, tier=tier(model), beta_ref=beta_ref,
                            beta_gls=float(fit["beta"][1]), V=V, mu0=mu0, tau2=tau ** 2,
                            delta=delta, w=theory.prior_weight(V, tau ** 2),
                            mse_ratio=theory.posterior_mean_mse(V, tau ** 2, delta) / V,
                            predicted_help=bool(theory.prior_helps(V, tau ** 2, delta))))
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tbl_theory_coef.csv", index=False)
    aggregate(df, cfg.output_dir, out)


def aggregate(df, results, out):
    # With a standardized covariate, a slope MSE reduction of V - MSE lowers the expected
    # squared prediction error by about the same amount, so the predicted RMSE ratio is
    # sqrt(1 - (V - MSE) / RMSE_vague^2).
    df["slope_mse_gain"] = df.V * (1 - df.mse_ratio)
    # Keep only the models that entered llm_coef predictive fits, so predicted and observed
    # ratios describe the same model set.
    cells_long = results / "cells_long.csv"
    if cells_long.exists():
        cells = pd.read_csv(cells_long, low_memory=False,
                            usecols=["dataset", "condition", "elicit_model", "width_scale"])
        fitted = cells[(cells.condition == "llm_coef") & (cells.width_scale.fillna(1.0) == 1.0)] \
            [["dataset", "elicit_model"]].drop_duplicates() \
            .rename(columns={"dataset": "network", "elicit_model": "model"})
        df = df.merge(fitted, on=["network", "model"])
    agg = (df[~df.model.str.startswith(OUTLIER)]
           .groupby(["network", "density", "tier"])
           .agg(n_models=("model", "nunique"), n_train_median=("n", "median"),
                w_mean=("w", "mean"), mse_ratio_mean=("mse_ratio", "mean"),
                slope_mse_gain=("slope_mse_gain", "mean"),
                help_share=("predicted_help", "mean"))
           .reset_index())
    summary = results / "summary.csv"
    if summary.exists():
        s = pd.read_csv(summary)
        s = s[s.predictor == "bayesian_kriging"]
        s = s[~s.elicit_model.astype(str).str.startswith(OUTLIER)]
        vague = (s[s.condition == "vague"].groupby(["dataset", "density"]).rmse_mean.mean()
                 .rename("rmse_vague").reset_index())
        coef = s[s.condition == "llm_coef"].assign(tier=lambda d: d.elicit_model.map(tier))
        coef = (coef.groupby(["dataset", "density", "tier"]).rmse_mean.mean()
                .rename("rmse_coef").reset_index())
        agg = agg.merge(vague.rename(columns={"dataset": "network"}),
                        on=["network", "density"], how="left") \
                 .merge(coef.rename(columns={"dataset": "network"}),
                        on=["network", "density", "tier"], how="left")
        agg["rmse_ratio_predicted"] = np.sqrt(np.clip(
            1 - agg.slope_mse_gain / agg.rmse_vague ** 2, 0, None))
        agg["rmse_ratio_observed"] = agg.rmse_coef / agg.rmse_vague
    agg.to_csv(out / "tbl_theory_coef_tier.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
