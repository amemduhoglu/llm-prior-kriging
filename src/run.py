"""Experiment driver. Reads config.yaml. Fully checkpointed: every
(stage,predictor,condition,model,density,seed,fold) cell writes one JSON row to
results/cells/ the instant it finishes, and is SKIPPED on a re-run, so an interrupted
run resumes where it stopped.

Stages:
  baselines : OK / RK / RF on the chosen dataset, all densities/folds/seeds (no MCMC)
  bayes     : bayesian_kriging for --priors conditions (vague needs no LLM; llm_* needs a
              consensus from results/elicit/<model>/consensus.json)
  eval      : aggregate cells -> results/summary.csv + metric-vs-density figures
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
import data as datamod
import cv as cvmod
import metrics as metricsmod
import models as modelsmod
import priors as priormod


def _cells_dir(cfg) -> Path:
    d = cfg.output_dir / "cells"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cell_key(**kw) -> str:
    s = json.dumps(kw, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:16]


def _done(cfg, key) -> bool:
    return (_cells_dir(cfg) / f"{key}.json").exists()


def _write_cell(cfg, key, row):
    (_cells_dir(cfg) / f"{key}.json").write_text(json.dumps(row, indent=2))


def _run_cell(cfg, predictor, condition, elicit_model, density, seed, fold,
              tr, te, bundle, spec, width_scale=1.0):
    """Fit one model on one fold and return a metric row (predictor-agnostic)."""
    coords, X, y = bundle["coords"], bundle["X"], bundle["y"]
    fn = modelsmod.PREDICTORS[predictor]
    kwargs = dict(seed=seed)
    if predictor == "bayesian_kriging":
        kwargs.update(condition=condition, spec=spec,
                      covariates=bundle["covariates"], mcmc=cfg.mcmc,
                      x_sd=bundle["X_sd"], width_scale=width_scale)
    mu, sd = fn(coords[tr], X[tr], y[tr], coords[te], X[te], y[te], **kwargs)
    m = metricsmod.all_metrics(y[te], mu, sd)
    return m


def _iter_folds(cfg, coords, density, seed):
    """Density-subsample, then spatial blocked folds on the retained points."""
    n = len(coords)
    keep = cvmod.density_subsample(n, density, seed)
    sub_coords = coords[keep]
    k = int(cfg.cv.get("k", 5))
    for fi, (tr_s, te_s) in enumerate(cvmod.blocked_kfold(sub_coords, k, seed)):
        # map back to original indices
        yield fi, keep[tr_s], keep[te_s]


def stage_baselines(cfg, which="pilot"):
    bundle = datamod.load_dataset(cfg, which)
    coords = bundle["coords"]
    baselines = [p for p in cfg["predictors"] if p != "bayesian_kriging"]
    for predictor in baselines:
        for density in cfg.densities:
            for seed in cfg.seeds:
                for fold, tr, te in _iter_folds(cfg, coords, density, seed):
                    key = _cell_key(stage="baselines", dataset=which, predictor=predictor,
                                    condition="classical", model="none",
                                    density=density, seed=seed, fold=fold)
                    if _done(cfg, key):
                        continue
                    t0 = time.time()
                    row = dict(stage="baselines", dataset=which, predictor=predictor,
                               condition="classical", elicit_model="none",
                               density=density, seed=seed, fold=fold)
                    try:
                        m = _run_cell(cfg, predictor, "classical", None, density, seed, fold,
                                      tr, te, bundle, None)
                        row.update(m); row["status"] = "ok"
                    except Exception as e:
                        row["status"] = f"error:{e}"
                        row["trace"] = traceback.format_exc()
                    row["seconds"] = round(time.time() - t0, 2)
                    _write_cell(cfg, key, row)
                    print(f"[baselines] {predictor} d{density} s{seed} f{fold} "
                          f"{row['status']} ({row['seconds']}s)", flush=True)


def _load_consensus(cfg, model_tag, which="pilot", protocol="v2"):
    from elicit import consensus_dirname
    f = cfg.output_dir / "elicit" / which / consensus_dirname(model_tag, protocol) / "consensus.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d.get("consensus")


def stage_bayes(cfg, conditions, elicit_models, which="pilot", protocol="v2",
                width_scales=(1.0,)):
    """Fit bayesian_kriging for every (condition, model, width, density, seed, fold) cell.
    The model is identical across conditions; only the prior block (priors.build_priors)
    changes. width_scales>1 inflates the elicited prior sds (overconfidence sweep)."""
    bundle = datamod.load_dataset(cfg, which)
    coords = bundle["coords"]
    # width>1 only widens *elicited* sds, so it is meaningless for vague/pc_range — skip them.
    ELICITED = set(priormod.COEF_INFORMED_CONDITIONS) | set(priormod.VARIO_LLM_CONDITIONS)
    for condition in conditions:
        # vague / pc_range need no LLM; llm_*/hybrid need each elicit model's consensus
        model_list = ["none"] if condition in priormod.NO_LLM_CONDITIONS else elicit_models
        cond_widths = [w for w in width_scales if w == 1.0 or condition in ELICITED]
        for base_tag in model_list:
            # protocol is part of the model identity in cells (v2 keeps the bare tag)
            model_tag = base_tag if protocol == "v2" or base_tag == "none" \
                else f"{base_tag}__{protocol}"
            spec = None
            if condition not in priormod.NO_LLM_CONDITIONS:
                spec = _load_consensus(cfg, base_tag, which, protocol)
                if spec is None:
                    print(f"[bayes] no consensus for {model_tag} -> "
                          f"falling back to vague for condition {condition}", flush=True)
            for width_scale in cond_widths:
              for density in cfg.densities:
                for seed in cfg.seeds:
                    for fold, tr, te in _iter_folds(cfg, coords, density, seed):
                        eff_cond = condition if (condition in priormod.NO_LLM_CONDITIONS or spec) else "vague_fallback"
                        key_kw = dict(stage="bayes", dataset=which,
                                      predictor="bayesian_kriging",
                                      condition=condition, model=model_tag,
                                      density=density, seed=seed, fold=fold)
                        # width=1 keeps the original key so existing cells stay cached
                        if width_scale != 1.0:
                            key_kw["width"] = width_scale
                        key = _cell_key(**key_kw)
                        if _done(cfg, key):
                            continue
                        t0 = time.time()
                        row = dict(stage="bayes", dataset=which, predictor="bayesian_kriging",
                                   condition=condition, elicit_model=model_tag,
                                   density=density, seed=seed, fold=fold,
                                   width_scale=width_scale,
                                   effective_condition=eff_cond)
                        try:
                            use_spec = spec if (condition not in priormod.NO_LLM_CONDITIONS) else None
                            use_cond = condition if (condition in priormod.NO_LLM_CONDITIONS or spec) else "vague"
                            m = _run_cell(cfg, "bayesian_kriging", use_cond, model_tag,
                                          density, seed, fold, tr, te, bundle, use_spec,
                                          width_scale=width_scale)
                            row.update(m); row["status"] = "ok"
                        except Exception as e:
                            row["status"] = f"error:{e}"
                            row["trace"] = traceback.format_exc()
                        row["seconds"] = round(time.time() - t0, 2)
                        _write_cell(cfg, key, row)
                        print(f"[bayes] {condition}/{model_tag} d{density} s{seed} f{fold} "
                              f"w{width_scale} {row['status']} ({row['seconds']}s)", flush=True)


def stage_eval(cfg):
    import pandas as pd
    rows = []
    for f in sorted(_cells_dir(cfg).glob("*.json")):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception:
            pass
    if not rows:
        print("[eval] no cells yet", flush=True)
        return
    df = pd.DataFrame(rows)
    out = cfg.output_dir
    df.to_csv(out / "cells_long.csv", index=False)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("[eval] no successful cells", flush=True)
        return
    if "dataset" not in ok.columns:
        ok["dataset"] = "pilot"
    # summary.csv carries the baseline (1x) prior width only so the width sweep does not
    # dilute the headline metrics; cells_long.csv above keeps every width for sensitivity use.
    if "width_scale" in ok.columns:
        ok = ok[ok["width_scale"].isna() | (ok["width_scale"] == 1.0)].copy()
    metric_cols = [c for c in ("rmse", "mae", "crps", "pit_ks", "coverage90",
                               "interval_width") if c in ok.columns]
    grp = (ok.groupby(["dataset", "predictor", "condition", "elicit_model", "density"])[metric_cols]
             .agg(["mean", "std"]))
    grp.columns = ["_".join(c) for c in grp.columns]
    grp = grp.reset_index()
    grp.to_csv(out / "summary.csv", index=False)
    print(f"[eval] wrote summary.csv ({len(grp)} rows) and cells_long.csv "
          f"({len(df)} cells)", flush=True)
    _figures(cfg, grp, metric_cols)


def _figures(cfg, grp, metric_cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grp["label"] = grp["predictor"] + "/" + grp["condition"] + grp["elicit_model"].apply(
        lambda m: "" if m == "none" else f"[{m}]")
    datasets = grp["dataset"].unique() if "dataset" in grp.columns else ["pilot"]
    for ds in datasets:
        figdir = cfg.output_dir / "figures" / ds
        figdir.mkdir(parents=True, exist_ok=True)
        gds = grp[grp["dataset"] == ds] if "dataset" in grp.columns else grp
        for metric in metric_cols:
            col = f"{metric}_mean"
            if col not in gds.columns:
                continue
            plt.figure(figsize=(7, 5))
            for lbl, sub in gds.groupby("label"):
                sub = sub.sort_values("density")
                plt.plot(sub["density"], sub[col], marker="o", label=lbl)
            plt.xlabel("data density (% retained)")
            plt.ylabel(f"{metric} (mean over seeds/folds)")
            plt.title(f"{ds}: {metric} vs density")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(figdir / f"{metric}_vs_density.png", dpi=120)
            plt.close()
        print(f"[eval] figures -> {figdir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", required=True, choices=["baselines", "bayes", "eval"])
    ap.add_argument("--priors", default="vague,llm_both",
                    help="comma list of conditions for the bayes stage")
    ap.add_argument("--models", default="local_small",
                    help="elicitation tier name whose models supply llm_* priors")
    ap.add_argument("--dataset", default="pilot",
                    choices=["pilot", "main", "andes", "prcp", "urban"])
    ap.add_argument("--protocol", default="v2", choices=["v2", "v3"])
    ap.add_argument("--width-scales", default="1",
                    help="comma list of multipliers for elicited prior sds (overconfidence "
                         "sweep); e.g. 1,2,3. width=1 reuses existing cached cells.")
    a = ap.parse_args()
    cfg = cfgmod.load(a.config)

    if a.stage == "baselines":
        stage_baselines(cfg, a.dataset)
    elif a.stage == "bayes":
        conditions = [c.strip() for c in a.priors.split(",") if c.strip()]
        tier = cfg.elicitation.get(a.models, [])
        elicit_models = [m["tag"] for m in tier] if tier else []
        width_scales = tuple(float(w) for w in a.width_scales.split(",") if w.strip())
        stage_bayes(cfg, conditions, elicit_models, a.dataset, a.protocol,
                    width_scales=width_scales)
    elif a.stage == "eval":
        stage_eval(cfg)


if __name__ == "__main__":
    main()
