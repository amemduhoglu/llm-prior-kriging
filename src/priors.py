"""Turn a prior CONDITION + (optional) elicited JSON into pymc distributions.

The Bayesian model is IDENTICAL across conditions — ONLY the priors change. That isolation
is the whole experiment, so the four conditions differ only in which blocks below are
'informed' vs. 'vague':

  vague          : everything weakly-informative
  llm_coef       : informed covariate coefficients, vague variogram
  llm_variogram  : vague coefficients, informed variogram hyperparameters
  llm_both       : both informed
  pc_range       : vague coefficients, principled PC-prior variogram (Fuglstad et al. 2019);
                   the LLM-free, domain-scaled baseline against which llm_variogram is judged.
  hybrid         : informed coefficients (LLM) + PC-prior variogram; the recommended
                   division of labour — elicit the trend, let a principled default set the range.
"""
from __future__ import annotations
import numpy as np
import pymc as pm

CONDITIONS = ("vague", "llm_coef", "llm_variogram", "llm_both", "pc_range", "hybrid")

# Conditions that need NO elicited spec (no LLM model attached); orchestration treats them
# like `vague` — a single "none" model, spec=None.
NO_LLM_CONDITIONS = ("vague", "pc_range")
# Conditions whose variogram block uses the principled PC prior instead of LLM/vague.
PC_VARIO_CONDITIONS = ("pc_range", "hybrid")
# Conditions whose coefficient block is informed by the LLM.
COEF_INFORMED_CONDITIONS = ("llm_coef", "llm_both", "hybrid")
# Conditions whose variogram block is informed by the LLM.
VARIO_LLM_CONDITIONS = ("llm_variogram", "llm_both")


def _lognormal_params(mean: float, sd: float):
    """Underlying-normal (mu, sigma) for a LogNormal with the given positive mean & sd."""
    mean = max(float(mean), 1e-6)
    sd = max(float(sd), 1e-6)
    var = sd ** 2
    sigma2 = np.log1p(var / mean ** 2)
    mu = np.log(mean) - 0.5 * sigma2
    return float(mu), float(np.sqrt(sigma2))


def _pc_invrange_rate(range0: float, alpha: float = 0.5) -> float:
    """PC-prior rate for inv-range. For a 2-D Matern field the PC prior on the range rho
    (Fuglstad, Simpson, Lindgren & Rue 2019) makes 1/rho ~ Exponential(lambda) with
    lambda = -ln(alpha) * range0, satisfying P(rho < range0) = alpha. With alpha=0.5 the
    median range is exactly range0 — we set range0 to a domain-scaled default (D/3), so the
    prior is centred like the vague one but carries the PC tail that penalises tiny ranges."""
    return -np.log(float(alpha)) * max(float(range0), 1e-6)


def _pc_sill_rate(sigma0: float, alpha: float = 0.05) -> float:
    """PC-prior rate for the marginal sd eta: eta ~ Exponential(rate=-ln(alpha)/sigma0),
    giving P(eta > sigma0) = alpha (Fuglstad et al. 2019). sigma0 = sd(y), alpha=0.05."""
    return -np.log(float(alpha)) / max(float(sigma0), 1e-6)


def build_priors(condition: str, spec: dict | None, *, n_cov: int, covariates: list[str],
                 domain_scale: float, y_scale: float, x_sd=None, width_scale: float = 1.0):
    """Create RVs inside an active pm.Model. Returns dict: beta0, betas (n_cov,), ls, eta, sigma.

    domain_scale: characteristic spatial extent (CRS units) — sets the vague range prior.
    y_scale:      sd of the (transformed) target — sets vague sill/nugget scale.
    x_sd:         per-covariate standard deviations. The LLM elicits coefficients in NATURAL
                  covariate units, but the model uses z-scored covariates, so a natural-unit
                  coefficient b maps to a standardized coefficient b * sd. Done here (in code,
                  post-elicitation) so no covariate statistic ever enters the prompt.
    spec:         parsed elicitation dict (coefficients{}, variogram{}) or None.
    width_scale:  multiplier applied ONLY to the *elicited* prior standard deviations (coef sds
                  and variogram-parameter sds). Probes the documented overconfidence of LLM
                  priors (Selby et al. 2024): width_scale>1 inflates the stated sd to test the
                  calibration/accuracy trade-off. Vague and PC priors are unaffected — they are
                  not the overconfident component under test.
    """
    if x_sd is None:
        x_sd = np.ones(n_cov)
    x_sd = np.asarray(x_sd, float)
    coef_informed = condition in COEF_INFORMED_CONDITIONS and spec is not None
    vario_llm = condition in VARIO_LLM_CONDITIONS and spec is not None
    vario_pc = condition in PC_VARIO_CONDITIONS

    beta0 = pm.Normal("beta0", mu=0.0, sigma=10.0 * y_scale)

    # ---- coefficients ----
    if coef_informed:
        mus, sds = [], []
        cspec = spec.get("coefficients", {})
        for i, c in enumerate(covariates):
            e = cspec.get(c, {})
            # natural-unit coefficient -> standardized-covariate scale (multiply by sd)
            mus.append(float(e.get("mean", 0.0)) * x_sd[i])
            sds.append(max(float(e.get("sd", 1.0)) * x_sd[i] * width_scale, 1e-3))
        betas = pm.Normal("betas", mu=np.array(mus), sigma=np.array(sds), shape=n_cov)
    else:
        betas = pm.Normal("betas", mu=0.0, sigma=10.0, shape=n_cov)

    # ---- variogram / GP hyperparameters (all positive) ----
    # Three mutually exclusive regimes per parameter: PC prior (pc_range/hybrid), LLM-informed
    # (llm_variogram/llm_both), or vague. PC priors act on the GRF (range, marginal sd) only;
    # the nugget (observation noise) keeps its weakly-informative prior in every non-LLM case.
    vspec = spec.get("variogram", {}) if spec else {}

    # range / lengthscale
    if vario_pc:
        inv_ls = pm.Exponential("inv_ls", lam=_pc_invrange_rate(domain_scale / 3.0))
        ls = pm.Deterministic("ls", 1.0 / inv_ls)
    elif vario_llm and "range" in vspec:
        mu, sig = _lognormal_params(vspec["range"]["mean"],
                                    vspec["range"].get("sd", vspec["range"]["mean"]) * width_scale)
        ls = pm.LogNormal("ls", mu=mu, sigma=max(sig, 0.1))
    else:
        # vague: centered near a fraction of the domain, broad
        ls = pm.LogNormal("ls", mu=np.log(domain_scale / 3.0), sigma=1.0)

    # sill (marginal variance eta2 / sd eta)
    if vario_pc:
        eta = pm.Exponential("eta", lam=_pc_sill_rate(y_scale))
        eta2 = pm.Deterministic("eta2", eta ** 2)
    elif vario_llm and "sill" in vspec:
        mu, sig = _lognormal_params(vspec["sill"]["mean"],
                                    vspec["sill"].get("sd", vspec["sill"]["mean"]) * width_scale)
        eta2 = pm.LogNormal("eta2", mu=mu, sigma=max(sig, 0.1))
        eta = pm.Deterministic("eta", pm.math.sqrt(eta2))
    else:
        eta2 = pm.HalfNormal("eta2", sigma=y_scale)
        eta = pm.Deterministic("eta", pm.math.sqrt(eta2))

    # nugget (observation noise) — vague unless the LLM informed it
    if vario_llm and "nugget" in vspec:
        mu, sig = _lognormal_params(vspec["nugget"]["mean"],
                                    vspec["nugget"].get("sd", vspec["nugget"]["mean"]) * width_scale)
        sigma2 = pm.LogNormal("sigma2", mu=mu, sigma=max(sig, 0.1))
    else:
        sigma2 = pm.HalfNormal("sigma2", sigma=y_scale)
    sigma = pm.Deterministic("sigma", pm.math.sqrt(sigma2))

    return {"beta0": beta0, "betas": betas, "ls": ls, "eta": eta, "sigma": sigma}


def validate_spec(spec: dict, covariates: list[str]) -> tuple[bool, str]:
    """Light schema check on an elicited prior. Returns (ok, reason)."""
    if not isinstance(spec, dict):
        return False, "not a dict"
    if "coefficients" not in spec or "variogram" not in spec:
        return False, "missing coefficients/variogram"
    for c in covariates:
        e = spec["coefficients"].get(c)
        if not e or "mean" not in e or "sd" not in e:
            return False, f"coef {c} incomplete"
    for k in ("range", "sill", "nugget"):
        e = spec["variogram"].get(k)
        if not e or "mean" not in e or "sd" not in e:
            return False, f"variogram {k} incomplete"
    return True, "ok"
