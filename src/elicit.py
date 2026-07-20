"""Elicit Bayesian priors from an LLM using ONLY the task description, covariate
names/units, region, and target units. NEVER observed values or their summaries (leakage).

Output per model:
  results/elicit/<safe_tag>/phrasing_<k>.json   raw text + parsed spec + status
  results/elicit/<safe_tag>/consensus.json       aggregated spec used by the bayes stage
Every prompt is versioned (prompts/elicit_v1.txt). Raw responses are always logged.
"""
from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfgmod
from priors import validate_spec

# Two structurally different elicitation protocols (robustness against "bad prompt"):
#   v2 — point estimate + sd per parameter (default; all existing results)
#   v3 — p5/p50/p95 quantiles + forced physical-process decomposition before numbers
PROTOCOLS = ("v2", "v3")
DEFAULT_PROTOCOL = "v2"


def _prompt_file(protocol: str) -> Path:
    return cfgmod.ROOT / "prompts" / f"elicit_{protocol}.txt"

# 5 phrasing wrappers for the H4 robustness probe — wording only, same information.
PHRASINGS = [
    "Answer carefully and concisely.",
    "Think like a senior soil-geochemistry statistician setting a Bayesian prior.",
    "Be conservative: when unsure, widen the prior sd rather than guess a tight value.",
    "Give your honest best-guess priors; brief rationale each.",
    "Provide priors suitable for a sparse-data setting where the prior will matter.",
]

# Leakage guard. The prompt is built ONLY from config text fields (description, units,
# region, covariate names) + the fixed template — never from data arrays — so observed
# values cannot enter structurally. As defence-in-depth we forbid the vocabulary of
# data SUMMARY STATISTICS (the thing the leakage protocol explicitly bans), which would be the only
# way a human could smuggle data into the config text.
_LEAK_PATTERNS = re.compile(
    r"\b(mean|average|median|std|standard deviation|variance|correlat|covarianc|"
    r"minimum|maximum|quantile|percentile|histogram|sample of|observed value|"
    r"measured value|n\s*=\s*\d|r\s*=\s*[-\d])\b", re.IGNORECASE)
# "annual mean" in the temperature description is a variable definition, not a data summary.
_LEAK_ALLOW = ("annual mean", "annual-mean")


def _safe(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", tag)


def build_prompt(cfg, phrasing: str, which: str = "pilot", protocol: str = DEFAULT_PROTOCOL) -> str:
    p = cfg.dataset(which)
    tmpl = _prompt_file(protocol).read_text()
    tgt = p["target"]
    if p.get("log_transform", False):
        scale_note = (f"IMPORTANT: priors are for the NATURAL LOGARITHM of {tgt}. "
                      f"Give coefficients as the change in log({tgt}) per one natural unit "
                      f"of the covariate, and sill/nugget as variances on the log scale.")
    else:
        scale_note = (f"IMPORTANT: priors are for {tgt} on its natural scale "
                      f"({p.get('units','')}). Give coefficients as the change in {tgt} "
                      f"(in those units) per one natural unit of the covariate, and "
                      f"sill/nugget as variances on that natural scale.")
    extent = p.get("approx_extent_km")
    dist_note = ("Coordinates and the range/lengthscale are in METRES."
                 + (f" The study region spans roughly {extent} km across, so reason about "
                    f"the range in metres accordingly." if extent else ""))
    fields = dict(
        target=tgt,
        target_units=p.get("units", "unspecified"),
        region=p.get("region", "unspecified"),
        description=" ".join(str(p.get("description", "")).split()),
        covariate_block="\n".join(f"  - {c}" for c in p["covariates"]),
        target_scale_note=scale_note,
        distance_note=dist_note,
    )
    # Guard ONLY the human-authored config text (description/units/region/covariate names) —
    # the one place observed data or its summary statistics could be smuggled in. The fixed
    # template and the generated scale/distance notes legitimately use words like "variance".
    _assert_no_leakage("\n".join([fields["description"], fields["target_units"],
                                  fields["region"], fields["covariate_block"]]))
    return tmpl.format(**fields) + "\n\nGUIDANCE: " + phrasing


def _assert_no_leakage(prompt: str):
    scrub = prompt.lower()
    for a in _LEAK_ALLOW:
        scrub = scrub.replace(a, "")
    hit = _LEAK_PATTERNS.search(scrub)
    if hit:
        raise AssertionError(f"LEAKAGE GUARD: summary-stat term '{hit.group(0)}' in prompt")


def _extract_json(text: str) -> dict | None:
    """Best-effort: strip fences, grab the outermost {...}, parse; light repair."""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*", "", t).strip().strip("`").strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = t[start:end + 1]
    for attempt in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):  # drop trailing commas
        try:
            return json.loads(attempt)
        except Exception:
            continue
    return None


def _normalize_spec(parsed: dict, protocol: str) -> dict | None:
    """Bring any protocol's output to the internal {mean, sd, sign, ...} schema.

    v3 elicits p5/p50/p95. Coefficients (real-valued) map to a Normal: mean = p50,
    sd = (p95-p5)/3.29. Variogram params (positive) are treated as LogNormal in quantile
    space — sigma_log = (ln p95 - ln p5)/3.29, mu_log = ln p50 — then converted to the
    moment (mean, sd) parameterization that priors.py expects.
    """
    if protocol == "v2" or parsed is None:
        return parsed
    try:
        out = {"coefficients": {}, "variogram": {}}
        for c, e in parsed.get("coefficients", {}).items():
            p5, p50, p95 = float(e["p5"]), float(e["p50"]), float(e["p95"])
            out["coefficients"][c] = {"mean": p50, "sd": max((p95 - p5) / 3.29, 1e-9),
                                      "sign": e.get("sign", "unknown"),
                                      "confidence": e.get("confidence", ""),
                                      "quantiles": [p5, p50, p95]}
        for k, e in parsed.get("variogram", {}).items():
            p5, p50, p95 = float(e["p5"]), float(e["p50"]), float(e["p95"])
            if p5 <= 0 or p50 <= 0 or p95 <= 0 or p95 < p5:
                return None
            s = max((np.log(p95) - np.log(p5)) / 3.29, 1e-6)
            mu = np.log(p50)
            mean = float(np.exp(mu + s * s / 2))
            sd = float(mean * np.sqrt(np.expm1(s * s)))
            out["variogram"][k] = {"mean": mean, "sd": sd, "sign": "positive",
                                   "confidence": e.get("confidence", ""),
                                   "quantiles": [p5, p50, p95]}
        return out
    except (KeyError, TypeError, ValueError):
        return None


def _call_openrouter(model_id: str, prompt: str, temperature: float, timeout_s: int,
                     provider_pref: dict | None = None) -> str:
    """Elicitation via OpenRouter (API-served models). Reasoning effort low + capped
    max_tokens keep the per-call cost roughly flat; the call falls back without the
    response_format/reasoning fields for providers that reject them. provider_pref is an
    optional OpenRouter routing block (e.g. to pin the served quantization)."""
    import requests
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    # headroom for reasoning models that spend output tokens on hidden CoT before the JSON
    base = dict(model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=8000)
    if provider_pref:
        base["provider"] = provider_pref
    rich = dict(base, response_format={"type": "json_object"},
                reasoning={"effort": "low"})
    for payload in (rich, base):
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                          headers={"Authorization": f"Bearer {key}"},
                          json=payload, timeout=timeout_s)
        if r.status_code == 400:
            continue  # provider rejected an optional field — retry bare
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    r.raise_for_status()
    return ""


def _call_ollama(model_tag: str, prompt: str, temperature: float, timeout_s: int,
                 provider_pref: dict | None = None) -> str:  # provider_pref ignored (local)
    import ollama
    client = ollama.Client(timeout=timeout_s)
    # think=False: Qwen3/reasoning models otherwise burn time on hidden CoT and (with
    # format=json) return empty content. Disabling it yields fast, parseable JSON.
    try:
        resp = client.chat(
            model=model_tag, messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}, format="json", think=False,
        )
    except Exception:
        # models that don't accept the think flag
        resp = client.chat(
            model=model_tag, messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}, format="json",
        )
    return resp.message.content or ""


def consensus_dirname(tag: str, protocol: str = DEFAULT_PROTOCOL) -> str:
    """Directory / identity for a (model, protocol) pair. v2 keeps the historical name so
    all existing results stay valid; other protocols get a __<protocol> suffix."""
    return _safe(tag if protocol == DEFAULT_PROTOCOL else f"{tag}__{protocol}")


def elicit_model(cfg, entry, which: str = "pilot", protocol: str = DEFAULT_PROTOCOL) -> dict:
    """Run all phrasings for one model entry (dict with tag/provider, or bare tag string),
    write artifacts, return the consensus spec (or None)."""
    if isinstance(entry, str):
        entry = {"tag": entry}
    model_tag = entry["tag"]
    provider = entry.get("provider", "ollama")
    el = cfg.elicitation
    covs = cfg.dataset(which)["covariates"]
    outdir = cfg.output_dir / "elicit" / which / consensus_dirname(model_tag, protocol)
    outdir.mkdir(parents=True, exist_ok=True)

    # frontier models use fewer phrasings: phrasing robustness is already established on
    # the 9-model local zoo; the frontier tier only needs a spread check (cost control)
    if provider == "ollama":
        n_phr = int(el.get("phrasings", 5))
    else:
        n_phr = int(el.get("phrasings_frontier", 3))
    temp = float(el.get("temperature", 0.2))
    timeout_s = int(el.get("request_timeout_s", 180))
    retries = int(el.get("max_retries", 2))
    if provider != "ollama":
        retries = min(retries, 1)   # paid calls: 2 attempts max bounds worst-case spend
    call = _call_ollama if provider == "ollama" else _call_openrouter
    # optional quantization pin (entry carries `quantization: bf16`): route only to providers
    # serving that quant, so the same-weights precision control stays valid
    qz = entry.get("quantization")
    provider_pref = {"quantizations": [qz], "allow_fallbacks": True} if qz else None

    # Idempotent caching: a phrasing that already succeeded (status "ok") is reused
    # instead of re-calling the model/API. This makes elicitation safe to restart at any
    # time WITHOUT re-spending OpenRouter money or re-running slow local models. Failed
    # phrasings (status != "ok") are NOT cached, so they get retried. ELICIT_FORCE=1
    # overrides the cache for an intentional fresh re-elicitation.
    force = os.environ.get("ELICIT_FORCE", "") == "1"
    valid_specs = []
    for k in range(n_phr):
        phrasing = PHRASINGS[k % len(PHRASINGS)]
        cache_f = outdir / f"phrasing_{k}.json"
        if not force and cache_f.exists():
            try:
                prev = json.loads(cache_f.read_text())
                if prev.get("status") == "ok" and prev.get("parsed") is not None:
                    valid_specs.append(prev["parsed"])
                    print(f"  [{model_tag}/{protocol}] phrasing {k}: cached-ok (reused, no call)",
                          flush=True)
                    continue
            except Exception:
                pass  # corrupt/partial cache -> fall through and re-elicit
        prompt = build_prompt(cfg, phrasing, which, protocol)
        raw, parsed, status = "", None, "fail"
        for attempt in range(retries + 1):
            try:
                raw = call(model_tag, prompt, temp, timeout_s, provider_pref=provider_pref)
                parsed = _normalize_spec(_extract_json(raw), protocol)
                if parsed is not None:
                    ok, reason = validate_spec(parsed, covs)
                    status = "ok" if ok else f"invalid:{reason}"
                    if ok:
                        valid_specs.append(parsed)
                        break
                else:
                    status = "unparseable"
            except Exception as e:  # missing model, timeout, server down, no API key
                status = f"error:{type(e).__name__}:{e}"
            time.sleep(1.5 * (attempt + 1))
        (outdir / f"phrasing_{k}.json").write_text(json.dumps({
            "model": model_tag, "provider": provider, "prompt_version": protocol,
            "phrasing": phrasing, "status": status, "raw": raw, "parsed": parsed,
        }, indent=2))
        print(f"  [{model_tag}/{protocol}] phrasing {k}: {status}", flush=True)

    consensus = _aggregate(valid_specs, covs) if valid_specs else None
    (outdir / "consensus.json").write_text(json.dumps({
        "model": model_tag, "provider": provider, "prompt_version": protocol,
        "n_valid": len(valid_specs), "n_phrasings": n_phr, "consensus": consensus,
    }, indent=2))
    print(f"  [{model_tag}/{protocol}] consensus from {len(valid_specs)}/{n_phr} valid phrasings", flush=True)
    return consensus


def _agg_param(entries: list[dict]) -> dict:
    means = np.array([float(e.get("mean", 0.0)) for e in entries])
    sds = np.array([max(float(e.get("sd", 1.0)), 1e-6) for e in entries])
    signs = [e.get("sign", "unknown") for e in entries]
    # combined sd = within-phrasing + between-phrasing spread
    comb_sd = float(np.sqrt(np.mean(sds ** 2) + np.var(means)))
    maj = max(set(signs), key=signs.count) if signs else "unknown"
    return {"mean": float(np.mean(means)), "sd": comb_sd, "sign": maj,
            "n": len(entries)}


def _aggregate(specs: list[dict], covs: list[str]) -> dict:
    out = {"coefficients": {}, "variogram": {}}
    for c in covs:
        ent = [s["coefficients"][c] for s in specs if c in s.get("coefficients", {})]
        if ent:
            out["coefficients"][c] = _agg_param(ent)
    for k in ("range", "sill", "nugget"):
        ent = [s["variogram"][k] for s in specs if k in s.get("variogram", {})]
        if ent:
            out["variogram"][k] = _agg_param(ent)
    return out


def run_tier(cfg, tier: str, which: str = "pilot", protocol: str = DEFAULT_PROTOCOL):
    el = cfg.elicitation
    models = el.get(tier, [])
    if not models:
        print(f"[elicit] tier '{tier}' empty/missing — skip", flush=True)
        return
    for m in models:
        print(f"[elicit] {which}/{tier}/{protocol}: {m['tag']}", flush=True)
        try:
            elicit_model(cfg, m, which, protocol)
        except Exception as e:
            print(f"[elicit] {m['tag']} FAILED hard: {e} — continuing", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tier", default="local_small")
    ap.add_argument("--dataset", default="pilot",
                    choices=["pilot", "main", "andes", "prcp", "urban"])
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL, choices=list(PROTOCOLS))
    a = ap.parse_args()
    cfg = cfgmod.load(a.config)
    run_tier(cfg, a.tier, a.dataset, a.protocol)


if __name__ == "__main__":
    main()
