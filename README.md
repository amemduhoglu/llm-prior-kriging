# Trend and covariance priors for sparse-network Bayesian kriging

Methodology code for a study of what the prior is worth in Bayesian regression kriging
when the network is sparse, and of whether a language model can supply one. Data,
likelihood and sampler are held fixed and only the prior block changes, which is what
isolates the prior's contribution.

Seven prior conditions are supported. Four take elicited values: for the trend
coefficients, for the covariance hyperparameters (range, sill, nugget), for both, and
for the trend with a principled covariance default. Three contain no language model:
the vague reference, a penalized-complexity range prior, and a zero-centred trend prior
at unit-information width, the control that separates shrinkage towards zero from what
an elicited prior states.

A large language model supplies the elicited values from a task description alone:
region, target and covariate names with units, never an observed value or a summary of
one. A guard enforces that, and every raw response is logged.

## Repository layout

```
src/
  config.py            config.yaml loader
  data.py              dataset loading + density subsampling
  elicit.py            LLM -> priors (strict JSON), with the leakage guard
  priors.py            JSON priors -> PyMC distributions
  models.py            ordinary / regression / Bayesian kriging + random forest
  cv.py                spatial blocked cross-validation
  metrics.py           RMSE / MAE / CRPS / PIT / interval coverage
  run.py               experiment driver (reads config.yaml)
  sim.py               known-truth range-misspecification simulation
  theory.py            prior weight and the help condition for a Normal trend prior
  theory_check.py      applies the help condition to the networks' training folds
prompts/               the two elicitation templates, exactly as sent (v2 point estimates,
                       v3 quantiles after a physical decomposition of the range)
config.yaml            single source of truth: datasets, density levels, seeds,
                       prior conditions, model tiers
requirements.txt       Python dependencies
```

This repository is the methodology pipeline only: data retrieval, elicitation, the
prior-conditioned Bayesian model, spatial cross-validation, metrics, and the known-truth
simulation, and the closed-form check of the trend prior. The `eval` stage writes `results/summary.csv` and basic metric-vs-density
plots; the figures and tables in the paper are produced separately and are not included.

## Requirements

- Python 3.12 (any 3.10+ works), the packages in `requirements.txt`.
- [Ollama](https://ollama.com) running locally for the local open-weight tiers; the model
  tags in `config.yaml` must be pulled first (`ollama pull qwen3.5:9b`).
- `OPENROUTER_API_KEY` in the environment for the API-served tiers
  (`frontier_openrouter`, `frontier_openweight`, `precision_control`). No key is needed
  for the local tiers or for any stage that does not elicit.

## Data

No data files are shipped. Each loader in `data.py` downloads and caches its source once
into `data/raw/`:

- Meuse pilot: `meuse.rda` from the CRAN `sp` package.
- Temperature networks: GHCN-M v4 (`ghcnm.tavg.latest.qcu.tar.gz`, NOAA NCEI).
- Precipitation network: GHCN v2 annual totals (NOAA NCEI).

Station selection (bounding box, target year, minimum valid months, station cap) is
defined per dataset in `config.yaml`, so a network is reproducible from source.

## Reproducibility

`config.yaml` is the single source of truth for datasets, density levels, seeds, prior
conditions, and the model tiers. All seeds are fixed there; every run is reproducible
from the configuration file.

**Leakage guard.** Elicitation prompts are built only from the configuration's text
fields (variable description, units, region, covariate names) plus a fixed template,
never from data arrays. As defence-in-depth, `elicit.py` rejects the vocabulary of data
summary statistics in the config text. Observed values and their summaries never enter
the prompt.

## Running the pipeline

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. classical + ML baselines
python src/run.py    --config config.yaml --stage baselines --dataset main

# 2. elicit priors from a model tier (writes priors JSON + raw log)
python src/elicit.py --config config.yaml --tier local_small --dataset main

# 3. fit the Bayesian model under each prior condition (model held identical)
python src/run.py    --config config.yaml --stage bayes --dataset main \
                     --priors vague,llm_coef,llm_variogram,llm_both,pc_range,hybrid,shrink_zero \
                     --models local_small

# 4. evaluate: metrics vs. density for all conditions (writes results/summary.csv)
python src/run.py    --config config.yaml --stage eval
```

`--dataset` selects a network (`pilot`, `main`, `andes`, `prcp`, `urban`) and `--models` an
elicitation tier from `config.yaml`. Two elicitation protocols are available via
`--protocol` (`v2`, point estimate plus standard deviation; `v3`, p5/p50/p95 quantiles).
The prior-width sweep inflates the elicited standard deviations, leaving the vague and
PC-prior conditions untouched:

```bash
python src/run.py --config config.yaml --stage bayes --dataset main \
                  --priors llm_both --models frontier_openrouter --width-scales 1,2,3
```

The known-truth range-misspecification simulation runs on its own:

```bash
python src/sim.py --config config.yaml
```

The closed-form trend-prior check reads the elicited coefficient priors and, once the
`bayes` and `eval` stages have run, compares its predicted RMSE ratios with the observed ones:

```bash
python src/theory_check.py --config config.yaml
```

Stages are independent and checkpointed per cell, so an interrupted run resumes where it
stopped. Available dataset names, model tiers, prior conditions, and density levels are
all defined in `config.yaml`.

## Outputs

Everything is written under `results/` (set by `project.output_dir`):

- `elicit/<tier>/<model>/`: `phrasing_<k>.json` (raw model text plus the parsed spec and
  status, one per phrasing) and `consensus.json`, the pooled prior the `bayes` stage reads.
- `cells/`: one JSON per fitted cell (dataset, density, seed, fold, condition, model),
  written the instant it finishes and skipped on a re-run.
- `cells_long.csv` and `summary.csv`: the per-cell and aggregated metric-versus-density
  tables written by the `eval` stage (RMSE, MAE, CRPS, PIT, 90% coverage, interval width).
- `sim/cells/`: the known-truth range-misspecification results from `sim.py`, with the
  held-out metrics and posterior quantiles of the range, sill, nugget and microergodic
  parameter for each fit.

## Citation

Memduhoğlu, A., Duman, H. Trend and covariance priors for sparse-network Bayesian kriging
from language-model elicitation. Under review.

## License

Released under the MIT License (see `LICENSE`).
