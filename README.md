# LLM-elicited priors for geostatistical prediction

Methodology code for the study on whether priors elicited from large language models
(LLMs) improve Bayesian geostatistical prediction in data-sparse regions, compared
with the same model under vague priors and with classical kriging.

The LLM supplies, from a task description alone (no observed data), prior distributions
for (i) covariate coefficients and (ii) variogram / Gaussian-process hyperparameters
(range, sill, nugget). The Bayesian model is held identical across prior conditions;
only the priors change, which isolates the contribution of the elicited priors.

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
prompts/               versioned elicitation templates (elicit_v1..v3)
config.yaml            single source of truth: datasets, density levels, seeds,
                       prior conditions, model tiers
requirements.txt       Python dependencies
```

This repository is the methodology pipeline only: data retrieval, elicitation, the
prior-conditioned Bayesian model, spatial cross-validation, metrics, and the known-truth
simulation. The `eval` stage writes `results/summary.csv` and basic metric-vs-density
plots; the figures and tables in the paper are produced separately and are not included.

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
                     --priors vague,llm_coef,llm_variogram,llm_both,pc_range,hybrid \
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

Stages are independent and checkpointed per cell, so an interrupted run resumes where it
stopped. Available dataset names, model tiers, prior conditions, and density levels are
all defined in `config.yaml`.

## License

Released under the MIT License (see `LICENSE`).
