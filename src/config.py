"""Load and validate config.yaml — the single source of truth."""
from __future__ import annotations
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


class Config:
    def __init__(self, d: dict, path: Path):
        self._d = d
        self.path = path
        self.root = ROOT

    def __getitem__(self, k):
        return self._d[k]

    def get(self, k, default=None):
        return self._d.get(k, default)

    # convenience accessors
    @property
    def seeds(self):
        return list(self._d["project"]["seeds"])

    @property
    def output_dir(self) -> Path:
        p = ROOT / self._d["project"].get("output_dir", "results/")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def pilot(self):
        return self._d["datasets"]["pilot"]

    def dataset(self, which: str):
        return self._d["datasets"][which]

    @property
    def densities(self):
        return list(self._d["experiment"]["density_levels"])

    @property
    def cv(self):
        return self._d["experiment"]["spatial_cv"]

    @property
    def mcmc(self):
        return self._d["experiment"]["mcmc"]

    @property
    def metrics(self):
        return list(self._d["metrics"])

    @property
    def elicitation(self):
        return self._d["elicitation"]


def load(path: str | os.PathLike = "config.yaml") -> Config:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as f:
        d = yaml.safe_load(f)
    _validate(d)
    return Config(d, p)


def _validate(d: dict):
    assert "project" in d and "seeds" in d["project"], "config: project.seeds missing"
    assert "datasets" in d and "pilot" in d["datasets"], "config: datasets.pilot missing"
    pilot = d["datasets"]["pilot"]
    for k in ("name", "target", "covariates", "crs"):
        assert k in pilot, f"config: pilot.{k} missing"
    assert d["experiment"]["spatial_cv"]["kind"] != "random", "RANDOM CV FORBIDDEN (autocorr leak)"
    assert "elicitation" in d, "config: elicitation missing"
