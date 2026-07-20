"""Predictive metrics: point accuracy + probabilistic calibration. All operate on a
predictive Gaussian (mean, sd) per held-out point, plus the truth."""
from __future__ import annotations
import numpy as np
from scipy import stats


def rmse(y, mu):
    y, mu = np.asarray(y), np.asarray(mu)
    return float(np.sqrt(np.mean((y - mu) ** 2)))


def mae(y, mu):
    y, mu = np.asarray(y), np.asarray(mu)
    return float(np.mean(np.abs(y - mu)))


def crps_gaussian(y, mu, sd):
    """Closed-form CRPS for a Gaussian forecast (Gneiting & Raftery 2007)."""
    y, mu, sd = np.asarray(y), np.asarray(mu), np.asarray(sd)
    sd = np.clip(sd, 1e-9, None)
    z = (y - mu) / sd
    return float(np.mean(sd * (z * (2 * stats.norm.cdf(z) - 1)
                               + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))))


def pit_values(y, mu, sd):
    """Probability Integral Transform values; should be ~Uniform(0,1) if calibrated."""
    y, mu, sd = np.asarray(y), np.asarray(mu), np.asarray(sd)
    sd = np.clip(sd, 1e-9, None)
    return stats.norm.cdf((y - mu) / sd)


def coverage90(y, mu, sd):
    """Empirical coverage of the central 90% predictive interval."""
    y, mu, sd = np.asarray(y), np.asarray(mu), np.asarray(sd)
    sd = np.clip(sd, 1e-9, None)
    z = stats.norm.ppf(0.95)
    lo, hi = mu - z * sd, mu + z * sd
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_width(mu, sd):
    sd = np.clip(np.asarray(sd), 1e-9, None)
    z = stats.norm.ppf(0.95)
    return float(np.mean(2 * z * sd))


def all_metrics(y, mu, sd):
    """Return the full metric dict for one fold's held-out predictions.
    PIT is summarized by its KS distance to Uniform(0,1) (0 = perfectly calibrated)."""
    pit = pit_values(y, mu, sd)
    ks = float(stats.kstest(pit, "uniform").statistic) if len(pit) > 1 else np.nan
    return {
        "rmse": rmse(y, mu),
        "mae": mae(y, mu),
        "crps": crps_gaussian(y, mu, sd),
        "pit_ks": ks,
        "coverage90": coverage90(y, mu, sd),
        "interval_width": interval_width(mu, sd),
        "n_test": int(len(y)),
    }
