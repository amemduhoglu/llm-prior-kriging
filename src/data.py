"""Load datasets. NO observed values ever leak to elicitation (see elicit.py).

Loaders (selected per dataset via config `name`):
  meuse     : small, dense floodplain pilot / negative control
  ghcnm     : GHCN-M v4 annual-mean temperature stations (the temperature networks)
  ghcn_prcp : GHCN v2 annual precipitation totals (the weak-covariate network)
Retrieval is included: each loader downloads and caches its source once. All return the
SAME bundle shape so the rest of the pipeline is dataset-agnostic.
"""
from __future__ import annotations
import glob
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from config import ROOT

RAW = ROOT / "data" / "raw"
MEUSE_CSV = RAW / "meuse.csv"
MEUSE_RDA_URL = "https://raw.githubusercontent.com/cran/sp/master/data/meuse.rda"
GHCNM_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/v4/ghcnm.tavg.latest.qcu.tar.gz"
GHCN_PRCP_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/v2/v2.prcp.gz"
GHCN_PRCP_INV_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/v2/v2.prcp.inv"


# ----------------------------------------------------------------- shared helper

def _bundle(coords, X_raw, y, covs, frame):
    Xmu = X_raw.mean(0)
    Xsd = X_raw.std(0)
    Xsd[Xsd == 0] = 1.0
    X = (X_raw - Xmu) / Xsd
    return {"coords": coords, "X": X, "y": y, "covariates": covs,
            "X_mean": Xmu, "X_sd": Xsd, "n": len(y), "frame": frame}


def _project_aeqd(lon, lat):
    """Project lon/lat to metres with a region-centred Azimuthal Equidistant projection so
    kriging distances are honest over a large mountainous extent."""
    from pyproj import Transformer
    lon0, lat0 = float(np.mean(lon)), float(np.mean(lat))
    tr = Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs",
        always_xy=True)
    x, y = tr.transform(lon, lat)
    return np.column_stack([x, y])


# ----------------------------------------------------------------- meuse (pilot)

def _ensure_meuse() -> Path:
    if MEUSE_CSV.exists():
        return MEUSE_CSV
    RAW.mkdir(parents=True, exist_ok=True)
    rda = RAW / "meuse.rda"
    if not rda.exists():
        urllib.request.urlretrieve(MEUSE_RDA_URL, rda)
    import pyreadr
    res = pyreadr.read_r(str(rda))
    res[list(res.keys())[0]].to_csv(MEUSE_CSV, index=False)
    return MEUSE_CSV


def _load_meuse(p) -> dict:
    df = pd.read_csv(_ensure_meuse())
    coords = df[["x", "y"]].to_numpy(float)
    covs = list(p["covariates"])
    X_raw = df[covs].to_numpy(float)
    y = df[p["target"]].to_numpy(float)
    if p.get("log_transform", False):
        y = np.log(y)
    return _bundle(coords, X_raw, y, covs, df)


# ----------------------------------------------------------------- ghcnm (main)

def _ensure_ghcnm() -> Path:
    """Return the extracted GHCN-M v4 directory, downloading+extracting once."""
    existing = sorted(glob.glob(str(RAW / "ghcnm.v4*")))
    dirs = [d for d in existing if Path(d).is_dir()]
    if dirs:
        return Path(dirs[-1])
    RAW.mkdir(parents=True, exist_ok=True)
    tgz = RAW / "ghcnm.tavg.tar.gz"
    if not tgz.exists():
        urllib.request.urlretrieve(GHCNM_URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(RAW)
    dirs = [d for d in sorted(glob.glob(str(RAW / "ghcnm.v4*"))) if Path(d).is_dir()]
    return Path(dirs[-1])


def _parse_ghcnm_inv(inv_path: Path) -> pd.DataFrame:
    rows = []
    for ln in open(inv_path):
        try:
            rows.append((ln[0:11], float(ln[12:20]), float(ln[21:30]), float(ln[31:37])))
        except ValueError:
            continue
    df = pd.DataFrame(rows, columns=["id", "lat", "lon", "elev"]).set_index("id")
    return df[(df.elev > -500) & (df.elev < 8849)]  # drop 9999 missing-elevation sentinel


def _parse_ghcnm_year(dat_path: Path, ids: set, year: int, min_months: int) -> dict:
    """Annual-mean temperature (degC) per station for one year, missing=-9999."""
    out = {}
    for ln in open(dat_path):
        sid = ln[0:11]
        if sid not in ids or int(ln[11:15]) != year:
            continue
        vals = []
        for m in range(12):
            v = ln[19 + m * 8:19 + m * 8 + 5].strip()
            if v and v != "-9999":
                vals.append(int(v) / 100.0)
        if len(vals) >= min_months:
            out[sid] = float(np.mean(vals))
    return out


def _load_ghcnm(p) -> dict:
    d = _ensure_ghcnm()
    inv = _parse_ghcnm_inv(Path(glob.glob(str(d / "*.inv"))[0]))
    bb = p["bbox"]
    reg = inv[inv.lat.between(bb[0], bb[1]) & inv.lon.between(bb[2], bb[3])]
    temps = _parse_ghcnm_year(Path(glob.glob(str(d / "*.dat"))[0]),
                              set(reg.index), int(p["year"]),
                              int(p.get("min_valid_months", 10)))
    ids = [i for i in reg.index if i in temps]
    if len(ids) < 10:
        raise RuntimeError(f"ghcnm: only {len(ids)} stations — widen bbox or change year")
    cap = int(p.get("max_stations", 0))
    if cap and len(ids) > cap:
        # deterministic spatial subsample: thins a dense network to sparse-monitoring
        # conditions AND keeps the GP tractable (same scheme as the precip loader).
        rng = np.random.default_rng(int(p.get("subsample_seed", 7)))
        ids = sorted(rng.choice(ids, size=cap, replace=False))
    sub = reg.loc[ids]
    y = np.array([temps[i] for i in ids], float)
    coords = _project_aeqd(sub.lon.to_numpy(float), sub.lat.to_numpy(float))
    X_raw = sub[["elev"]].to_numpy(float)
    frame = sub.assign(tavg=y).reset_index()
    return _bundle(coords, X_raw, y, list(p["covariates"]), frame)


# ----------------------------------------------------------------- ghcn v2 precip (weak covariate)

def _ensure_ghcn_prcp() -> tuple[Path, Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    dat = RAW / "v2.prcp.gz"
    inv = RAW / "v2.prcp.inv"
    if not dat.exists():
        urllib.request.urlretrieve(GHCN_PRCP_URL, dat)
    if not inv.exists():
        urllib.request.urlretrieve(GHCN_PRCP_INV_URL, inv)
    return dat, inv


def _parse_prcp_inv(inv_path: Path) -> pd.DataFrame:
    """GHCN v2 precip inventory: id(11) name lat lon elev — fixed-width, same layout family
    as v2 temperature inventories. Parsed defensively; sanity-checked at load."""
    rows = []
    for ln in open(inv_path, encoding="latin-1"):
        try:
            sid = ln[0:11].strip()
            lat = float(ln[43:49])
            lon = float(ln[49:57])
            elev = float(ln[57:62])
        except (ValueError, IndexError):
            continue
        rows.append((sid, lat, lon, elev))
    df = pd.DataFrame(rows, columns=["id", "lat", "lon", "elev"]).set_index("id")
    df = df[(df.lat.between(-90, 90)) & (df.lon.between(-180, 180))
            & (df.elev > -500) & (df.elev < 8849)]
    if len(df) < 1000:
        raise RuntimeError(f"v2.prcp.inv parse suspicious: only {len(df)} stations — "
                           f"check the fixed-width column offsets against the file")
    return df


def _load_ghcn_prcp(p) -> dict:
    """Annual total precipitation (mm) per station for one year. v2.prcp records:
    id(11) + duplicate(1) + year(4) + 12 x 5-char monthly totals in TENTHS of mm,
    missing = -9999 (trace = -8888 -> treat as 0)."""
    import gzip
    dat, inv = _ensure_ghcn_prcp()
    inv_df = _parse_prcp_inv(inv)
    bb = p["bbox"]
    reg = inv_df[inv_df.lat.between(bb[0], bb[1]) & inv_df.lon.between(bb[2], bb[3])]
    ids = set(reg.index)
    year = int(p["year"])
    min_months = int(p.get("min_valid_months", 10))
    totals = {}
    with gzip.open(dat, "rt", encoding="latin-1") as f:
        for ln in f:
            sid = ln[0:11]
            if sid not in ids:
                continue
            try:
                if int(ln[12:16]) != year:
                    continue
            except ValueError:
                continue
            vals = []
            for m in range(12):
                v = ln[16 + m * 5:16 + m * 5 + 5].strip()
                if not v or v == "-9999":
                    continue
                x = int(v)
                vals.append(0.0 if x == -8888 else x / 10.0)
            if len(vals) >= min_months:
                # scale to a full-year total so 10- and 12-month stations are comparable
                totals[sid] = float(np.sum(vals) * 12.0 / len(vals))
    keep = [i for i in reg.index if i in totals and totals[i] > 0]
    if len(keep) < 10:
        raise RuntimeError(f"ghcn_prcp: only {len(keep)} stations for {year} in bbox — "
                           f"try another year or check the .prcp record layout")
    cap = int(p.get("max_stations", 0))
    if cap and len(keep) > cap:
        # deterministic spatial subsample: keeps the network sparse AND the GP tractable
        rng = np.random.default_rng(int(p.get("subsample_seed", 7)))
        keep = sorted(rng.choice(keep, size=cap, replace=False))
    sub = reg.loc[keep]
    y = np.array([totals[i] for i in keep], float)
    if p.get("log_transform", False):
        y = np.log(y)
    coords = _project_aeqd(sub.lon.to_numpy(float), sub.lat.to_numpy(float))
    X_raw = sub[["elev"]].to_numpy(float)
    frame = sub.assign(prcp=np.array([totals[i] for i in keep])).reset_index()
    return _bundle(coords, X_raw, y, list(p["covariates"]), frame)


# ----------------------------------------------------------------- dispatch

_LOADERS = {"meuse": _load_meuse, "ghcnm": _load_ghcnm, "ghcn_prcp": _load_ghcn_prcp}


def load_dataset(cfg, which: str = "pilot") -> dict:
    p = cfg.dataset(which)
    name = p["name"]
    if name not in _LOADERS:
        raise NotImplementedError(f"dataset loader for '{name}' not wired")
    return _LOADERS[name](p)


def load_pilot(cfg) -> dict:  # backwards-compatible
    return load_dataset(cfg, "pilot")
