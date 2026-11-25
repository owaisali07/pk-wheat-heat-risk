# %% [markdown]
# # Explainable GeoAI assessment of wheat heat-stress livelihood risk in Pakistan
#
# Complete analysis notebook for the paper *"Explainable GeoAI Locates Structural Wheat Heat-Stress Risk in
# Pakistan: Integrating Earth Observation, Climate Projections and Socioeconomic Data"*.
#
# The notebook follows the seven stages of the methodology:
#
# | Section | Stage | Content |
# |---|---|---|
# | 1 | - | Setup, district yield target (Theil-Sen detrended anomalies) |
# | 2 | A | Wheat-area-weighted extraction of ERA5-Land, CHIRPS and MODIS to districts; stage-wise predictors |
# | 3 | A | XGBoost, random forest, ridge and trend baseline under three blocked cross-validation designs; Diebold-Mariano tests |
# | 4 | A | SHAP attribution, stage-wise heat thresholds, accumulated local effects |
# | 5 | A | Heat-attributable losses and hazard, present climate and 1.5/2/3 degC warming (35 NEX-GDDP-CMIP6 models) |
# | 6 | B, C | Exposure; PCA-weighted vulnerability index and its validation |
# | 7 | D | Livelihood risk with Monte Carlo uncertainty |
# | 8 | E | Local Moran's I hotspots; concentration index of losses across the wealth distribution |
# | 9 | F | SHAP on a surrogate of the risk index; district policy typology |
# | 10 | G | Out-of-sample test against the 2021-22 (March-April 2022) heatwave season |
# | 11 | - | Figures and tables for the manuscript |
#
# All inputs are open data stored under `data_raw/` (see `data_raw/README.md`). Expensive intermediate
# products (daily district climate series, CMIP6 deltas) are cached in `data_processed/analysis/`; delete
# that folder to recompute everything from the raw files.

# %%
import glob
import json
import os
import re
import time
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.windows import from_bounds
from scipy import sparse, stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(os.environ.get("PAPER8_ROOT", ".."))  # the notebook runs in analysis/
RAW = ROOT / "data_raw"
CACHE = ROOT / "data_processed" / "analysis"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "figures"
for p in (CACHE, TAB, FIG):
    p.mkdir(parents=True, exist_ok=True)

SEED = 20260514
rng_global = np.random.default_rng(SEED)
N_JOBS = max(1, os.cpu_count() - 2)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(N_JOBS))  # avoids a harmless loky core-count warning on Windows
BBOX = (60.0, 23.0, 79.0, 38.0)  # W, S, E, N
STUDY_PROVINCES = ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan", "Islamabad"]
HOLDOUT_SEASON = 2022  # 2021-22 season (harvest year 2022) is withheld from all model fitting

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
    "axes.linewidth": 0.6, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "savefig.dpi": 300, "savefig.bbox": "tight",
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42,
})
print("cores used:", N_JOBS)

# %% [markdown]
# ## 1. Districts and the yield target
#
# The district wheat panel (1981-82 to 2024-25, 2022 boundaries) was compiled from PBS, provincial Crop
# Reporting Services, Bureaus of Statistics and MNFSR publications (`scripts/yield_panel/`). Only
# district-own observations (`flag` = own / own_merged_units) enter the models; values inherited from a
# parent district are kept for a robustness check. Seasons are indexed by harvest year (2021-22 -> 2022).
#
# The technology trend of each district is a Theil-Sen line, re-centred so that the district's mean relative
# anomaly over the fitting seasons is zero (equivalent to a district fixed effect).

# %%
adm = gpd.read_file(RAW / "01_boundaries" / "pak_admin2.shp").to_crs(4326)
adm = adm[["adm2_pcode", "adm2_name", "adm1_name", "geometry"]]
study = adm[adm.adm1_name.isin(STUDY_PROVINCES)].reset_index(drop=True)
PCODES = study.adm2_pcode.tolist()
print("districts in study provinces:", len(study))

panel = pd.read_csv(ROOT / "data_processed" / "yield_panel" / "wheat_yield_district_panel_1982_2025.csv")
panel["own"] = panel.flag.str.startswith("own")
print(panel.flag.value_counts().to_string())

# %%
def theil_sen_trend(years, values, fit_mask):
    """District technology trend: Theil-Sen slope/intercept fitted on fit_mask, evaluated on all years."""
    res = stats.theilslopes(values[fit_mask], years[fit_mask])
    return res.slope, res.intercept


MIN_YEARS = 12  # minimum district-own seasons (outside the hold-out) to fit a trend
rows = []
for pc, g in panel[panel.yield_t_ha.notna()].groupby("adm2_pcode"):
    g = g.sort_values("year")
    own = g.own.values
    fit = own & (g.year.values != HOLDOUT_SEASON)
    if fit.sum() < MIN_YEARS:
        continue
    slope, icpt = theil_sen_trend(g.year.values.astype(float), g.yield_t_ha.values, fit)
    trend = icpt + slope * g.year.values
    # centre the trend so that the mean relative anomaly over the fitting seasons is zero (district fixed
    # effect): otherwise skewed yield distributions leave district-specific offsets that location-proxy
    # predictors (absolute temperatures) can learn
    off = np.mean((g.yield_t_ha.values[fit] - trend[fit]) / trend[fit])
    trend = trend * (1 + off)
    slope = slope * (1 + off)
    rows.append(pd.DataFrame({"adm2_pcode": pc, "year": g.year.values, "yield_t_ha": g.yield_t_ha.values,
                              "trend_t_ha": trend, "own": own, "flag": g.flag.values,
                              "area_ha": g.area_ha.values, "slope_t_ha_yr": slope}))
tgt = pd.concat(rows, ignore_index=True)
tgt["anom"] = (tgt.yield_t_ha - tgt.trend_t_ha) / tgt.trend_t_ha
tgt = tgt[tgt.trend_t_ha > 0.1]

# districts that grow little wheat are noisy; keep those with a median own-season area >= 1,000 ha
med_area = tgt[tgt.own].groupby("adm2_pcode").area_ha.median()
keep = med_area[med_area >= 1000].index
tgt = tgt[tgt.adm2_pcode.isin(keep)].copy()
# implausible anomalies (data errors) are removed: |anomaly| > 75 % of trend
n_out = int((tgt.anom.abs() > 0.75).sum())
tgt = tgt[tgt.anom.abs() <= 0.75].copy()
tgt = tgt.merge(study[["adm2_pcode", "adm2_name", "adm1_name"]], on="adm2_pcode")
print(f"districts with a trend: {tgt.adm2_pcode.nunique()} | own obs: {tgt.own.sum()} | "
      f"inherited obs: {(~tgt.own).sum()} | removed |anom|>0.75: {n_out}")
print(tgt[tgt.own].anom.describe().round(3).to_string())
tgt.to_csv(CACHE / "yield_target.csv", index=False)

# %% [markdown]
# ## 2. Climate and Earth-observation predictors (Stage A)
#
# ### 2.1 Wheat-area weights
# Each district's climate is the average over its wheat area: every 5-arcmin SPAM2010 pixel inside a district
# carries its wheat physical area as weight (plus 0.001 ha so that districts without mapped wheat fall back to
# a uniform average), and is assigned to the climate grid cell that contains it.

# %%
def read_spam(tech):
    with rasterio.open(RAW / "04_crop" / "spam2010" / f"spam2010V2r0_global_A_WHEA_{tech}.tif") as src:
        win = from_bounds(*BBOX, transform=src.transform)
        a = src.read(1, window=win).astype("float64")
        tr = src.window_transform(win)
    a[a < 0] = 0.0
    return a, tr


spam_a, spam_tr = read_spam("A")
spam_i, _ = read_spam("I")
r_idx, c_idx = np.indices(spam_a.shape)
px_lon = spam_tr.c + (c_idx + 0.5) * spam_tr.a
px_lat = spam_tr.f + (r_idx + 0.5) * spam_tr.e
pts = gpd.GeoDataFrame({"wheat": spam_a.ravel(), "wheat_irr": spam_i.ravel()},
                       geometry=gpd.points_from_xy(px_lon.ravel(), px_lat.ravel()), crs=4326)
pts = gpd.sjoin(pts, study[["adm2_pcode", "geometry"]], predicate="within", how="inner")
pts["lon"], pts["lat"] = pts.geometry.x, pts.geometry.y
pts["w"] = pts.wheat + 1e-3
didx = {pc: i for i, pc in enumerate(PCODES)}
pts["d"] = pts.adm2_pcode.map(didx)
print("SPAM pixels inside study districts:", len(pts), "| wheat area (Mha):", round(pts.wheat.sum() / 1e6, 2))

irr = pts.groupby("adm2_pcode")[["wheat", "wheat_irr"]].sum()
irr_share = (irr.wheat_irr / irr.wheat.replace(0, np.nan)).clip(0, 1).fillna(0).rename("irr_share")
spam_wheat_ha = irr.wheat.rename("spam_wheat_ha")


def weight_matrix(lat, lon):
    """Sparse (n_districts x n_cells) matrix mapping a (lat, lon) grid to wheat-weighted district means."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    il = np.abs(pts.lat.values[:, None] - lat[None, :]).argmin(1)
    jl = np.abs(pts.lon.values[:, None] - lon[None, :]).argmin(1)
    cell = il * len(lon) + jl
    W = sparse.csr_matrix((pts.w.values, (pts.d.values, cell)), shape=(len(PCODES), len(lat) * len(lon)))
    return W


def district_mean(W, arr):
    """arr: (time, lat, lon) -> (time, n_districts), NaN-aware weighted mean."""
    X = arr.reshape(arr.shape[0], -1).T  # cells x time
    ok = np.isfinite(X)
    num = W @ np.where(ok, X, 0.0)
    den = W @ ok.astype("float64")
    return (num / np.where(den > 0, den, np.nan)).T


# %% [markdown]
# ### 2.2 Daily district series (cached)
# ERA5-Land daily Tmax/Tmin (0.1 deg), dew point and volumetric soil moisture (0-7 and 7-28 cm), and CHIRPS
# daily rainfall (0.05 deg), 1980-11 to 2025-05.

# %%
Y0, Y1 = 1980, 2025
CLIM = RAW / "02_climate"


def extract_var(name, files, var, conv=lambda a: a):
    out = CACHE / f"daily_{name}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    W, parts = None, []
    for f in files:
        with xr.open_dataset(f, engine="h5netcdf") as ds:
            latn = "lat" if "lat" in ds.coords else "latitude"
            lonn = "lon" if "lon" in ds.coords else "longitude"
            if W is None:
                W = weight_matrix(ds[latn].values, ds[lonn].values)
            a = conv(ds[var].transpose("time", latn, lonn).values.astype("float64"))
            parts.append(pd.DataFrame(district_mean(W, a), index=pd.to_datetime(ds.time.values).normalize(),
                                      columns=PCODES))
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated()]
    df.to_parquet(out)
    return df


t0 = time.time()
K2C = lambda a: a - 273.15 if np.nanmean(a) > 200 else a
tmax = extract_var("tmax", [CLIM / "era5land_tmax" / f"era5land_tmax_{y}.nc" for y in range(Y0, Y1 + 1)], "tmax", K2C)
tmin = extract_var("tmin", [CLIM / "era5land_tmin" / f"era5land_tmin_{y}.nc" for y in range(Y0, Y1 + 1)], "tmin", K2C)
dfiles = sorted(glob.glob(str(CLIM / "era5land_dewpoint_soilmoisture" / "era5land_dew_swvl_PAK_*.nc")))
dew = extract_var("dew", dfiles, "dewpoint_temperature_2m", K2C)
sm1 = extract_var("swvl1", dfiles, "volumetric_soil_water_layer_1")
sm2 = extract_var("swvl2", dfiles, "volumetric_soil_water_layer_2")
prcp = extract_var("prcp", [CLIM / "chirps_v2_daily" / f"chirps-v2.0.{y}.days_p05_PAK.nc" for y in range(1981, Y1 + 1)],
                   "precip", lambda a: np.where(a < 0, np.nan, a))
sm = (sm1 * 7 + sm2 * 21) / 28  # 0-28 cm depth-weighted
print(f"daily series ready ({time.time() - t0:.0f} s):", tmax.shape, prcp.shape, "Tmax range",
      round(np.nanmin(tmax.values), 1), round(np.nanmax(tmax.values), 1))

# %% [markdown]
# ### 2.3 Stage-wise predictors
# Five stages from the provincial sowing calendars (sowing 1-20/25 Nov in Punjab and KP, Nov-Dec in Sindh):
# sowing (Nov), tillering (Dec-Jan), jointing (Feb), anthesis (Mar) and grain filling (Apr). For each stage:
# growing degree days (base 0 degC, capped at 30 degC), frost days (Tmin < 0 degC), mean vapour-pressure deficit,
# rainfall, longest dry spell (< 1 mm/day), mean soil moisture, mean Tmax and killing degree days above
# 30 and 34 degC (Eq. 2).

# %%
STAGES = {"sow": [(11, 1, 11, 30)], "till": [(12, 1, 12, 31), (1, 1, 1, 31)], "joint": [(2, 1, 2, 29)],
          "anth": [(3, 1, 3, 31)], "fill": [(4, 1, 4, 30)]}
STAGE_ORDER = list(STAGES)


def stage_labels(idx):
    m, d, y = idx.month.values, idx.day.values, idx.year.values
    season = np.where(m >= 11, y + 1, y)
    lab = np.full(len(idx), "", dtype=object)
    for s, wins in STAGES.items():
        for (m0, d0, m1, d1) in wins:
            sel = (m == m0) & (d >= d0) & (d <= d1) if m0 == m1 else np.zeros(len(idx), bool)
            lab[sel] = s
    return season, lab


def es(t):  # saturation vapour pressure (kPa), Tetens
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def longest_run(mask2d):
    """Longest run of True along axis 0 for each column."""
    best = np.zeros(mask2d.shape[1], int)
    cur = np.zeros(mask2d.shape[1], int)
    for row in mask2d:
        cur = np.where(row, cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def stage_features(tx, tn, td, pr, smx, seasons):
    """Return long table (adm2_pcode, year) x stage features from daily district frames."""
    idx = tx.index
    season, lab = stage_labels(idx)
    tmean = (tx + tn) / 2
    vpd = (es(tx) + es(tn)) / 2 - es(td)
    gdd = tmean.clip(lower=0, upper=30)
    out = []
    for yr in seasons:
        rec = {"year": np.full(len(PCODES), yr), "adm2_pcode": PCODES}
        for s in STAGE_ORDER:
            sel = (season == yr) & (lab == s)
            if sel.sum() == 0:
                continue
            X, N = tx.values[sel], tn.values[sel]
            rec[f"gdd_{s}"] = gdd.values[sel].sum(0)
            rec[f"frost_{s}"] = (N < 0).sum(0)
            rec[f"vpd_{s}"] = np.nanmean(vpd.values[sel], 0)
            rec[f"tmax_{s}"] = np.nanmean(X, 0)
            rec[f"kdd30_{s}"] = np.clip(X - 30, 0, None).sum(0)
            rec[f"kdd34_{s}"] = np.clip(X - 34, 0, None).sum(0)
            if pr is not None:
                P = pr.reindex(idx).values[sel]
                rec[f"rain_{s}"] = np.nansum(P, 0)
                rec[f"dry_{s}"] = longest_run(np.nan_to_num(P, nan=0) < 1.0)
            if smx is not None:
                rec[f"sm_{s}"] = np.nanmean(smx.reindex(idx).values[sel], 0)
        out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True)


SEASONS = list(range(1982, 2026))
feat_path = CACHE / "features_obs.parquet"
if feat_path.exists():
    feat = pd.read_parquet(feat_path)
else:
    win = (tmax.index >= "1981-11-01") & (tmax.index <= "2025-04-30")
    feat = stage_features(tmax[win], tmin[win], dew.reindex(tmax.index[win]), prcp, sm, SEASONS)
    feat.to_parquet(feat_path)
feat = feat.merge(irr_share, left_on="adm2_pcode", right_index=True, how="left")
HEAT_VARS = [f"{v}_{s}" for s in STAGE_ORDER for v in ("tmax", "kdd30", "kdd34")]
feat["irr_x_kdd30_late"] = feat.irr_share * (feat.kdd30_anth + feat.kdd30_fill)
CLIM_FEATURES = [c for c in feat.columns if c not in ("year", "adm2_pcode")]
print("predictors:", len(CLIM_FEATURES))
print(feat[["tmax_anth", "tmax_fill", "kdd30_fill", "kdd34_fill", "rain_till", "sm_fill"]].describe().round(2).to_string())

# %% [markdown]
# ### 2.4 MODIS NDVI/EVI (2001-2025, supplementary model)
# District cropland means of 16-day MOD13Q1 composites (good-quality pixels, WorldCover cropland mask) are
# summarised per season: mean NDVI in jointing (Feb), peak NDVI and mean EVI in anthesis-grain filling (Mar-Apr).
# Because EO indices cannot be projected into the future, they enter only a supplementary model that measures
# the added skill of Earth observation over climate alone.

# %%
mod = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW / "03_earth_observation" /
                                                              "modis_mod13q1_ndvi_evi" / "*.csv")))])
mod["date"] = pd.to_datetime(mod.composite)
mod = mod[(mod.n_good_cropland_px >= 20) & mod.adm2_pcode.isin(PCODES)]
mod["m"] = mod.date.dt.month
mod["year"] = mod.date.dt.year
eo = pd.concat([
    mod[mod.m == 2].groupby(["adm2_pcode", "year"]).NDVI_mean.mean().rename("ndvi_joint"),
    mod[mod.m.isin([3, 4])].groupby(["adm2_pcode", "year"]).NDVI_mean.max().rename("ndvi_peak_late"),
    mod[mod.m.isin([3, 4])].groupby(["adm2_pcode", "year"]).EVI_mean.mean().rename("evi_late"),
], axis=1).reset_index()
EO_FEATURES = ["ndvi_joint", "ndvi_peak_late", "evi_late"]
print(eo.describe().round(3).to_string())

# %% [markdown]
# ### 2.5 Modelling table

# %%
data = tgt.merge(feat, on=["adm2_pcode", "year"], how="inner").merge(eo, on=["adm2_pcode", "year"], how="left")
data = data.dropna(subset=CLIM_FEATURES)
train_all = data[data.own & (data.year != HOLDOUT_SEASON)].reset_index(drop=True)
print(f"modelling rows (own, excl. 2022): {len(train_all)} | districts: {train_all.adm2_pcode.nunique()} | "
      f"seasons: {train_all.year.min()}-{train_all.year.max()} ({train_all.year.nunique()})")
data.to_parquet(CACHE / "model_table.parquet")
