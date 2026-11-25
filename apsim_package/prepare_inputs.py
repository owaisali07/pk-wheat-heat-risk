"""Prepare all inputs for the APSIM-Wheat site simulations (run once, from the repository root).

Outputs (in inputs/):
  sites.csv             site, district code, province, wheat-weighted centroid, management, phenology targets
  met/<site>_obs.met    daily weather 1984-2025: ERA5-Land Tmax/Tmin and CHIRPS rain (district wheat-area means,
                        identical to the hazard model), NASA POWER radiation at the centroid, annual CO2 (NOAA)
  deltas.csv            monthly changes (Tmax, Tmin additive; rain ratio) per site, GCM and warming level, taken from
                        the same 31 NEX-GDDP-CMIP6 models and warming-level windows as the hazard model
  co2_scenarios.csv     SSP5-8.5 CO2 averaged over each GCM's warming-level window (RCMIP / Meinshausen et al. 2020)
  soils.json            six-layer soil profiles from SoilGrids 2.0 with Saxton & Rawls (2006) hydraulic properties
  observed_yield.csv    district wheat yields (harmonised panel) for the evaluation of the hindcast
"""
import os
import json
import time
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask

ROOT = Path(os.environ.get("PAPER8_ROOT", "."))
PKG = ROOT / "apsim_package"
INP = PKG / "inputs"
(INP / "met").mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "data_processed" / "analysis"
H = {"User-Agent": "apsim-input-preparation/1.0"}

# ------------------------------------------------------------------ sites and management
# Sowing dates follow the provincial recommendations for timely sowing; the anthesis and maturity targets
# (day of year) are approximate windows consistent with harvest in March-April in Sindh and April-May in the
# north. They are used only to choose among generic spring cultivars, not as observations.
SITES = pd.DataFrame([
    # site, pcode, province, irrigated, sowing (d-mmm), anthesis DOY, maturity DOY
    ("Sanghar", "PK718", "Sindh", True, "10-nov", 60, 90),
    ("ShaheedBenazirAbad", "PK719", "Sindh", True, "10-nov", 64, 95),
    ("Bahawalpur", "PK603", "Punjab", True, "20-nov", 74, 105),
    ("Faisalabad", "PK608", "Punjab", True, "15-nov", 79, 110),
    ("Chakwal", "PK605", "Punjab", False, "30-oct", 91, 125),
    ("DIKhan", "PK509", "Khyber Pakhtunkhwa", True, "15-nov", 74, 105),
], columns=["site", "pcode", "province", "irrigated", "sow", "anth_doy", "mat_doy"])

adm = gpd.read_file(ROOT / "data_raw" / "01_boundaries" / "pak_admin2.shp").to_crs(4326).set_index("adm2_pcode")
lat, lon = [], []
with rasterio.open(ROOT / "data_raw" / "04_crop" / "spam2010" / "spam2010V2r0_global_A_WHEA_A.tif") as src:
    for pc in SITES.pcode:
        arr, tr = rio_mask(src, [adm.loc[pc].geometry.__geo_interface__], crop=True, filled=True, nodata=0)
        a = np.where(arr[0] > 0, arr[0], 0.0)
        rr, cc = np.indices(a.shape)
        x = tr.c + (cc + 0.5) * tr.a
        y = tr.f + (rr + 0.5) * tr.e
        lon.append(round(float((a * x).sum() / a.sum()), 3))
        lat.append(round(float((a * y).sum() / a.sum()), 3))
SITES["lat"], SITES["lon"] = lat, lon
SITES["district"] = [adm.loc[p, "adm2_name"] for p in SITES.pcode]
SITES.to_csv(INP / "sites.csv", index=False)
print(SITES.to_string(index=False))

# ------------------------------------------------------------------ CO2
noaa = pd.read_csv(INP / "raw" / "co2_annmean_mlo.csv", comment="#")
co2_hist = dict(zip(noaa.year, noaa["mean"]))
last = max(co2_hist)
for y in range(last + 1, 2026):              # extend by the 2015-2024 mean growth rate if needed
    co2_hist[y] = co2_hist[y - 1] + np.mean(np.diff([co2_hist[k] for k in range(2014, last + 1)]))
rc = pd.read_csv(INP / "raw" / "rcmip-concentrations-annual-means-v5-1-0.csv")
ssp = rc[(rc.Scenario == "ssp585") & (rc.Region == "World") & (rc.Variable == "Atmospheric Concentrations|CO2")].iloc[0]
co2_ssp = {int(c): float(ssp[c]) for c in rc.columns if c.isdigit() and 2015 <= int(c) <= 2100}
for y in range(1980, 2015):                    # windows that start before 2015 use observed concentrations
    co2_ssp[y] = co2_hist[y]

# ------------------------------------------------------------------ radiation from NASA POWER
def power_radiation(la, lo):
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN&community=AG"
           f"&longitude={lo}&latitude={la}&start=19840101&end=20251231&format=JSON")
    for attempt in range(4):
        try:
            js = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=180))
            s = pd.Series(js["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"])
            s.index = pd.to_datetime(s.index, format="%Y%m%d")
            return s.where(s > -900)
        except Exception as e:
            print("  POWER retry", attempt, e)
            time.sleep(10)
    raise RuntimeError("NASA POWER request failed")


tmax = pd.read_parquet(CACHE / "daily_tmax.parquet")
tmin = pd.read_parquet(CACHE / "daily_tmin.parquet")
prcp = pd.read_parquet(CACHE / "daily_prcp.parquet")


def write_met(path, df, la, comment):
    t = (df.maxt + df.mint) / 2
    mon = t.groupby(df.index.month).mean()
    tav, amp = float(t.mean()), float(mon.max() - mon.min())
    with open(path, "w", newline="\n") as f:
        f.write("[weather.met.weather]\n")
        f.write(f"! {comment}\n")
        f.write(f"latitude = {la:.3f} (DECIMAL DEGREES)\n")
        f.write(f"tav = {tav:.2f} (oC) ! annual average ambient temperature\n")
        f.write(f"amp = {amp:.2f} (oC) ! annual amplitude in mean monthly temperature\n")
        f.write("year day radn maxt mint rain co2\n")
        f.write("() () (MJ/m^2) (oC) (oC) (mm) (ppm)\n")
        for d, r in df.iterrows():
            f.write(f"{d.year} {d.dayofyear} {r.radn:.2f} {r.maxt:.2f} {r.mint:.2f} {r.rain:.2f} {r.co2:.1f}\n")


for _, s in SITES.iterrows():
    rad = power_radiation(s.lat, s.lon)
    idx = pd.date_range("1984-01-01", min(tmax.index.max(), prcp.index.max(), rad.index.max()), freq="D")
    df = pd.DataFrame({"maxt": tmax[s.pcode].reindex(idx), "mint": tmin[s.pcode].reindex(idx),
                       "rain": prcp[s.pcode].reindex(idx).clip(lower=0), "radn": rad.reindex(idx)}, index=idx)
    clim = df.radn.groupby(df.index.dayofyear).transform("mean")      # fill missing radiation by day-of-year mean
    df["radn"] = df.radn.fillna(clim)
    df["mint"] = np.minimum(df.mint, df.maxt - 0.1)
    df["co2"] = [co2_hist[y] for y in df.index.year]
    n_missing = int(df[["maxt", "mint", "rain", "radn"]].isna().sum().sum())
    assert n_missing == 0, (s.site, n_missing)
    write_met(INP / "met" / f"{s.site}_obs.met", df, s.lat,
              f"{s.district} ({s.pcode}): ERA5-Land Tmax/Tmin and CHIRPS rain (wheat-area district means), "
              "NASA POWER ALLSKY_SFC_SW_DWN radiation, NOAA Mauna Loa annual CO2")
    print("met written", s.site, idx[0].date(), idx[-1].date())
    time.sleep(2)

# ------------------------------------------------------------------ climate deltas (same as the hazard model)
win = pd.read_csv(ROOT / "results" / "tables" / "S1_gcm_warming_windows.csv")
rows, co2_rows = [], []
for model in sorted(win.model.unique()):
    df = pd.read_parquet(CACHE / "nex" / f"{model}.parquet")
    df = df[df.adm2_pcode.isin(SITES.pcode)]
    piv = df.pivot_table(index=["year", "month", "adm2_pcode"], columns="var", values="value").reset_index()
    base = piv[piv.year.between(1991, 2020)].groupby(["month", "adm2_pcode"])[["tasmax", "tasmin", "pr"]].mean()
    for _, w in win[win.model == model].iterrows():
        fut = piv[piv.year.between(w.start, w.end)].groupby(["month", "adm2_pcode"])[["tasmax", "tasmin", "pr"]].mean()
        d = pd.DataFrame({"dtmax": fut.tasmax - base.tasmax, "dtmin": fut.tasmin - base.tasmin,
                          "pr_ratio": (fut.pr / base.pr.replace(0, np.nan)).clip(0.2, 5).fillna(1.0)}).reset_index()
        d["model"], d["gwl"] = model, w.gwl
        rows.append(d)
        co2_rows.append({"model": model, "gwl": w.gwl, "start": w.start, "end": w.end,
                         "co2": np.mean([co2_ssp[y] for y in range(int(w.start), int(w.end) + 1)])})
deltas = pd.concat(rows, ignore_index=True).merge(SITES[["pcode", "site"]], left_on="adm2_pcode", right_on="pcode")
# deltas exist for November-April; October takes the November change and May the April change
ext = []
for src_m, dst_m in ((11, 10), (4, 5)):
    e = deltas[deltas.month == src_m].copy()
    e["month"] = dst_m
    ext.append(e)
deltas = pd.concat([deltas] + ext, ignore_index=True)
deltas[["site", "model", "gwl", "month", "dtmax", "dtmin", "pr_ratio"]].round(4).to_csv(INP / "deltas.csv", index=False)
co2s = pd.DataFrame(co2_rows)
co2s.round(1).to_csv(INP / "co2_scenarios.csv", index=False)
base_co2 = float(np.mean([co2_hist[y] for y in range(1991, 2021)]))
json.dump({"baseline_1991_2020_co2": round(base_co2, 1)}, open(INP / "co2_baseline.json", "w"))
print("deltas:", deltas.model.nunique(), "models;", "baseline CO2", round(base_co2, 1),
      "| SSP5-8.5 window CO2 by GWL (median):", co2s.groupby("gwl").co2.median().round(0).to_dict())

# ------------------------------------------------------------------ soils (SoilGrids 2.0 + Saxton & Rawls 2006)
DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]
THICK = [50, 100, 150, 300, 400, 1000]


def saxton_rawls(S, C, OM):
    """S, C as fractions, OM in %; returns LL15, DUL, SAT (v/v) and Ks (mm/h)."""
    t1500t = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
    t1500 = t1500t + (0.14 * t1500t - 0.02)
    t33t = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
    t33 = t33t + (1.283 * t33t ** 2 - 0.374 * t33t - 0.015)
    ts33t = 0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * S * OM - 0.027 * C * OM - 0.584 * S * C + 0.078
    ts33 = ts33t + (0.636 * ts33t - 0.107)
    tS = t33 + ts33 - 0.097 * S + 0.043
    B = (np.log(1500) - np.log(33)) / (np.log(t33) - np.log(t1500))
    ks = 1930 * (tS - t33) ** (3 - 1 / B)
    return t1500, t33, tS, ks


def soilgrids(la, lo):
    q = "&".join([f"property={p}" for p in ("clay", "sand", "silt", "soc", "bdod", "phh2o")] +
                 [f"depth={d}" for d in DEPTHS] + ["value=mean"])
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lo}&lat={la}&{q}"
    for attempt in range(5):
        try:
            js = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=120))
            out = {}
            for layer in js["properties"]["layers"]:
                fac = layer["unit_measure"]["d_factor"]
                out[layer["name"]] = [d["values"]["mean"] / fac if d["values"]["mean"] is not None else np.nan
                                      for d in sorted(layer["depths"], key=lambda d: DEPTHS.index(d["label"]))]
            return out
        except Exception as e:
            print("  SoilGrids retry", attempt, e)
            time.sleep(15)
    raise RuntimeError("SoilGrids request failed")


soils = {}
for _, s in SITES.iterrows():
    g = soilgrids(s.lat, s.lon)
    # after the d_factor: clay/sand/silt in %, soc in g/kg (-> % /10), bdod in kg/dm3, pH
    S = np.array(g["sand"]) / 100
    C = np.array(g["clay"]) / 100
    OC = np.array(g["soc"]) / 10
    BD = np.array(g["bdod"])
    PH = np.array(g["phh2o"])
    if np.isnan(np.r_[S, C, OC, BD, PH]).any():           # fill gaps from the layer above
        for arr in (S, C, OC, BD, PH):
            for k in range(1, len(arr)):
                if np.isnan(arr[k]):
                    arr[k] = arr[k - 1]
    OM = np.clip(OC * 1.724, 0.05, 8)
    ll15, dul, sat, ks = saxton_rawls(S, C, OM)
    sat = np.minimum(sat, 1 - BD / 2.65 - 0.01)
    dul = np.minimum(dul, sat - 0.02)
    ll15 = np.minimum(ll15, dul - 0.05)
    soils[s.site] = {
        "source": f"SoilGrids 2.0 at {s.lat}, {s.lon}; hydraulic properties from Saxton and Rawls (2006)",
        "Thickness": THICK, "Sand": S.round(3).tolist(), "Clay": C.round(3).tolist(),
        "BD": BD.round(3).tolist(), "LL15": ll15.round(3).tolist(), "DUL": dul.round(3).tolist(),
        "SAT": sat.round(3).tolist(), "KS": (ks * 24).round(1).tolist(),
        "AirDry": np.r_[ll15[0] * 0.5, ll15[1] * 0.8, ll15[2:]].round(3).tolist(),
        "Carbon": np.clip(OC, 0.05, None).round(3).tolist(), "PH": PH.round(2).tolist(),
    }
    print("soil", s.site, "clay", C.round(2), "PAWC mm", round(float(((dul - ll15) * np.array(THICK)).sum()), 0))
    time.sleep(3)
json.dump(soils, open(INP / "soils.json", "w"), indent=1)

# ------------------------------------------------------------------ observed yields for evaluation
panel = pd.read_csv(ROOT / "data_processed" / "yield_panel" / "wheat_yield_district_panel_1982_2025.csv")
cols = ["adm2_pcode", "adm2_name", "season", "year", "yield_t_ha", "area_ha", "prod_t", "flag", "source"]
panel[panel.adm2_pcode.isin(SITES.pcode)][cols].to_csv(INP / "observed_yield.csv", index=False)
print("inputs written to", INP)
