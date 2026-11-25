"""Process-based crop simulations for Paper 8 (Stage H) with WOFOST 7.3 (PCSE 6.0).

WOFOST is the first of the two crop models (APSIM-Wheat in apsim_package/ is the second).

Sites, soils, climate deltas and CO2 are shared with the APSIM package (apsim_package/inputs). Weather is the
district series of the hazard model: ERA5-Land Tmax/Tmin/dew point and CHIRPS rain (wheat-area district means),
with NASA POWER radiation and 2-m wind at the wheat-weighted centroid.

Experiments per site (each simulated season separately):
  hist      observed weather and CO2, 1985-2025, calibrated phenology
  base      observed 1991-2020 weather, constant 1991-2020 mean CO2; options: std, early, longer, heattol, early+longer
  gwl       31 GCMs x GWL 1.5/2/3 delta change on the 1991-2020 weather, with SSP5-8.5 CO2 and with baseline CO2
  median    ensemble-median change at each GWL with SSP5-8.5 CO2, with the adaptation options
Irrigated sites use potential production (WOFOST73_PP); rainfed Chakwal uses water-limited production
(WOFOST73_WLP_CWB, classic water balance).
"""
import datetime as dt
import json
import os
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
ROOT = Path(os.environ.get("PAPER8_ROOT", "."))
INP = ROOT / "apsim_package" / "inputs"
CM = ROOT / "crop_model"
OUT = CM / "outputs"
(CM / "inputs").mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "data_processed" / "analysis"
H = {"User-Agent": "crop-model-inputs/1.0"}
BASE_VARIETY = "Winter_wheat_101"     # WOFOST crop parameter repository; vernalisation/daylength switched off
TBASEM, TEFFMX, TSUMEM = 0.0, 30.0, 120.0
HEAT_TOLERANT_TMPFTB = [0.0, 0.01, 10.0, 0.6, 15.0, 1.0, 28.0, 1.0, 38.0, 0.0]   # AMAX decline shifted +3 degC
N_JOBS = max(1, (os.cpu_count() or 4) - 2)

sites = pd.read_csv(INP / "sites.csv")
soils = json.load(open(INP / "soils.json"))
deltas = pd.read_csv(INP / "deltas.csv")
co2s = pd.read_csv(INP / "co2_scenarios.csv")
BASE_CO2 = json.load(open(INP / "co2_baseline.json"))["baseline_1991_2020_co2"]


# ------------------------------------------------------------------------------------------- weather
def power_wind(site, la, lo):
    f = CM / "inputs" / f"power_wind_{site}.json"
    if not f.exists():
        url = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=WS2M&community=AG"
               f"&longitude={lo}&latitude={la}&start=19840101&end=20251231&format=JSON")
        for attempt in range(4):
            try:
                js = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=180))
                json.dump({"elev": js["geometry"]["coordinates"][2], "ws2m": js["properties"]["parameter"]["WS2M"]},
                          open(f, "w"))
                break
            except Exception as e:
                print("POWER retry", attempt, e)
                time.sleep(10)
    js = json.load(open(f))
    s = pd.Series(js["ws2m"])
    s.index = pd.to_datetime(s.index, format="%Y%m%d")
    s = s.where(s > -900)
    return float(js["elev"]), s.fillna(s.groupby(s.index.dayofyear).transform("mean"))


def read_met_radiation(site):
    path = INP / "met" / f"{site}_obs.met"
    with open(path) as fh:
        lines = fh.readlines()
    k = next(i for i, l in enumerate(lines) if l.startswith("year"))
    df = pd.read_csv(path, sep=r"\s+", skiprows=k + 2, header=None, names=["year", "day", "radn", "maxt", "mint", "rain", "co2"])
    df.index = pd.to_datetime(df.year.astype(str) + df.day.astype(str).str.zfill(3), format="%Y%j")
    return df


def vap_from_dew(td):
    return 6.108 * np.exp(17.27 * td / (td + 237.3))       # hPa


def site_weather(s):
    tmax = pd.read_parquet(CACHE / "daily_tmax.parquet")[s.pcode]
    tmin = pd.read_parquet(CACHE / "daily_tmin.parquet")[s.pcode]
    dew = pd.read_parquet(CACHE / "daily_dew.parquet")[s.pcode]
    met = read_met_radiation(s.site)
    elev, wind = power_wind(s.site, s.lat, s.lon)
    idx = met.index
    w = pd.DataFrame({"TMAX": tmax.reindex(idx), "TMIN": tmin.reindex(idx), "TDEW": dew.reindex(idx),
                      "RAIN_MM": met.rain, "IRRAD": met.radn * 1e6, "WIND": wind.reindex(idx), "CO2": met.co2}, index=idx)
    w["TMIN"] = np.minimum(w.TMIN, w.TMAX - 0.1)
    w["TDEW"] = np.minimum(w.TDEW, w.TMIN)
    assert not w.isna().any().any(), s.site
    return elev, w


def perturb(w, dm):
    """Delta method as in Stage A: Tmax/Tmin additive, rain multiplicative, dew point shifted with mean T."""
    w = w.loc["1990-01-01":"2020-12-31"].copy()
    dm = dm.set_index("month").reindex(range(1, 13))
    dm[["dtmax", "dtmin"]] = dm[["dtmax", "dtmin"]].fillna(0.0)
    dm["pr_ratio"] = dm["pr_ratio"].fillna(1.0)
    m = w.index.month
    dtx, dtn = m.map(dm.dtmax).values, m.map(dm.dtmin).values
    w["TMAX"] = w.TMAX + dtx
    w["TMIN"] = np.minimum(w.TMIN + dtn, w.TMAX - 0.1)
    w["TDEW"] = np.minimum(w.TDEW + (dtx + dtn) / 2, w.TMIN)
    w["RAIN_MM"] = w.RAIN_MM * m.map(dm.pr_ratio).values
    return w


# ------------------------------------------------------------------------------------------- WOFOST
def make_provider(w, la, lo, elev):
    from pcse.base import WeatherDataContainer, WeatherDataProvider
    from pcse.util import reference_ET

    class FrameWeather(WeatherDataProvider):
        def __init__(self):
            super().__init__()
            self.description = ["Paper 8 district weather"]
            for d, r in w.iterrows():
                day = d.date()
                vap = float(vap_from_dew(r.TDEW))
                e0, es0, et0 = reference_ET(day, la, elev, r.TMIN, r.TMAX, r.IRRAD, vap, r.WIND, 0.25, 0.5)
                wdc = WeatherDataContainer(LAT=la, LON=lo, ELEV=elev, DAY=day, IRRAD=float(r.IRRAD), TMIN=float(r.TMIN),
                                           TMAX=float(r.TMAX), VAP=vap, RAIN=float(r.RAIN_MM) / 10.0, WIND=float(r.WIND),
                                           E0=e0 / 10.0, ES0=es0 / 10.0, ET0=et0 / 10.0)
                self._store_WeatherDataContainer(wdc, day)
    return FrameWeather()


def soil_params(soil):
    th = np.array(soil["Thickness"], float)
    z = np.cumsum(th) / 10.0                           # cm
    wt = np.clip(np.minimum(z, 100) - np.r_[0, z[:-1]], 0, None)   # weights over the top 100 cm
    avg = lambda k: float(np.average(soil[k], weights=wt))
    ks = np.array(soil["KS"]) / 10.0                   # mm/day -> cm/day
    return {"SMW": avg("LL15"), "SMFCF": avg("DUL"), "SM0": avg("SAT"), "CRAIRC": 0.06, "RDMSOL": 120.0,
            "K0": float(ks[0]), "SOPE": float(min(ks[0], 50.0)), "KSUB": float(min(ks[-1], 20.0))}


_CROP = None


def crop_data(reload=False):
    global _CROP
    if _CROP is None or reload:
        from pcse.input import YAMLCropDataProvider
        from pcse.models import Wofost73_PP
        _CROP = YAMLCropDataProvider(model=Wofost73_PP, force_reload=reload)   # WOFOST 7.3 parameter branch
    return _CROP


def thermal_calibration(w, sow, anth_doy, mat_doy, years=range(1991, 2021)):
    """TSUM1/TSUM2 that place mean simulated anthesis and maturity on the target days (temperature-only development)."""
    t1, t2 = [], []
    anth_doy, mat_doy = int(anth_doy), int(mat_doy)
    for y in years:
        sd = dt.datetime.strptime(f"{sow}-{y - 1}", "%d-%b-%Y")
        seg = w.loc[sd: sd + dt.timedelta(days=240)]
        temp = (seg.TMAX + seg.TMIN) / 2
        dtsume = np.clip(temp - TBASEM, 0, TEFFMX - TBASEM)
        emerg = seg.index[np.searchsorted(dtsume.cumsum().values, TSUMEM)]
        anth = dt.datetime(y, 1, 1) + dt.timedelta(days=anth_doy - 1)
        mat = dt.datetime(y, 1, 1) + dt.timedelta(days=mat_doy - 1)
        dtsm = np.clip(temp, 0, 30)                      # DTSMTB of the parameter set: linear 0-30 degC, flat above
        t1.append(dtsm.loc[emerg + dt.timedelta(days=1): anth].sum())
        t2.append(dtsm.loc[anth + dt.timedelta(days=1): mat].sum())
    return float(np.mean(t1)), float(np.mean(t2))


def quiet_pcse():
    """PCSE logs to one rotating file; parallel workers make that fail noisily, so logging is switched off."""
    import logging
    import pcse  # noqa: F401
    logging.raiseExceptions = False
    lg = logging.getLogger("pcse")
    lg.setLevel(logging.CRITICAL)
    for h in list(lg.handlers):
        lg.removeHandler(h)


def run_seasons(w, s, elev, sow, tsum1, tsum2, co2, years, heattol=False):
    quiet_pcse()
    from pcse.base import ParameterProvider
    from pcse.input import WOFOST73SiteDataProvider
    from pcse.models import Wofost73_PP, Wofost73_WLP_CWB
    w = w[w.index.month.isin([9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7])]     # growing-season days only (speed)
    wp = make_provider(w, float(s.lat), float(s.lon), elev)
    cd = crop_data()
    soil = soil_params(soils[s.site])
    Model = Wofost73_PP if s.irrigated else Wofost73_WLP_CWB
    out = []
    for y in years:
        # PCSE clears parameter overrides when a crop ends, so the provider is rebuilt for every season
        cd.set_active_crop("wheat", BASE_VARIETY)
        # site parameters as a plain dict: the provider class only accepts CO2 of 320-700 ppm, whereas the crop
        # parameter set defines CO2 responses up to 2000 ppm (10 of 31 GCMs exceed 700 ppm at GWL 3 degC)
        site = dict(WOFOST73SiteDataProvider(WAV=(5.0 if s.irrigated else 10.0), CO2=400.0))
        site["CO2"] = float(co2)
        params = ParameterProvider(cropdata=cd, soildata=soil, sitedata=site)
        params.set_override("IDSL", 0)
        params.set_override("TSUM1", tsum1)
        params.set_override("TSUM2", tsum2)
        if heattol:
            params.set_override("TMPFTB", HEAT_TOLERANT_TMPFTB)
        sd = dt.datetime.strptime(f"{sow}-{y - 1}", "%d-%b-%Y").date()
        agro = [{sd - dt.timedelta(days=5): {"CropCalendar": {
            "crop_name": "wheat", "variety_name": BASE_VARIETY, "crop_start_date": sd, "crop_start_type": "sowing",
            "crop_end_date": None, "crop_end_type": "maturity", "max_duration": 230},
            "TimedEvents": None, "StateEvents": None}}]
        try:
            m = Model(params, wp, agro)
            m.run_till_terminate()
            r = m.get_summary_output()[0]
            out.append({"season": y, "sow_date": sd, "anthesis_date": r.get("DOA"), "maturity_date": r.get("DOM"),
                        "Yield": r.get("TWSO"), "Biomass": r.get("TAGP"), "LAImax": r.get("LAIMAX"),
                        "CTRAT": r.get("CTRAT")})
        except Exception as e:
            out.append({"season": y, "sow_date": sd, "error": str(e)[:200]})
    return pd.DataFrame(out)


def heat_diag(res, w):
    rows = []
    for r in res.itertuples(index=False):
        rec = {"season": r.season}
        a, m = getattr(r, "anthesis_date", None), getattr(r, "maturity_date", None)
        if a is not None and not pd.isna(a):
            a = pd.Timestamp(a)
            win = w.loc[a - pd.Timedelta(days=7): a + pd.Timedelta(days=7)]
            rec["days_tmax_ge34_anthesis"] = int((win.TMAX >= 34).sum())
            if m is not None and not pd.isna(m):
                gf = w.loc[a: pd.Timestamp(m)]
                rec["tmax_mean_grain_fill"] = float(gf.TMAX.mean())
                rec["kdd34_grain_fill"] = float((gf.TMAX - 34).clip(lower=0).sum())
        rows.append(rec)
    return res.merge(pd.DataFrame(rows), on="season", how="left")


def task(s, elev, w, tag, sow, tsum1, tsum2, co2, years, heattol=False):
    res = run_seasons(w, s, elev, sow, tsum1, tsum2, co2, years, heattol)
    res = heat_diag(res, w)
    res.insert(0, "scenario", tag)
    res.insert(0, "site", s.site)
    return res


def shift(sow, days):
    d = dt.datetime.strptime(sow + "-2001", "%d-%b-%Y") + dt.timedelta(days=days)
    return d.strftime("%d-%b").lower()


# ------------------------------------------------------------------------------------------- main
if __name__ == "__main__":
    t0 = time.time()
    crop_data(reload=True)                                # download/cache the WOFOST 7.3 crop parameters once
    for _, s in sites.iterrows():
        power_wind(s.site, s.lat, s.lon)                  # fetch wind serially before the parallel runs
    jobs, cal_rows = [], []
    for _, s in sites.iterrows():
        elev, w = site_weather(s)
        tsum1, tsum2 = thermal_calibration(w, s.sow, s.anth_doy, s.mat_doy)
        cal_rows.append({"site": s.site, "district": s.district, "elev_m": elev, "TSUM1": round(tsum1, 1),
                         "TSUM2": round(tsum2, 1), "target_anthesis_doy": s.anth_doy, "target_maturity_doy": s.mat_doy,
                         **{k: round(v, 3) for k, v in soil_params(soils[s.site]).items()}})
        early = shift(s.sow, -15)
        base_w = w.loc["1990-01-01":"2020-12-31"]
        yrs = range(1991, 2021)
        jobs.append((s, elev, w, "hist", s.sow, tsum1, tsum2, None, range(1985, 2026)))
        for opt, sow, f, ht in (("std", s.sow, 1.0, False), ("early", early, 1.0, False), ("longer", s.sow, 1.1, False),
                                ("heattol", s.sow, 1.0, True), ("early+longer", early, 1.1, False)):
            jobs.append((s, elev, base_w, f"base|{opt}", sow, tsum1 * f, tsum2 * f, BASE_CO2, yrs, ht))
        ds = deltas[deltas.site == s.site]
        for (model, gwl), dm in ds.groupby(["model", "gwl"]):
            pw = perturb(w, dm)
            c = float(co2s[(co2s.model == model) & (co2s.gwl == gwl)].co2.iloc[0])
            jobs.append((s, elev, pw, f"gwl{gwl}|{model}|co2", s.sow, tsum1, tsum2, c, yrs))
            jobs.append((s, elev, pw, f"gwl{gwl}|{model}|noco2", s.sow, tsum1, tsum2, BASE_CO2, yrs))
        for gwl, dg in ds.groupby("gwl"):
            med = dg.groupby("month")[["dtmax", "dtmin", "pr_ratio"]].median().reset_index()
            pw = perturb(w, med)
            c = float(co2s[co2s.gwl == gwl].co2.median())
            for opt, sow, f, ht in (("std", s.sow, 1.0, False), ("early", early, 1.0, False),
                                    ("longer", s.sow, 1.1, False), ("heattol", s.sow, 1.0, True),
                                    ("early+longer", early, 1.1, False)):
                jobs.append((s, elev, pw, f"gwl{gwl}|median|{opt}", sow, tsum1 * f, tsum2 * f, c, yrs, ht))
    pd.DataFrame(cal_rows).to_csv(OUT / "calibration.csv", index=False)
    print(pd.DataFrame(cal_rows).to_string(index=False))

    # the hindcast uses each year's observed CO2: run it per year with that year's concentration
    hist_jobs = [j for j in jobs if j[3] == "hist"]
    jobs = [j for j in jobs if j[3] != "hist"]
    print(f"{len(jobs)} scenario tasks + hindcast; {N_JOBS} workers")

    def hist_task(s, elev, w, tag, sow, tsum1, tsum2, _co2, years):
        parts = [task(s, elev, w, tag, sow, tsum1, tsum2, float(w.loc[f"{y}-01-01", "CO2"]), [y]) for y in years]
        return pd.concat(parts, ignore_index=True)

    res_hist = Parallel(n_jobs=N_JOBS)(delayed(hist_task)(*j) for j in hist_jobs)
    res = Parallel(n_jobs=N_JOBS, batch_size=1)(delayed(task)(*j) for j in jobs)
    allr = pd.concat(res_hist + res, ignore_index=True)
    allr.to_csv(OUT / "wofost_results.csv", index=False)
    n_err = int(allr["error"].notna().sum()) if "error" in allr else 0
    print(f"done: {len(allr)} simulated seasons, {n_err} failed, {(time.time() - t0) / 60:.1f} min")
