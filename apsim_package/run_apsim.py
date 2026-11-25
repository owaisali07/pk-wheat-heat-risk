"""Build and run the APSIM Next Generation wheat simulations for Paper 8, then collect the results.

Runs wherever APSIM Next Generation can execute (Linux, Windows or Google Colab; see README.md). It needs the ApsimX source tree (for the example templates) and a
built Models.dll.

    python run_apsim.py --apsimx /path/to/ApsimX --stage smoke        # 1 site, 3 seasons: checks the set-up
    python run_apsim.py --apsimx /path/to/ApsimX --stage calibrate    # chooses a cultivar per site
    python run_apsim.py --apsimx /path/to/ApsimX --stage production   # hindcast, baseline, warming, adaptation

Results are written to outputs/ as CSV files (one row per site, scenario and season).
"""
import argparse
import copy
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent
INP, WORK, OUT = PKG / "inputs", PKG / "work", PKG / "outputs"
CANDIDATE_CULTIVARS = ["Spring Quick-Mid", "Spring Mid", "Spring Mid-Slow", "Spring Slow", "Spring Slow-Very Slow",
                       "Spring Very Slow", "Yecora"]
SLOWER = {"Spring Very Quick-Quick": "Spring Quick", "Spring Quick": "Spring Quick-Mid",
          "Spring Quick-Mid": "Spring Mid", "Spring Mid": "Spring Mid-Slow", "Spring Mid-Slow": "Spring Slow",
          "Spring Slow": "Spring Slow-Very Slow", "Spring Slow-Very Slow": "Spring Very Slow",
          "Spring Very Slow": "Spring Very Slow", "Yecora": "Spring Mid"}
STAGE_ANTHESIS, STAGE_END_GRAIN_FILL = 8.0, 10.0      # APSIM-Wheat phenology stage numbers
FERT_N = (75.0, 75.0)                                 # kg N/ha at sowing and about 45 days later
SOWING_IRRIGATION_CODE = """using Models.Soils;
using System;
using System.Linq;
using Models.Core;
using Models.PMF;
namespace Models
{
    [Serializable]
    public class Script : Model
    {
        [Link] Irrigation Irrigation;

        [Description("Crop")]
        public IPlant Crop { get; set; }

        [Description("Irrigation at sowing (mm)")]
        public double Amount { get; set; }

        [EventSubscribe("Sowing")]
        private void OnSowing(object sender, EventArgs e)
        {
            Model crop = sender as Model;
            if (Crop != null && crop.Name.ToLower() == (Crop as IModel).Name.ToLower())
                Irrigation.Apply(Amount);
        }
    }
}"""


# ------------------------------------------------------------------------------------------- helpers
def find(node, name):
    if node.get("Name") == name:
        return node
    for c in node.get("Children", []) or []:
        r = find(c, name)
        if r is not None:
            return r
    return None


def find_type(node, type_prefix, acc=None):
    acc = [] if acc is None else acc
    if node.get("$type", "").startswith(type_prefix):
        acc.append(node)
    for c in node.get("Children", []) or []:
        find_type(c, type_prefix, acc)
    return acc


def set_params(manager, **kv):
    params = {p["Key"]: p for p in manager["Parameters"]}
    for k, v in kv.items():
        if k not in params:
            raise KeyError(f"manager '{manager['Name']}' has no parameter {k}")
        params[k]["Value"] = str(v)


def dmmm(sow, offset_days):
    d = datetime.strptime(sow + "-2001", "%d-%b-%Y") + timedelta(days=offset_days)
    return d.strftime("%d-%b").lstrip("0").lower()


def read_met(path):
    with open(path) as f:
        lines = f.readlines()
    head_end = next(i for i, l in enumerate(lines) if l.startswith("year"))
    header = lines[:head_end]
    df = pd.read_csv(path, sep=r"\s+", skiprows=head_end + 2, header=None,
                     names=["year", "day", "radn", "maxt", "mint", "rain", "co2"])
    return header, df


def write_met(path, header, df):
    with open(path, "w", newline="\n") as f:
        f.writelines(header)
        f.write("year day radn maxt mint rain co2\n() () (MJ/m^2) (oC) (oC) (mm) (ppm)\n")
        for r in df.itertuples(index=False):
            f.write(f"{int(r.year)} {int(r.day)} {r.radn:.2f} {r.maxt:.2f} {r.mint:.2f} {r.rain:.2f} {r.co2:.1f}\n")


def perturbed_met(site, tag, d_month, co2, years=(1990, 2020)):
    """Delta method: monthly Tmax/Tmin shifts and rain ratios applied to observed 1990-2020 weather."""
    path = WORK / "met" / f"{site}_{tag}.met"
    if path.exists():
        return path
    header, df = read_met(INP / "met" / f"{site}_obs.met")
    df = df[df.year.between(*years)].copy()
    month = pd.to_datetime(df.year.astype(str) + df.day.astype(str).str.zfill(3), format="%Y%j").dt.month
    dm = d_month.set_index("month").reindex(range(1, 13))
    dm[["dtmax", "dtmin"]] = dm[["dtmax", "dtmin"]].fillna(0.0)
    dm["pr_ratio"] = dm["pr_ratio"].fillna(1.0)
    df["maxt"] = df.maxt + month.map(dm.dtmax).values
    df["mint"] = np.minimum(df.mint + month.map(dm.dtmin).values, df.maxt - 0.1)
    df["rain"] = df.rain * month.map(dm.pr_ratio).values
    df["co2"] = co2
    t = (df.maxt + df.mint) / 2
    mon = t.groupby(month.values).mean()
    header = [h for h in header if not h.startswith(("!", "tav", "amp"))]
    header += [f"tav = {t.mean():.2f} (oC) ! annual average ambient temperature\n",
               f"amp = {mon.max() - mon.min():.2f} (oC) ! annual amplitude in mean monthly temperature\n",
               f"! delta-method scenario {tag}, CO2 {co2:.1f} ppm\n"]
    write_met(path, header, df)
    return path


def baseline_met(site, co2):
    """Observed 1990-2020 weather with a constant CO2 (the reference for the warming-level runs)."""
    path = WORK / "met" / f"{site}_base.met"
    if not path.exists():
        header, df = read_met(INP / "met" / f"{site}_obs.met")
        df = df[df.year.between(1990, 2020)].copy()
        df["co2"] = co2
        write_met(path, header, df)
    return path


# ------------------------------------------------------------------------------------------- templates
class Templates:
    def __init__(self, apsimx_root):
        root = Path(apsimx_root)
        self.wheat = json.load(open(root / "Examples" / "Wheat.apsimx", encoding="utf-8"))
        ag = json.load(open(root / "Examples" / "AgPasture.apsimx", encoding="utf-8"))
        irr_mgr = [m for m in find_type(ag, "Models.Manager") if any("Irrigation.Apply" in l for l in m.get("CodeArray", []))]
        irr_node = find_type(ag, "Models.Irrigation")
        if not irr_mgr or not irr_node:
            raise RuntimeError("could not find the AutomaticIrrigation manager / Irrigation model in AgPasture.apsimx")
        self.irr_mgr = {k: v for k, v in irr_mgr[0].items()}
        self.irr_mgr["Name"] = "AutomaticIrrigation"
        self.irr_node = {k: v for k, v in irr_node[0].items() if k != "Children"}
        self.irr_node["Children"] = []
        self.sim = find_type(self.wheat, "Models.Core.Simulation,")[0]      # the Simulation, not the Simulations root
        self.datastore = find_type(self.wheat, "Models.Storage.DataStore")[0]


def merge_top_layers(s):
    """Combine the first two SoilGrids layers (0-5, 5-15 cm) into one 150-mm layer, thickness-weighted.
    A 50-mm top layer holds only a few mm of plant-available water; it dries within days after sowing in hot
    weather and confines the seedling roots, which stalls establishment in APSIM."""
    if s["Thickness"][0] >= 100:
        return s
    t0, t1 = s["Thickness"][0], s["Thickness"][1]
    out = dict(s)
    for k, v in s.items():
        if isinstance(v, list) and len(v) == len(s["Thickness"]):
            if k == "Thickness":
                out[k] = [t0 + t1] + v[2:]
            else:
                out[k] = [round((v[0] * t0 + v[1] * t1) / (t0 + t1), 4)] + v[2:]
    out["AirDry"] = [round(out["LL15"][0] * 0.5, 4)] + out["AirDry"][1:]
    return out


def build_soil(soil_node, s, irrigated):
    s = merge_top_layers(s)
    n = len(s["Thickness"])
    for child in soil_node["Children"]:
        t = child["$type"]
        if "Thickness" in child:
            child["Thickness"] = [float(x) for x in s["Thickness"]]
        if t.startswith("Models.Soils.Physical"):
            child.update(BD=s["BD"], AirDry=s["AirDry"], LL15=s["LL15"], DUL=s["DUL"], SAT=s["SAT"], KS=s["KS"],
                         ParticleSizeClay=[100 * c for c in s["Clay"]], ParticleSizeSand=[100 * x for x in s["Sand"]],
                         ParticleSizeSilt=[100 * (1 - c - x) for c, x in zip(s["Clay"], s["Sand"])])
            for k in list(child.keys()):
                if k.endswith("Metadata"):
                    child[k] = None
            for sc in child.get("Children", []):
                if sc["$type"].startswith("Models.Soils.SoilCrop"):
                    sc["LL"] = s["LL15"]
                    sc["KL"] = [0.06, 0.06, 0.04, 0.02, 0.01][:n]
                    sc["XF"] = [1.0] * n
                    for k in ("LLMetadata", "KLMetadata", "XFMetadata"):
                        sc[k] = None
        elif t.startswith("Models.WaterModel.WaterBalance"):
            child["SWCON"] = [0.3] * n
            child["KLAT"] = None
        elif t.startswith("Models.Soils.Organic"):
            child["Carbon"] = s["Carbon"]
            child["SoilCNRatio"] = [12.0] * n
            child["FBiom"] = [0.04, 0.02, 0.01, 0.01, 0.01][:n]
            child["FInert"] = [0.5, 0.8, 1.0, 1.0, 1.0][:n]
            child["FOM"] = [150.0, 60.0, 30.0, 15.0, 5.0][:n]
            child["CarbonMetadata"] = child["FOMMetadata"] = None
        elif t.startswith("Models.Soils.Chemical"):
            child["PH"] = s["PH"]
            for k in ("EC", "ESP", "CEC"):
                child[k] = None
        elif t.startswith("Models.Soils.Water"):
            frac = 0.9 if irrigated else 0.5     # pre-sowing irrigation (rauni) vs. residual monsoon moisture
            init = [ll + frac * (du - ll) for ll, du in zip(s["LL15"], s["DUL"])]
            child["InitialValues"] = init
            child["InitialPAWmm"] = float(sum((i - ll) * th for i, ll, th in zip(init, s["LL15"], s["Thickness"])))
            child["RelativeTo"] = "LL15"
        elif t.startswith("Models.Soils.Solute"):
            if child["Name"] == "NO3":
                child["InitialValues"] = [10.0, 5.0, 3.0, 2.0, 1.0][:n]
            elif child["Name"] == "NH4":
                child["InitialValues"] = [1.0, 0.3, 0.1, 0.1, 0.1][:n]
            else:
                child["InitialValues"] = [0.0] * n


def make_simulation(T, name, site, soil, metfile, start, end, cultivar, sow, irrigated, daily=True):
    sim = copy.deepcopy(T.sim)
    sim["Name"] = name
    clock = find(sim, "Clock")
    clock["Start"], clock["End"] = f"{start}T00:00:00", f"{end}T00:00:00"
    find(sim, "Weather")["FileName"] = str(Path(metfile).resolve()).replace("\\", "/")
    sim["Children"] = [c for c in sim["Children"] if not c["$type"].startswith("Models.Graph")]
    field = find(sim, "Field")
    soil_node = find_type(field, "Models.Soils.Soil, ")[0]
    soil_node.update(Site=site["district"], Country="Pakistan", Latitude=float(site["lat"]), Longitude=float(site["lon"]),
                     DataSource=soil["source"])
    build_soil(soil_node, soil, irrigated)
    sow_mgr = find(field, "Sow using a variable rule")
    # sowing on a fixed date (one-day window, no moisture condition); irrigated sites are also irrigated at sowing,
    # at the rainfed site germination waits for moisture in the seed layer
    sow_mgr["Name"] = "Sow on a fixed date"
    set_params(sow_mgr, StartDate=sow, EndDate=sow, MinESW=-1.0, MinRain=-1.0, CultivarName=cultivar,
               SowingDepth=40.0, RowSpacing=225.0, Population=250.0)
    fert = find(field, "Fertilise at sowing")
    set_params(fert, Amount=FERT_N[0], FertiliserType="NO3N")
    # second N dose about 45 days after sowing: same code pattern as the sowing rule, keyed on a date
    top = copy.deepcopy(fert)
    top["Name"] = "Fertilise at first irrigation"
    top["CodeArray"] = [l.replace('[EventSubscribe("Sowing")]', '[EventSubscribe("DoManagement")]') for l in fert["CodeArray"]]
    code = "\n".join(top["CodeArray"])
    old = ("            Model crop = sender as Model;\n"
           "            if (Crop != null && crop.Name.ToLower() == (Crop as IModel).Name.ToLower())\n"
           "                Fertiliser.Apply(Amount: Amount, Type: FertiliserType);")
    new = ("            if (Crop != null && Crop.IsAlive && Clock.Today.Day == ApplyDay && Clock.Today.Month == ApplyMonth)\n"
           "                Fertiliser.Apply(Amount: Amount, Type: FertiliserType);")
    if old not in code:
        raise RuntimeError("unexpected 'Fertilise at sowing' code; adjust run_apsim.py to this APSIM version")
    code = code.replace(old, new).replace(
        '        public double Amount { get; set; }',
        '        public double Amount { get; set; }\n'
        '        [Description("Day of month of the application")] public int ApplyDay { get; set; }\n'
        '        [Description("Month of the application")] public int ApplyMonth { get; set; }')
    top["CodeArray"] = code.split("\n")
    d2 = datetime.strptime(dmmm(sow, 45) + "-2001", "%d-%b-%Y")
    top["Parameters"] = top["Parameters"] + [{"Key": "ApplyDay", "Value": str(d2.day)},
                                             {"Key": "ApplyMonth", "Value": str(d2.month)}]
    set_params(top, Amount=FERT_N[1])
    kids = field["Children"]
    kids.insert(kids.index(fert) + 1, top)
    # irrigation (canal rotation: return period 7 days; rainfed sites have irrigation switched off)
    irr = copy.deepcopy(T.irr_mgr)
    set_params(irr, allowIrrigation=str(bool(irrigated)), seasonStart=sow, seasonEnd="15-may", seasonsAllocation=10000,
               triggerDeficit=50, targetDeficit=100, returndays=7, maximumAmount=100, depthPAWC=600)
    kids.append(copy.deepcopy(T.irr_node))
    kids.append(irr)
    if irrigated:
        sow_irr = copy.deepcopy(fert)
        sow_irr["Name"] = "Irrigate at sowing"
        sow_irr["CodeArray"] = SOWING_IRRIGATION_CODE.split("\n")
        sow_irr["Parameters"] = [{"Key": "Crop", "Value": "Wheat"}, {"Key": "Amount", "Value": "75"}]
        kids.append(sow_irr)
    # reports
    rep = find(field, "Report")
    rep["Name"] = "HarvestReport"
    rep["VariableNames"] = ["[Clock].Today", "[Wheat].Grain.Total.Wt*10 as Yield", "[Wheat].AboveGround.Wt*10 as Biomass",
                            "[Wheat].Grain.Number as GrainNumber", "[Wheat].Grain.Size as GrainSize",
                            "[Wheat].DaysAfterSowing as DAS"]
    rep["EventNames"] = ["[Wheat].Harvesting"]
    if daily:
        drep = copy.deepcopy(rep)
        drep["Name"] = "DailyReport"
        drep["VariableNames"] = ["[Clock].Today", "[Wheat].Phenology.Stage as Stage", "[Wheat].DaysAfterSowing as DAS",
                                 "[Wheat].IsAlive as Alive"]
        drep["EventNames"] = ["[Clock].DoReport"]
        kids.insert(kids.index(rep) + 1, drep)
    return sim


def write_apsimx(T, sims, path):
    doc = copy.deepcopy(T.wheat)
    doc["Children"] = sims + [copy.deepcopy(T.datastore)]
    json.dump(doc, open(path, "w", encoding="utf-8"), indent=1)


def run_models(models_cmd, apsimx, log):
    t0 = time.time()
    db = Path(apsimx).with_suffix(".db")
    if db.exists():
        db.unlink()                       # APSIM appends to an existing database; start each run clean
    cmd = models_cmd + [str(apsimx)]
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"  {apsimx.name}: exit {p.returncode} in {(time.time() - t0) / 60:.1f} min (log {log.name})")
    if p.returncode != 0:
        print(open(log).read()[-3000:])
        raise SystemExit(f"APSIM failed for {apsimx}")


def collect(db_path):
    con = sqlite3.connect(db_path)
    sims = pd.read_sql("SELECT ID as SimulationID, Name as sim FROM _Simulations", con)
    harv = pd.read_sql("SELECT * FROM HarvestReport", con).merge(sims, on="SimulationID")
    try:
        daily = pd.read_sql("SELECT SimulationID, [Clock.Today] as date, Stage, DAS FROM DailyReport WHERE Alive = 1", con)
        daily = daily.merge(sims, on="SimulationID")
    except Exception:
        daily = pd.DataFrame()
    con.close()
    harv["date"] = pd.to_datetime(harv["Clock.Today"])
    harv["season"] = harv.date.dt.year
    harv["sow_date"] = harv.date - pd.to_timedelta(harv.DAS, unit="D")
    if len(daily):
        daily["date"] = pd.to_datetime(daily.date)
        daily["season"] = np.where(daily.date.dt.month >= 8, daily.date.dt.year + 1, daily.date.dt.year)
        ph = daily.groupby(["sim", "season"]).apply(lambda g: pd.Series({
            "anthesis_date": g.loc[g.Stage >= STAGE_ANTHESIS, "date"].min(),
            "end_grain_fill_date": g.loc[g.Stage >= STAGE_END_GRAIN_FILL, "date"].min()})).reset_index()
        harv = harv.merge(ph, on=["sim", "season"], how="left")
    return harv


def heat_diagnostics(res, metfiles):
    """Heat exposure in the simulated anthesis (+/-7 d) and grain-filling windows, from the weather the run used."""
    out = []
    cache = {}
    for r in res.itertuples(index=False):
        mf = metfiles[r.sim]
        if mf not in cache:
            _, w = read_met(mf)
            w.index = pd.to_datetime(w.year.astype(str) + w.day.astype(str).str.zfill(3), format="%Y%j")
            cache[mf] = w
        w = cache[mf]
        rec = {"sim": r.sim, "season": r.season}
        if pd.notna(getattr(r, "anthesis_date", np.nan)):
            a = pd.Timestamp(r.anthesis_date)
            win = w.loc[a - pd.Timedelta(days=7): a + pd.Timedelta(days=7)]
            rec["days_tmax_ge34_anthesis"] = int((win.maxt >= 34).sum())
            if pd.notna(r.end_grain_fill_date):
                gf = w.loc[a: pd.Timestamp(r.end_grain_fill_date)]
                rec["tmax_mean_grain_fill"] = float(gf.maxt.mean())
                rec["kdd30_grain_fill"] = float((gf.maxt - 30).clip(lower=0).sum())
                rec["kdd34_grain_fill"] = float((gf.maxt - 34).clip(lower=0).sum())
        out.append(rec)
    return res.merge(pd.DataFrame(out), on=["sim", "season"], how="left")


# ------------------------------------------------------------------------------------------- stages
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apsimx", required=True, help="path to the ApsimX source tree (Examples/ and bin/)")
    ap.add_argument("--models", default=None, help="command to run APSIM, e.g. 'dotnet /path/Models.dll'")
    ap.add_argument("--stage", choices=["smoke", "calibrate", "production"], required=True)
    ap.add_argument("--sites", default=None, help="comma-separated subset of sites")
    a = ap.parse_args()

    models_cmd = a.models.split() if a.models else None
    if models_cmd is None:
        for cand in ("bin/Release/net10.0/Models", "bin/Release/net10.0/Models.exe"):
            if (Path(a.apsimx) / cand).exists():
                models_cmd = [str(Path(a.apsimx) / cand)]
                break
        dll = Path(a.apsimx) / "bin" / "Release" / "net10.0" / "Models.dll"
        if models_cmd is None and dll.exists():
            models_cmd = ["dotnet", str(dll)]
    if models_cmd is None:
        raise SystemExit("Models executable not found; build ApsimX (see README.md) or pass --models")

    for d in (WORK / "met", WORK / "apsimx", OUT):
        d.mkdir(parents=True, exist_ok=True)
    T = Templates(a.apsimx)
    sites = pd.read_csv(INP / "sites.csv")
    if a.sites:
        sites = sites[sites.site.isin(a.sites.split(","))]
    soils = json.load(open(INP / "soils.json"))
    base_co2 = json.load(open(INP / "co2_baseline.json"))["baseline_1991_2020_co2"]

    if a.stage == "smoke":
        s = sites.iloc[0]
        sim = make_simulation(T, "smoke", s, soils[s.site], INP / "met" / f"{s.site}_obs.met", "2016-08-01",
                              "2019-07-31", "Spring Quick-Mid", s.sow, s.irrigated)
        f = WORK / "apsimx" / "smoke.apsimx"
        write_apsimx(T, [sim], f)
        run_models(models_cmd, f, WORK / "apsimx" / "smoke.log")
        res = collect(f.with_suffix(".db"))
        print(res[["sim", "season", "Yield", "Biomass", "DAS", "anthesis_date", "end_grain_fill_date"]].to_string())
        ok = len(res) >= 2 and res.Yield.gt(500).all()
        print("SMOKE TEST", "PASSED" if ok else "CHECK OUTPUT (yields or harvests missing)")
        res.to_csv(OUT / "smoke_results.csv", index=False)
        return

    if a.stage == "calibrate":
        rows = []
        for _, s in sites.iterrows():
            sims, metfiles = [], {}
            for cv in CANDIDATE_CULTIVARS:
                nm = f"{s.site}|cal|{cv}"
                sims.append(make_simulation(T, nm, s, soils[s.site], INP / "met" / f"{s.site}_obs.met",
                                            "1990-08-01", "2020-07-31", cv, s.sow, s.irrigated))
            f = WORK / "apsimx" / f"calibrate_{s.site}.apsimx"
            write_apsimx(T, sims, f)
            run_models(models_cmd, f, f.with_suffix(".log"))
            res = collect(f.with_suffix(".db"))
            res["cultivar"] = res.sim.str.split("|").str[2]
            res["anth_doy"] = pd.to_datetime(res.anthesis_date).dt.dayofyear
            res["mat_doy"] = pd.to_datetime(res.end_grain_fill_date).dt.dayofyear
            g = res.groupby("cultivar")[["anth_doy", "mat_doy", "Yield"]].mean()
            g["error_days"] = (g.anth_doy - s.anth_doy).abs() + (g.mat_doy - s.mat_doy).abs()
            g["site"] = s.site
            rows.append(g.reset_index())
            res.to_csv(OUT / f"calibration_runs_{s.site}.csv", index=False)
        cal = pd.concat(rows, ignore_index=True)
        cal.to_csv(OUT / "calibration_all.csv", index=False)
        best = cal.loc[cal.groupby("site").error_days.idxmin()][["site", "cultivar", "anth_doy", "mat_doy", "error_days"]]
        best.to_csv(OUT / "calibration_selected.csv", index=False)
        print(best.to_string(index=False))
        return

    # production
    cal = pd.read_csv(OUT / "calibration_selected.csv").set_index("site")
    deltas = pd.read_csv(INP / "deltas.csv")
    co2s = pd.read_csv(INP / "co2_scenarios.csv")
    all_res = []
    for _, s in sites.iterrows():
        cv = cal.loc[s.site, "cultivar"]
        sims, metfiles = [], {}

        def add(tag, met, cultivar, sow, start="1990-08-01", end="2020-07-31", daily=True):
            nm = f"{s.site}|{tag}"
            sims.append(make_simulation(T, nm, s, soils[s.site], met, start, end, cultivar, sow, s.irrigated, daily))
            metfiles[nm] = str(met)

        early = dmmm(s.sow, -15)
        # 1. hindcast with observed weather and CO2 (evaluation against district yields)
        add("hist", INP / "met" / f"{s.site}_obs.met", cv, s.sow, "1984-08-01", "2025-07-31")
        # 2. baseline 1991-2020 with constant CO2, and the adaptation options under it
        bm = baseline_met(s.site, base_co2)
        add("base|std", bm, cv, s.sow)
        add("base|early", bm, cv, early)
        add("base|slower", bm, SLOWER[cv], s.sow)
        add("base|early+slower", bm, SLOWER[cv], early)
        dsite = deltas[deltas.site == s.site]
        for (model, gwl), dm in dsite.groupby(["model", "gwl"]):
            co2 = float(co2s[(co2s.model == model) & (co2s.gwl == gwl)].co2.iloc[0])
            add(f"gwl{gwl}|{model}|co2", perturbed_met(s.site, f"{model}_gwl{gwl}_co2", dm, co2), cv, s.sow)
            add(f"gwl{gwl}|{model}|noco2", perturbed_met(s.site, f"{model}_gwl{gwl}_noco2", dm, base_co2), cv, s.sow)
        # 3. adaptation under the ensemble-median change at each warming level (with CO2)
        for gwl, dg in dsite.groupby("gwl"):
            med = dg.groupby("month")[["dtmax", "dtmin", "pr_ratio"]].median().reset_index()
            co2 = float(co2s[co2s.gwl == gwl].co2.median())
            mm = perturbed_met(s.site, f"median_gwl{gwl}", med, co2)
            add(f"gwl{gwl}|median|std", mm, cv, s.sow)
            add(f"gwl{gwl}|median|early", mm, cv, early)
            add(f"gwl{gwl}|median|slower", mm, SLOWER[cv], s.sow)
            add(f"gwl{gwl}|median|early+slower", mm, SLOWER[cv], early)
        f = WORK / "apsimx" / f"production_{s.site}.apsimx"
        write_apsimx(T, sims, f)
        print(f"{s.site}: {len(sims)} simulations")
        run_models(models_cmd, f, f.with_suffix(".log"))
        res = collect(f.with_suffix(".db"))
        res = heat_diagnostics(res, metfiles)
        res.insert(0, "site", s.site)
        res["cultivar"] = cv
        res.to_csv(OUT / f"production_{s.site}.csv", index=False)
        all_res.append(res)
    pd.concat(all_res).to_csv(OUT / "production_all.csv", index=False)
    print("production results written to", OUT)


if __name__ == "__main__":
    main()
