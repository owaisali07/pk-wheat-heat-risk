"""Diagnose a failing APSIM season with a detailed daily report."""
import json
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_apsim as R  # noqa: E402

APSIMX = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"tools/ApsimX")
T = R.Templates(APSIMX)
sites = pd.read_csv(R.INP / "sites.csv")
soils = json.load(open(R.INP / "soils.json"))
s = sites[sites.site == "Sanghar"].iloc[0]
base_co2 = json.load(open(R.INP / "co2_baseline.json"))["baseline_1991_2020_co2"]
met = R.baseline_met(s.site, base_co2)
sim = R.make_simulation(T, "debug", s, soils[s.site], met, "1990-08-01", "1996-07-31", "Spring Slow",
                        R.dmmm(s.sow, -15), s.irrigated)  # early sowing
field = R.find(sim, "Field")
drep = R.find(field, "DailyReport")
drep["VariableNames"] = ["[Clock].Today", "[Wheat].Phenology.Stage as Stage", "[Wheat].Leaf.LAI as LAI",
                         "[Wheat].AboveGround.Wt as AGB", "[Wheat].Leaf.Live.Wt as LeafWt",
                         "sum([Soil].SoilWater.ESW) as ESW", "sum([Soil].Nutrient.NO3.kgha) as NO3",
                         "[Irrigation].IrrigationApplied as Irr", "[Weather].MaxT as MaxT", "[Weather].MinT as MinT",
                         "[Wheat].Leaf.Fw as Fw", "[Wheat].Leaf.Fn as Fn", "[Wheat].IsAlive as Alive"]
f = R.WORK / "apsimx" / "debug.apsimx"
R.write_apsimx(T, [sim], f)
os.environ["DOTNET_ROOT"] = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"tools\dotnet")
R.run_models([str(Path(APSIMX) / "bin/Release/net10.0/Models.exe")], f, f.with_suffix(".log"))
c = sqlite3.connect(f.with_suffix(".db"))
d = pd.read_sql("select * from DailyReport", c)
d["Clock.Today"] = pd.to_datetime(d["Clock.Today"])
for a, b in (("1993-10-24", "1994-01-20"), ("1994-10-24", "1995-01-20")):
    x = d[(d["Clock.Today"] >= a) & (d["Clock.Today"] <= b)]
    print(x.iloc[::6].drop(columns=["CheckpointID", "SimulationID", "Zone"]).round(3).to_string())
print(pd.read_sql("select * from HarvestReport", c)[["Clock.Today", "Yield", "Biomass"]])
