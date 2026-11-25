"""Offline check: build the smoke-test and one production .apsimx without running APSIM.

Verifies that the templates are found, that every edited manager script still contains the expected code, that
soil arrays have consistent lengths, and that the scenario weather files can be generated.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_apsim as R  # noqa: E402

apsimx_root = sys.argv[1]
T = R.Templates(apsimx_root)
sites = pd.read_csv(R.INP / "sites.csv")
soils = json.load(open(R.INP / "soils.json"))
for d in (R.WORK / "met", R.WORK / "apsimx"):
    d.mkdir(parents=True, exist_ok=True)
s = sites.iloc[0]
sim = R.make_simulation(T, "check", s, soils[s.site], R.INP / "met" / f"{s.site}_obs.met", "2016-08-01",
                        "2019-07-31", "Spring Quick-Mid", s.sow, s.irrigated)
R.write_apsimx(T, [sim], R.WORK / "apsimx" / "check.apsimx")
doc = json.load(open(R.WORK / "apsimx" / "check.apsimx", encoding="utf-8"))
field = R.find(doc, "Field")
names = [c["Name"] for c in field["Children"]]
print("Field children:", names)
soil = R.find_type(field, "Models.Soils.Soil, ")[0]
lens = {c["Name"]: len(c["Thickness"]) for c in soil["Children"] if c.get("Thickness")}
print("soil layer counts:", lens)
assert len(set(lens.values())) == 1
phys = R.find_type(soil, "Models.Soils.Physical")[0]
for k in ("BD", "AirDry", "LL15", "DUL", "SAT", "KS"):
    assert len(phys[k]) == len(phys["Thickness"]), k
assert all(l < d < sa for l, d, sa in zip(phys["LL15"], phys["DUL"], phys["SAT"]))
top = R.find(field, "Fertilise at first irrigation")
code = "\n".join(top["CodeArray"])
assert "Clock.Today.Day == ApplyDay" in code and 'EventSubscribe("DoManagement")' in code
print("top-dressing parameters:", top["Parameters"])
print("sowing parameters:", R.find(field, "Sow on a fixed date")["Parameters"])
print("irrigation parameters:", R.find(field, "AutomaticIrrigation")["Parameters"])
print("weather file:", R.find(doc, "Weather")["FileName"])
# one perturbed weather file
dl = pd.read_csv(R.INP / "deltas.csv")
dm = dl[(dl.site == s.site) & (dl.model == dl.model.iloc[0]) & (dl.gwl == 3.0)]
p = R.perturbed_met(s.site, "check_gwl3", dm, 650.0)
h, w = R.read_met(p)
_, w0 = R.read_met(R.INP / "met" / f"{s.site}_obs.met")
w0 = w0[w0.year.between(1990, 2020)]
print("perturbed met:", p.name, "rows", len(w), "| mean Tmax change (all days): %.2f" % (w.maxt.mean() - w0.maxt.mean()),
      "| CO2", w.co2.unique())
print("".join(h))
print("OFFLINE BUILD CHECK PASSED")
