"""Calibrate the temperature response of maximum leaf photosynthesis (TMPFTB) of the WOFOST wheat parameter set.

The European parameter set reduces AMAX from 25 degC and stops photosynthesis at 35 degC, which gives potential
yields below farmers' yields in Pakistan's hot spring. We choose the end of the optimum plateau (P) and the
zero point (Z) so that observed district yields are about 52% of simulated potential yield at the five
irrigated sites (2011-2020), the midpoint of the 48-56% that Khaliq et al. (2019) found for wheat in Punjab.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wofost_pipeline as W  # noqa: E402

TARGET_RATIO = 0.52
YEARS = range(2011, 2021)
obs = pd.read_csv(W.INP / "observed_yield.csv")
obs = obs[(obs.flag != "inherited") & obs.year.isin(YEARS)]


def one(site, P, Z):
    s = W.sites[W.sites.site == site].iloc[0]
    elev, w = W.site_weather(s)
    t1, t2 = W.thermal_calibration(w, s.sow, s.anth_doy, s.mat_doy)
    W.HEAT_TOLERANT_TMPFTB = [0.0, 0.01, 10.0, 0.6, 15.0, 1.0, float(P), 1.0, float(Z), 0.0]
    r = W.run_seasons(w.loc["2009":"2020"], s, elev, s.sow, t1, t2, float(w.loc["2015-01-01", "CO2"]), YEARS, heattol=True)
    return {"site": site, "P": P, "Z": Z, "sim_t_ha": r.Yield.mean() / 1000}


if __name__ == "__main__":
    irrig = list(W.sites[W.sites.irrigated].site)
    grid = list(itertools.product([25, 28, 30, 32], [35, 38, 40, 42, 45]))
    grid = [(p, z) for p, z in grid if z - p >= 6]
    W.crop_data(reload=True)
    for _, s in W.sites.iterrows():
        W.power_wind(s.site, s.lat, s.lon)
    rows = Parallel(n_jobs=W.N_JOBS)(delayed(one)(s, p, z) for s in irrig for p, z in grid)
    df = pd.DataFrame(rows)
    om = obs.groupby("adm2_pcode").yield_t_ha.mean()
    df["obs_t_ha"] = df.site.map(W.sites.set_index("site").pcode).map(om)
    g = df.groupby(["P", "Z"]).apply(lambda d: pd.Series({"ratio": d.obs_t_ha.sum() / d.sim_t_ha.sum(),
                                                         "sim_mean": d.sim_t_ha.mean()})).reset_index()
    g["err"] = (g.ratio - TARGET_RATIO).abs()
    best = g.loc[g.err.idxmin()]
    df.to_csv(W.OUT / "tmpftb_calibration_sites.csv", index=False)
    g.to_csv(W.OUT / "tmpftb_calibration.csv", index=False)
    print(g.round(3).to_string(index=False))
    sel = {"P": float(best.P), "Z": float(best.Z), "ratio_obs_to_sim": round(float(best.ratio), 3),
           "TMPFTB": [0.0, 0.01, 10.0, 0.6, 15.0, 1.0, float(best.P), 1.0, float(best.Z), 0.0],
           "TMPFTB_heat_tolerant": [0.0, 0.01, 10.0, 0.6, 15.0, 1.0, float(best.P) + 3, 1.0, float(best.Z) + 3, 0.0]}
    json.dump(sel, open(W.OUT / "tmpftb_selected.json", "w"), indent=1)
    print("selected", sel)
