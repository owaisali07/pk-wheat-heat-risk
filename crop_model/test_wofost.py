"""Quick test: one site, five seasons, baseline weather."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wofost_pipeline as W  # noqa: E402

site = sys.argv[1] if len(sys.argv) > 1 else "Faisalabad"
s = W.sites[W.sites.site == site].iloc[0]
elev, w = W.site_weather(s)
t1, t2 = W.thermal_calibration(w, s.sow, s.anth_doy, s.mat_doy)
print(site, "elev", elev, "TSUM1 %.0f TSUM2 %.0f" % (t1, t2), "soil", W.soil_params(W.soils[site]))
t0 = time.time()
r = W.task(s, elev, w.loc["1990":"2020"], "test", s.sow, t1, t2, W.BASE_CO2, range(2011, 2016))
print(r.drop(columns=["site"]).to_string())
print("seconds per season: %.2f" % ((time.time() - t0) / 5))
r2 = W.task(s, elev, w.loc["1990":"2020"], "test650", s.sow, t1, t2, 650.0, range(2011, 2016))
print("CO2 650 vs baseline yield ratio: %.3f" % (r2.Yield.mean() / r.Yield.mean()))
