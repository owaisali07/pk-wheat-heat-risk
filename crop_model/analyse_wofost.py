"""Tables and figure from the WOFOST simulations (Stage H) for the manuscript.

Outputs: results/tables/T9_crop_model_warming_levels.csv, T10_crop_model_adaptation.csv,
         S16_crop_model_calibration_evaluation.csv, S17_crop_model_vs_hazard.csv, figures/Fig10_crop_model.{pdf,png}
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

ROOT = Path(os.environ.get("PAPER8_ROOT", "."))
CM = ROOT / "crop_model"
TAB, FIG = ROOT / "results" / "tables", ROOT / "figures"
sites = pd.read_csv(ROOT / "apsim_package" / "inputs" / "sites.csv")
LABEL = dict(zip(sites.site, sites.district))
SHORT = {s: LABEL[s].replace("Shaheed Benazir Abad", "S. B. Abad") for s in LABEL}
ORDER = list(sites.site)

res = pd.read_csv(CM / "outputs" / "wofost_results.csv", parse_dates=["anthesis_date", "maturity_date", "sow_date"])
n_fail = int(res["error"].notna().sum()) if "error" in res else 0
res = res[res.Yield.notna()].copy()
res["anth_doy"] = res.anthesis_date.dt.dayofyear
res["gf_days"] = (res.maturity_date - res.anthesis_date).dt.days
res["Yield_t"] = res.Yield / 1000
print("seasons:", len(res), "failed:", n_fail)

# ---------------------------------------------------------------- 1. calibration and hindcast
cal = pd.read_csv(CM / "outputs" / "calibration.csv")
obs = pd.read_csv(ROOT / "apsim_package" / "inputs" / "observed_yield.csv")
obs = obs[obs.flag != "inherited"]


def detrend(x):
    sl, ic, _, _ = stats.theilslopes(x.values, x.index.values)
    tr = ic + sl * x.index.values
    return pd.Series((x.values - tr) / tr, index=x.index)


ev = []
for _, s in sites.iterrows():
    h = res[(res.site == s.site) & (res.scenario == "hist")].set_index("season")
    o = obs[obs.adm2_pcode == s.pcode].groupby("year").yield_t_ha.mean().dropna()
    yrs = sorted(set(h.index) & set(o.index) - {2022})
    sim, ob = h.loc[yrs, "Yield_t"], o.loc[yrs]
    r, p = stats.pearsonr(detrend(sim), detrend(ob))
    b = res[(res.site == s.site) & (res.scenario == "base|std")]
    ev.append({"site": s.site, "district": s.district, "n_seasons": len(yrs), "r_detrended_anomaly": r, "p": p,
               "sim_anthesis_doy": b.anth_doy.mean(), "target_anthesis_doy": s.anth_doy,
               "sim_grain_fill_days": b.gf_days.mean(), "sim_yield_1991_2020_t_ha": b.Yield_t.mean(),
               "obs_yield_2011_2020_t_ha": o.loc[2011:2020].mean()})
ev = pd.DataFrame(ev).merge(cal[["site", "TSUM1", "TSUM2"]], on="site")
ev.round(3).to_csv(TAB / "S16_crop_model_calibration_evaluation.csv", index=False)
print(ev.round(2).to_string(index=False))

# ---------------------------------------------------------------- 2. warming levels (31 GCMs)
base = res[res.scenario == "base|std"].groupby("site")[["Yield", "anth_doy", "gf_days", "days_tmax_ge34_anthesis",
                                                        "kdd34_grain_fill"]].mean()
f = res[res.scenario.str.match(r"gwl[\d.]+\|(?!median)")].copy()
p3 = f.scenario.str.split("|", expand=True)
f["gwl"], f["model"], f["co2"] = p3[0].str[3:].astype(float), p3[1], p3[2]
g = f.groupby(["site", "gwl", "co2", "model"])[["Yield", "anth_doy", "gf_days", "days_tmax_ge34_anthesis",
                                                 "kdd34_grain_fill"]].mean().reset_index()
g["dYield_pct"] = 100 * (g.Yield / g.site.map(base.Yield) - 1)
g["dAnthesis_days"] = g.anth_doy - g.site.map(base.anth_doy)
g["dGrainFill_days"] = g.gf_days - g.site.map(base.gf_days)
q = lambda a: (lambda x: x.quantile(a))
summ = g.groupby(["site", "gwl", "co2"]).agg(
    dYield_median=("dYield_pct", "median"), dYield_p10=("dYield_pct", q(0.1)), dYield_p90=("dYield_pct", q(0.9)),
    dAnthesis_days=("dAnthesis_days", "median"), dGrainFill_days=("dGrainFill_days", "median"),
    heat_days_anthesis=("days_tmax_ge34_anthesis", "median"), kdd34_grain_fill=("kdd34_grain_fill", "median"),
    n_gcm=("model", "nunique")).reset_index()
summ["district"] = summ.site.map(LABEL)
bh = base[["days_tmax_ge34_anthesis", "kdd34_grain_fill"]].rename(columns=lambda c: c + "_base")
summ = summ.merge(bh, left_on="site", right_index=True)
summ.round(2).to_csv(TAB / "T9_crop_model_warming_levels.csv", index=False)
print(summ[["district", "gwl", "co2", "dYield_median", "dYield_p10", "dYield_p90", "dAnthesis_days", "dGrainFill_days",
            "heat_days_anthesis"]].round(1).to_string(index=False))

# all-site medians (equal site weights) for the text
allsite = g.groupby(["gwl", "co2", "model"]).dYield_pct.mean().groupby(["gwl", "co2"]).agg(
    median="median", p10=q(0.1), p90=q(0.9)).round(1)
allsite.to_csv(TAB / "T9b_crop_model_all_sites.csv")
print(allsite)

# ---------------------------------------------------------------- 3. adaptation (ensemble-median change, CO2)
ad = res[res.scenario.str.contains(r"\|median\||^base\|", regex=True)].copy()
ad["gwl"] = np.where(ad.scenario.str.startswith("base"), 0.0,
                     pd.to_numeric(ad.scenario.str.extract(r"gwl([\d.]+)")[0], errors="coerce"))
ad["option"] = ad.scenario.str.split("|").str[-1]
a = ad.groupby(["site", "gwl", "option"]).Yield.mean().unstack("option")
ref = res[res.scenario == "base|std"].groupby("site").Yield.mean()
adapt = (100 * (a.div(ref, axis=0, level="site") - 1)).round(1).reset_index()
adapt["district"] = adapt.site.map(LABEL)
adapt.to_csv(TAB / "T10_crop_model_adaptation.csv", index=False)
print(adapt.to_string(index=False))

# ---------------------------------------------------------------- 4. comparison with the hazard model
hz = pd.read_csv(TAB / "district_hazard.csv").set_index("adm2_pcode")
cmp = []
for _, s in sites.iterrows():
    for gwl in (1.5, 2.0, 3.0):
        r_ = summ[(summ.site == s.site) & (summ.gwl == gwl) & (summ.co2 == "noco2")].iloc[0]
        cmp.append({"site": s.site, "gwl": gwl, "crop_model_dYield_noco2_pct": r_.dYield_median,
                    "hazard_model_extra_heat_loss_pct": -100 * (hz.loc[s.pcode, f"H_{gwl}"] - hz.loc[s.pcode, "H"])})
cmp = pd.DataFrame(cmp)
rho = stats.spearmanr(cmp.crop_model_dYield_noco2_pct, cmp.hazard_model_extra_heat_loss_pct)
cmp.attrs["rho"] = rho.correlation
cmp.round(3).to_csv(TAB / "S17_crop_model_vs_hazard.csv", index=False)
print("crop model (no CO2) vs hazard model: Spearman %.2f (p %.3f)" % (rho.correlation, rho.pvalue))

# ---------------------------------------------------------------- figure 10
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7,
                     "axes.linewidth": 0.5})
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.7), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
cols = {"noco2": ["#f5b7b1", "#e74c3c", "#922b21"], "co2": ["#aed6f1", "#3498db", "#1b4f72"]}
for ax, co2, ttl in ((axs[0], "noco2", "(a) Yield change, climate only"), (axs[1], "co2", "(b) Yield change, with CO$_2$")):
    for i, gwl in enumerate((1.5, 2.0, 3.0)):
        d = [g[(g.site == s) & (g.gwl == gwl) & (g.co2 == co2)].dYield_pct.values for s in ORDER]
        ax.boxplot(d, positions=np.arange(len(ORDER)) * 4 + i, widths=0.8, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=cols[co2][i], lw=0.4), medianprops=dict(color="k", lw=0.6),
                   whiskerprops=dict(lw=0.4), capprops=dict(lw=0.4))
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(np.arange(len(ORDER)) * 4 + 1)
    ax.set_xticklabels([SHORT[s] for s in ORDER], rotation=40, ha="right")
    ax.set_ylabel("% of 1991–2020 simulated yield")
    ax.set_title(ttl, loc="left", fontsize=7.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))
    ax.legend(handles=[Patch(fc=c, label=f"GWL {k} °C") for c, k in zip(cols[co2], ("1.5", "2", "3"))],
              frameon=False, ncol=3, loc="upper center", fontsize=6, handlelength=1.2, columnspacing=0.8)
x = adapt[adapt.gwl == 3.0].set_index("site").reindex(ORDER)
opts = [("early", "#27ae60", "Sowing 15 d earlier"), ("longer", "#8e44ad", "Longer duration"),
        ("heattol", "#f39c12", "Heat-tolerant")]
w_ = 0.26
for j, (o_, c, lab) in enumerate(opts):
    axs[2].bar(np.arange(len(ORDER)) + (j - 1) * w_, x[o_] - x["std"], width=w_, color=c, label=lab)
axs[2].axhline(0, color="k", lw=0.4)
axs[2].set_xticks(np.arange(len(ORDER)))
axs[2].set_xticklabels([SHORT[s] for s in ORDER], rotation=40, ha="right")
axs[2].set_ylabel("Gain over standard practice (pp)")
axs[2].set_title("(c) Adaptation, GWL 3 °C", loc="left", fontsize=7.5)
lo, hi = axs[2].get_ylim()
axs[2].set_ylim(lo, hi + 0.35 * (hi - lo))
axs[2].legend(frameon=False, fontsize=6, loc="upper center", ncol=1, handlelength=1.2)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"Fig10_crop_model.{ext}", dpi=300)

_ED = Path(os.path.join(os.environ.get("PAPER8_ROOT", "."), r"figures/editable"))
_ED.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
for _ext, _kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600}),
                  ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
    fig.savefig(_ED / f"Fig10_crop_model.{_ext}", **_kw)
print("written tables and Fig10_crop_model")
