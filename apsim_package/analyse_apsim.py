"""Analyse the APSIM-Wheat simulations and compare them with WOFOST (both Stage H).

Outputs: results/tables/T11_apsim_warming_levels.csv, T12_apsim_adaptation.csv, S21_apsim_calibration_evaluation.csv,
         S22_apsim_vs_wofost.csv, figures/Fig11_apsim_vs_wofost.{pdf,png}
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ.get("PAPER8_ROOT", "."))
PKG = ROOT / "apsim_package"
TAB, FIG = ROOT / "results" / "tables", ROOT / "figures"
sites = pd.read_csv(PKG / "inputs" / "sites.csv")
LABEL = dict(zip(sites.site, sites.district))
SHORT = {s: LABEL[s].replace("Shaheed Benazir Abad", "S. B. Abad") for s in LABEL}
ORDER = list(sites.site)
q = lambda a: (lambda x: x.quantile(a))

res = pd.read_csv(PKG / "outputs" / "production_all.csv",
                  parse_dates=["anthesis_date", "end_grain_fill_date", "sow_date"])
parts = res.sim.str.split("|", expand=True)
res["scenario"] = res.sim.str.split("|", n=1).str[1]
res["anth_doy"] = res.anthesis_date.dt.dayofyear
res["gf_days"] = (res.end_grain_fill_date - res.anthesis_date).dt.days
res["Yield_t"] = res.Yield / 1000
print("APSIM seasons:", len(res), "| harvested with yield > 0:", int((res.Yield > 0).sum()))

# ---------------------------------------------------------------- calibration and hindcast
cal = pd.read_csv(PKG / "outputs" / "calibration_selected.csv")
obs = pd.read_csv(PKG / "inputs" / "observed_yield.csv")
obs = obs[obs.flag != "inherited"]


def detrend(x):
    sl, ic, _, _ = stats.theilslopes(x.values, x.index.values)
    tr = ic + sl * x.index.values
    return pd.Series((x.values - tr) / tr, index=x.index)


ev = []
for _, s in sites.iterrows():
    h = res[(res.site == s.site) & (res.scenario == "hist")].groupby("season").Yield_t.mean()
    o = obs[obs.adm2_pcode == s.pcode].groupby("year").yield_t_ha.mean().dropna()
    yrs = sorted(set(h.index) & set(o.index) - {2022})
    r, p = stats.pearsonr(detrend(h.loc[yrs]), detrend(o.loc[yrs]))
    b = res[(res.site == s.site) & (res.scenario == "base|std")]
    ev.append({"site": s.site, "district": s.district, "n_seasons": len(yrs), "r_detrended_anomaly": r, "p": p,
               "sim_anthesis_doy": b.anth_doy.mean(), "target_anthesis_doy": s.anth_doy,
               "sim_maturity_doy": b.end_grain_fill_date.dt.dayofyear.mean(), "target_maturity_doy": s.mat_doy,
               "sim_yield_1991_2020_t_ha": b.Yield_t.mean(), "obs_yield_2011_2020_t_ha": o.loc[2011:2020].mean(),
               "seasons_harvested_base": int(b.Yield.notna().sum())})
ev = pd.DataFrame(ev).merge(cal[["site", "cultivar"]], on="site")
ev.round(3).to_csv(TAB / "S21_apsim_calibration_evaluation.csv", index=False)
print(ev.round(2).to_string(index=False))

# ---------------------------------------------------------------- warming levels
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
summ = g.groupby(["site", "gwl", "co2"]).agg(
    dYield_median=("dYield_pct", "median"), dYield_p10=("dYield_pct", q(0.1)), dYield_p90=("dYield_pct", q(0.9)),
    dAnthesis_days=("dAnthesis_days", "median"), dGrainFill_days=("dGrainFill_days", "median"),
    heat_days_anthesis=("days_tmax_ge34_anthesis", "median"), kdd34_grain_fill=("kdd34_grain_fill", "median"),
    n_gcm=("model", "nunique")).reset_index()
summ["district"] = summ.site.map(LABEL)
summ = summ.merge(base[["days_tmax_ge34_anthesis", "kdd34_grain_fill"]].rename(columns=lambda c: c + "_base"),
                  left_on="site", right_index=True)
summ.round(2).to_csv(TAB / "T11_apsim_warming_levels.csv", index=False)
print(summ[["district", "gwl", "co2", "dYield_median", "dYield_p10", "dYield_p90", "dAnthesis_days", "dGrainFill_days",
            "heat_days_anthesis", "days_tmax_ge34_anthesis_base"]].round(1).to_string(index=False))

# ---------------------------------------------------------------- adaptation
ad = res[res.scenario.str.contains(r"median\||^base\|", regex=True)].copy()
ad["gwl"] = np.where(ad.scenario.str.startswith("base"), 0.0,
                     pd.to_numeric(ad.scenario.str.extract(r"gwl([\d.]+)")[0], errors="coerce"))
ad["option"] = ad.scenario.str.split("|").str[-1]
a = ad.groupby(["site", "gwl", "option"]).Yield.mean().unstack("option")
ref = res[res.scenario == "base|std"].groupby("site").Yield.mean()
adapt = (100 * (a.div(ref, axis=0, level="site") - 1)).round(1).reset_index()
adapt["district"] = adapt.site.map(LABEL)
adapt.to_csv(TAB / "T12_apsim_adaptation.csv", index=False)
print(adapt.to_string(index=False))

# ---------------------------------------------------------------- comparison with WOFOST
wo = pd.read_csv(TAB / "T9_crop_model_warming_levels.csv")
wa = pd.read_csv(TAB / "T10_crop_model_adaptation.csv")
cmp = summ[["site", "gwl", "co2", "dYield_median", "dAnthesis_days"]].merge(
    wo[["site", "gwl", "co2", "dYield_median", "dAnthesis_days"]], on=["site", "gwl", "co2"], suffixes=("_apsim", "_wofost"))
cmp.round(2).to_csv(TAB / "S22_apsim_vs_wofost.csv", index=False)
for co2 in ("noco2", "co2"):
    d = cmp[cmp.co2 == co2]
    r_ = stats.spearmanr(d.dYield_median_apsim, d.dYield_median_wofost)
    print(f"APSIM vs WOFOST yield change ({co2}): Spearman {r_.correlation:.2f} (p {r_.pvalue:.3f}); "
          f"median APSIM {d.dYield_median_apsim.median():.1f} vs WOFOST {d.dYield_median_wofost.median():.1f}")
r_ = stats.spearmanr(cmp.dAnthesis_days_apsim, cmp.dAnthesis_days_wofost)
print(f"anthesis shift: Spearman {r_.correlation:.2f}")
hz = pd.read_csv(TAB / "district_hazard.csv").set_index("adm2_pcode")
hh = []
for _, s in sites.iterrows():
    for gwl in (1.5, 2.0, 3.0):
        row = summ[(summ.site == s.site) & (summ.gwl == gwl) & (summ.co2 == "noco2")].iloc[0]
        hh.append((row.dYield_median, -100 * (hz.loc[s.pcode, f"H_{gwl}"] - hz.loc[s.pcode, "H"])))
hh = np.array(hh)
r_ = stats.spearmanr(hh[:, 0], hh[:, 1])
print(f"APSIM (no CO2) vs hazard model: Spearman {r_.correlation:.2f} (p {r_.pvalue:.3f})")

# ---------------------------------------------------------------- figure 11: two crop models
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7,
                     "axes.linewidth": 0.5})
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.6), constrained_layout=True)
x = np.arange(len(ORDER))
for ax, co2, ttl in ((axs[0], "noco2", "(a) Yield change at GWL 3 °C, climate only"),
                     (axs[1], "co2", "(b) Yield change at GWL 3 °C, with CO$_2$")):
    for k, (tab, lab, col, off) in enumerate(((summ, "APSIM-Wheat", "#1f618d", -0.17), (wo, "WOFOST", "#d35400", 0.17))):
        d = tab[(tab.gwl == 3.0) & (tab.co2 == co2)].set_index("site").reindex(ORDER)
        ax.errorbar(x + off, d.dYield_median, yerr=[d.dYield_median - d.dYield_p10, d.dYield_p90 - d.dYield_median],
                    fmt="o", ms=3.5, color=col, elinewidth=0.8, capsize=2, label=lab)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[s] for s in ORDER], rotation=40, ha="right")
    ax.set_ylabel("% of 1991–2020 simulated yield")
    ax.set_title(ttl, loc="left", fontsize=7.2)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.2 * (hi - lo))
    ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=6)
for tab, lab, col, mk in ((summ, "APSIM-Wheat", "#1f618d", "o"), (wo, "WOFOST", "#d35400", "s")):
    for gwl, alpha in ((1.5, 0.35), (2.0, 0.65), (3.0, 1.0)):
        d = tab[(tab.gwl == gwl) & (tab.co2 == "co2")].set_index("site").reindex(ORDER)
        axs[2].scatter(x + (-0.17 if lab.startswith("APSIM") else 0.17), d.dAnthesis_days, color=col, alpha=alpha,
                       marker=mk, s=14, label=f"{lab}" if gwl == 3.0 else None)
axs[2].axhline(0, color="k", lw=0.4)
axs[2].set_xticks(x)
axs[2].set_xticklabels([SHORT[s] for s in ORDER], rotation=40, ha="right")
axs[2].set_ylabel("Shift in anthesis (days)")
axs[2].set_title("(c) Anthesis shift (light to dark: 1.5, 2, 3 °C)", loc="left", fontsize=7.2)
lo, hi = axs[2].get_ylim()
axs[2].set_ylim(lo, hi + 0.25 * (hi - lo))
axs[2].legend(frameon=False, ncol=2, loc="upper center", fontsize=6)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"Fig11_apsim_vs_wofost.{ext}", dpi=300)

_ED = Path(os.path.join(os.environ.get("PAPER8_ROOT", "."), r"figures/editable"))
_ED.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
for _ext, _kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600}),
                  ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
    fig.savefig(_ED / f"Fig11_apsim_vs_wofost.{_ext}", **_kw)
print("APSIM tables and Fig11 written")
