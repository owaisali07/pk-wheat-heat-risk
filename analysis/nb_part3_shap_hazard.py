# %% [markdown]
# ## 4. SHAP attribution and stage-wise heat thresholds (RQ1)
#
# TreeSHAP values of the final XGBoost (and, for robustness, the random forest) decompose each prediction.
# For the stage mean maximum temperature the threshold is the temperature at which the LOWESS-smoothed SHAP
# dependence crosses from positive to negative; its 95 % interval comes from 200 district-bootstrap samples.
# The response above the threshold is the slope of SHAP on temperature (percentage points of yield per degC).
# Accumulated local effects (ALE) of the XGB-RF ensemble give a model-agnostic check.

# %%
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess

expl_x = shap.TreeExplainer(final["XGB"])
sv_x = expl_x.shap_values(Xd)
r_sub = np.random.default_rng(SEED).choice(len(Xd), size=min(1500, len(Xd)), replace=False)
expl_r = shap.TreeExplainer(final["RF"])
sv_r = expl_r.shap_values(Xd[r_sub], check_additivity=False)

imp = pd.DataFrame({"feature": FEATS, "mean_abs_shap_XGB": np.abs(sv_x).mean(0),
                    "mean_abs_shap_RF": np.abs(sv_r).mean(0)}).sort_values("mean_abs_shap_XGB", ascending=False)


def fgroup(f):
    if f.startswith("irr"):
        return "irrigation", "-"
    v, s = f.split("_", 1)
    return v, s


imp["variable"], imp["stage"] = zip(*imp.feature.map(fgroup))
imp.round(5).to_csv(TAB / "T3a_shap_importance.csv", index=False)
print(imp.head(15).round(4).to_string(index=False))
print((imp.groupby("stage").mean_abs_shap_XGB.sum() / imp.mean_abs_shap_XGB.sum()).round(3).to_string())


def zero_cross(x, s, frac=0.4):
    """Lowest x at which the LOWESS curve of s on x turns negative after being positive."""
    lo = lowess(s, x, frac=frac, return_sorted=True)
    xs, ys = lo[:, 0], lo[:, 1]
    for i in range(1, len(xs)):
        if ys[i - 1] > 0 >= ys[i]:
            return xs[i - 1] + (xs[i] - xs[i - 1]) * ys[i - 1] / (ys[i - 1] - ys[i]), lo
    return np.nan, lo


def threshold_table(sv, X, rows_idx, label):
    out = []
    dists = dv.adm2_pcode.values[rows_idx]
    ud = np.unique(dists)
    rb = np.random.default_rng(SEED)
    for s in STAGE_ORDER:
        j = FEATS.index(f"tmax_{s}")
        x, sh = X[:, j], sv[:, j]
        thr, lo = zero_cross(x, sh)
        above = x > thr if np.isfinite(thr) else np.zeros(len(x), bool)
        slope = stats.theilslopes(sh[above], x[above]).slope * 100 if above.sum() > 20 else np.nan
        bt = []
        for _ in range(200):
            pick = rb.choice(ud, size=len(ud), replace=True)
            ix = np.concatenate([np.where(dists == d)[0] for d in pick])
            bt.append(zero_cross(x[ix], sh[ix])[0])
        bt = np.array(bt)
        out.append({"model": label, "stage": s, "threshold_C": thr, "ci_lo": np.nanpercentile(bt, 2.5),
                    "ci_hi": np.nanpercentile(bt, 97.5), "share_boot_crossing": np.isfinite(bt).mean(),
                    "slope_pp_per_C_above": slope, "pct_obs_above": 100 * above.mean(),
                    "tmax_p5": np.percentile(x, 5), "tmax_p95": np.percentile(x, 95)})
    return pd.DataFrame(out)


thr_x = threshold_table(sv_x, Xd, np.arange(len(Xd)), "XGB")
thr_r = threshold_table(sv_r, Xd[r_sub], r_sub, "RF")


def ale_1d(predict, X, j, nbins=20):
    x = X[:, j]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    k = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    eff = np.zeros(len(edges) - 1)
    cnt = np.zeros(len(edges) - 1)
    lo_, hi_ = X.copy(), X.copy()
    lo_[:, j], hi_[:, j] = edges[k], edges[k + 1]
    d = predict(hi_) - predict(lo_)
    for b in range(len(edges) - 1):
        cnt[b] = (k == b).sum()
        eff[b] = d[k == b].mean() if cnt[b] else 0
    ale = np.concatenate([[0], np.cumsum(eff)])
    centers = ale - np.sum((ale[:-1] + ale[1:]) / 2 * cnt) / cnt.sum()
    return edges, centers


ale_res, ale_thr = {}, []
for s in STAGE_ORDER:
    j = FEATS.index(f"tmax_{s}")
    e, a = ale_1d(ens_predict, Xd, j)
    ale_res[s] = (e, a)
    cross = [e[i - 1] + (e[i] - e[i - 1]) * a[i - 1] / (a[i - 1] - a[i]) for i in range(1, len(e))
             if a[i - 1] > 0 >= a[i]]
    ale_thr.append({"stage": s, "ALE_threshold_C": cross[0] if cross else np.nan,
                    "ALE_range_pp": 100 * (a.max() - a.min())})
thr = pd.concat([thr_x, thr_r]).merge(pd.DataFrame(ale_thr), on="stage", how="left")
thr.round(3).to_csv(TAB / "T3_heat_thresholds.csv", index=False)
print(thr.round(2).to_string(index=False))

# %% [markdown]
# ## 5. Heat-attributable losses and hazard
#
# For every district and season the ensemble prediction under observed conditions is compared with the
# prediction in which the heat predictors (stage Tmax, KDD30, KDD34 and the irrigation-heat interaction) are
# set to the district's 1991-2020 climatology (Eq. 3). The hazard H_d is the mean loss over 1991-2020,
# expressed as a fraction of trend yield; it is also converted to tonnes with the 2025 trend yield and the
# 2021-25 mean harvested area.

# %%
risk_d = sorted(dv.adm2_pcode.unique())
base = feat[feat.adm2_pcode.isin(risk_d) & feat.year.between(1991, 2020)].reset_index(drop=True)
clim_heat = base.groupby("adm2_pcode")[HEAT_VARS].transform("mean")


def heat_loss(frame, clim_frame, models=None, heat_vars=None):
    heat_vars = heat_vars or HEAT_VARS
    Xo = frame[FEATS].copy()
    Xc = Xo.copy()
    Xc[heat_vars] = clim_frame[heat_vars].values
    Xc["irr_x_kdd30_late"] = Xc.irr_share * (Xc.kdd30_anth + Xc.kdd30_fill)
    fo, fc = ens_predict(Xo.values, models), ens_predict(Xc.values, models)
    return np.maximum(0.0, fc - fo), fo, fc


L, f_obs, f_clim = heat_loss(base, clim_heat)
base["loss"] = L
hz = base.groupby("adm2_pcode").loss.agg(H="mean", H_p90=lambda v: np.percentile(v, 90)).reset_index()
H_boot = np.vstack([base.assign(l=heat_loss(base, clim_heat, models=m)[0]).groupby("adm2_pcode").l.mean()
                    .reindex(risk_d).values for m in boots])  # B x districts

# convert to tonnes: 2025 trend yield x 2021-25 mean area
ty = tgt.groupby("adm2_pcode").apply(lambda g: g.trend_t_ha[g.year == g.year.max()].iloc[0]
                                     + g.slope_t_ha_yr.iloc[0] * (2025 - g.year.max()))
area_recent = panel[panel.own & panel.year.between(2021, 2025)].groupby("adm2_pcode").area_ha.mean()
area_any = panel[panel.own].groupby("adm2_pcode").area_ha.apply(lambda v: v.tail(5).mean())
hz["trend2025_t_ha"] = hz.adm2_pcode.map(ty)
hz["area_ha"] = hz.adm2_pcode.map(area_recent).fillna(hz.adm2_pcode.map(area_any))
hz["H_t"] = hz.H * hz.trend2025_t_ha * hz.area_ha
hz = hz.merge(study[["adm2_pcode", "adm2_name", "adm1_name"]], on="adm2_pcode")
yr_loss = base.groupby("year").apply(lambda g: np.average(g.loss, weights=g.adm2_pcode.map(hz.set_index("adm2_pcode").area_ha)))
print("national area-weighted mean heat loss 1991-2020 (% of trend yield):", round(100 * yr_loss.mean(), 2))
print("worst seasons:", (100 * yr_loss.sort_values(ascending=False).head(5)).round(2).to_dict())
print("expected annual heat loss (kt):", round(hz.H_t.sum() / 1e3, 1))
print(hz.sort_values("H", ascending=False).head(10)[["adm2_name", "adm1_name", "H", "H_t"]].round(4).to_string(index=False))
