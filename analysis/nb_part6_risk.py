# %% [markdown]
# ## 7. Livelihood risk with Monte Carlo uncertainty (Stage D)
#
# R_d = H~_d * E~_d * V~_d (Eq. 5). Hazard is rescaled with the present-climate range so that future values can
# exceed 1. Each of the 10,000 draws samples (i) one bootstrap refit of the hazard model, (ii) vulnerability
# weights from a Dirichlet distribution centred on the PCA weights (concentration 200), and (iii) for warming
# levels, one CMIP6 model (its hazard scaled by the bootstrap-to-central ratio of the draw).

# %%
NDRAW = 10_000
rng = np.random.default_rng(SEED)
R = R.loc[risk_d]
hzi = hz.set_index("adm2_pcode").loc[risk_d]
H0 = hzi.H.values
hmin, hmax = H0.min(), H0.max()
Hs = lambda h: (h - hmin) / (hmax - hmin)
Et = R.E.values
Zv = Z.loc[risk_d].values
ratio = np.where(H0 > 0, H_boot / np.where(H0 > 0, H0, 1), 1.0)  # B x d
Hf_arr = {g: Hf[Hf.gwl == g].pivot(index="model", columns="adm2_pcode", values="H")[risk_d].values for g in GWLS}


def vt_draw(w):
    v = Zv @ w
    return (v - v.min()) / (v.max() - v.min())


def mc(period):
    out = np.empty((NDRAW, len(risk_d)))
    for i in range(NDRAW):
        b = rng.integers(len(boots))
        w = rng.dirichlet(w_pca.values * 200)
        h = H_boot[b] if period == "present" else Hf_arr[period][rng.integers(len(Hf_arr[period]))] * ratio[b]
        out[i] = Hs(h) * Et * vt_draw(w)
    return out


mc_res = {p: mc(p) for p in ["present"] + GWLS}
riskdf = pd.DataFrame(index=risk_d)
for p, arr in mc_res.items():
    tag = "now" if p == "present" else f"{p}"
    rk = (-arr).argsort(1).argsort(1) + 1  # rank 1 = highest risk
    riskdf[f"R_{tag}"] = np.median(arr, 0)
    riskdf[f"R_{tag}_p05"], riskdf[f"R_{tag}_p95"] = np.percentile(arr, 5, 0), np.percentile(arr, 95, 0)
    riskdf[f"rank_{tag}"] = pd.Series(np.median(arr, 0), index=risk_d).rank(ascending=False).values
    riskdf[f"rank_{tag}_lo"], riskdf[f"rank_{tag}_hi"] = np.percentile(rk, 5, 0), np.percentile(rk, 95, 0)
riskdf["H_t"], riskdf["E_t"], riskdf["V_t"] = Hs(H0), Et, R.V.values
riskdf["R_point"] = riskdf.H_t * riskdf.E_t * riskdf.V_t
riskdf = riskdf.join(study.set_index("adm2_pcode")[["adm2_name", "adm1_name"]])
riskdf.index.name = "adm2_pcode"
riskdf.round(5).to_csv(TAB / "district_risk.csv")
print("rank interval width (median):", np.median(riskdf.rank_now_hi - riskdf.rank_now_lo))
print(riskdf.sort_values("R_now", ascending=False).head(15)[
    ["adm2_name", "adm1_name", "R_now", "rank_now_lo", "rank_now_hi", "H_t", "E_t", "V_t"]].round(3).to_string())
print("risk by province (median of district medians):\n", riskdf.groupby("adm1_name")[["R_now", "R_2.0", "R_3.0"]].median().round(3))

# %% [markdown]
# ## 8. Hotspots and inequality (Stage E)
#
# Local Moran's I (Eq. 6) of median present risk with queen contiguity among the risk districts, 9,999
# conditional permutations and Benjamini-Hochberg false-discovery-rate control (q = 0.05). Inequality of heat
# losses across the wealth distribution is measured with the concentration index (Eq. 7), districts ranked by
# the mean Relative Wealth Index (poorest first, so C < 0 means losses concentrate among poorer districts), and
# with the share of losses borne by the poorest 20 % of districts.

# %%
import esda
from libpysal.weights import Queen

gd_risk = study.set_index("adm2_pcode").loc[risk_d].reset_index()
wq = Queen.from_dataframe(gd_risk, use_index=False, silence_warnings=True)
wq.transform = "r"
yv = riskdf.loc[risk_d, "R_now"].values
gm = esda.Moran(yv, wq, permutations=9999)
lm = esda.Moran_Local(yv, wq, permutations=9999, seed=SEED)
qcut = esda.fdr(lm.p_sim, 0.05)
sig = lm.p_sim <= qcut
quad = np.array(["ns"] * len(yv), dtype=object)
labels_q = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
quad[sig] = [labels_q[q] for q in lm.q[sig]]
riskdf.loc[risk_d, "lisa"] = quad
riskdf.loc[risk_d, "lisa_p"] = lm.p_sim
print(f"global Moran's I = {gm.I:.3f} (p = {gm.p_sim:.4f}); islands: {wq.islands}; FDR p-cut = {qcut:.4f}")
print(pd.Series(quad).value_counts().to_string())
print("High-High districts:", riskdf.loc[riskdf.lisa == "High-High", "adm2_name"].tolist())

wealth = R.rwi.reindex(risk_d)
order_w = wealth.rank(method="first").values
frac_rank = (order_w - 0.5) / len(order_w)


def conc_index(loss):
    mu = loss.mean()
    return 2 / mu * np.cov(loss, frac_rank, bias=True)[0, 1] if mu > 0 else np.nan


def poorest20(loss):
    return loss[frac_rank <= 0.2].sum() / loss.sum()


def conc_curve(loss):
    o = np.argsort(frac_rank)
    return np.r_[0, np.arange(1, len(o) + 1) / len(o)], np.r_[0, np.cumsum(loss[o]) / loss.sum()]


Ht = hzi.H_t.values
ineq_rows, curves = [], {}
rb = np.random.default_rng(SEED)
for p in ["present"] + GWLS:
    for metric, vec in [("relative loss (% of trend yield)", H0 if p == "present" else hz.set_index("adm2_pcode").loc[risk_d, f"H_{p}"].values),
                        ("absolute loss (t)", Ht if p == "present" else hz.set_index("adm2_pcode").loc[risk_d, f"H_t_{p}"].values)]:
        ci = conc_index(vec)
        boot_ci = []
        for _ in range(1000):
            ix = rb.integers(0, len(vec), len(vec))
            mu = vec[ix].mean()
            boot_ci.append(2 / mu * np.cov(vec[ix], frac_rank[ix], bias=True)[0, 1])
        gcm_ci = []
        if p != "present":
            arr = Hf_arr[p] if metric.startswith("relative") else Hf_arr[p] * (hzi.trend2025_t_ha * hzi.area_ha).values
            gcm_ci = [conc_index(a) for a in arr]
        ineq_rows.append({"period": "1991-2020" if p == "present" else f"+{p} degC", "metric": metric, "C": ci,
                          "C_boot_lo": np.percentile(boot_ci, 2.5), "C_boot_hi": np.percentile(boot_ci, 97.5),
                          "C_gcm_p10": np.percentile(gcm_ci, 10) if gcm_ci else np.nan,
                          "C_gcm_p90": np.percentile(gcm_ci, 90) if gcm_ci else np.nan,
                          "share_poorest20": poorest20(vec), "share_richest20": vec[frac_rank > 0.8].sum() / vec.sum()})
        curves[(p, metric)] = conc_curve(vec)
ineq = pd.DataFrame(ineq_rows)
ineq.round(4).to_csv(TAB / "T6_inequality.csv", index=False)
print(ineq.round(3).to_string(index=False))
# risk and vulnerability are also checked against wealth
print("Spearman R_now vs RWI:", stats.spearmanr(riskdf.loc[risk_d, "R_now"], wealth))

# %% [markdown]
# ## 9. Explainable decision support (Stage F)
#
# A gradient-boosted surrogate is fitted to the median present risk using the rescaled hazard, exposure and
# the eight oriented vulnerability indicators; TreeSHAP decomposes each district's risk. The indicator with the
# largest positive contribution is the dominant driver, mapped to a policy lever.

# %%
SUR = pd.DataFrame({"Heat hazard": Hs(H0), "Exposure": Et}, index=risk_d).join(
    Z.loc[risk_d].rename(columns=VLAB))
sur = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=2, min_child_weight=3, subsample=0.8,
                       colsample_bytree=0.8, reg_lambda=5, random_state=SEED, n_jobs=N_JOBS)
sur.fit(SUR.values, riskdf.loc[risk_d, "R_now"].values)
sur_r2 = r2_score(riskdf.loc[risk_d, "R_now"], sur.predict(SUR.values))
sur_cv = np.zeros(len(SUR))
for a, b in GroupKFold(n_splits=5).split(SUR, groups=block.reindex(risk_d).fillna(-1).values):
    sur_cv[b] = xgb.XGBRegressor(**sur.get_params()).fit(SUR.values[a], riskdf.loc[risk_d, "R_now"].values[a]).predict(SUR.values[b])
sur_r2_cv = r2_score(riskdf.loc[risk_d, "R_now"], sur_cv)
print(f"surrogate R2 in-sample {sur_r2:.3f}; spatial-block CV R2 {sur_r2_cv:.3f}")
sv_sur = pd.DataFrame(shap.TreeExplainer(sur).shap_values(SUR.values), index=risk_d, columns=SUR.columns)
LEVER = {"Heat hazard": "Heat-tolerant varieties and earlier sowing",
         "Rainfed cultivated area": "Supplementary irrigation and water harvesting",
         "Exposure": "Livelihood and crop diversification",
         "Poverty headcount": "Social protection and crop insurance", "Poverty intensity": "Social protection and crop insurance",
         "Housing deprivation": "Social protection and crop insurance", "Rural illiteracy": "Extension and advisory services",
         "Farms < 5 acres": "Crop insurance and smallholder credit", "Tenant-operated farms": "Crop insurance and smallholder credit",
         "Travel time to city": "Market access and extension"}
riskdf.loc[risk_d, "driver"] = sv_sur.idxmax(axis=1)
riskdf.loc[risk_d, "lever"] = riskdf.loc[risk_d, "driver"].map(LEVER)
top_tercile = riskdf.R_now >= riskdf.R_now.quantile(2 / 3)
typ = (riskdf[top_tercile].groupby("lever").agg(n_districts=("adm2_name", "size"),
                                                 districts=("adm2_name", lambda s: ", ".join(sorted(s))))
       .sort_values("n_districts", ascending=False))
typ.to_csv(TAB / "T7_policy_typology_high_risk.csv")
print(riskdf.driver.value_counts().to_string())
print(typ.to_string())
shap_sur_mean = sv_sur.abs().mean().sort_values(ascending=False)
print("mean |SHAP| of surrogate:\n", shap_sur_mean.round(4).to_string())

# %% [markdown]
# ## 10. Out-of-sample test: the 2021-22 season and the March-April 2022 heatwave (Stage G)
#
# The 2021-22 season was withheld from detrending, tuning and fitting. (i) The final ensemble predicts the
# observed 2022 district anomalies; (ii) districts ranked at high livelihood risk are compared with the yield
# losses they recorded (Spearman correlation, Kruskal-Wallis test across risk terciles).

# %%
h22 = data[data.own & (data.year == HOLDOUT_SEASON)].copy()
h22["pred"] = ens_predict(h22[FEATS].values)
h22["pred_xgb"], h22["pred_rf"] = final["XGB"].predict(h22[FEATS].values), final["RF"].predict(h22[FEATS].values)
cl22 = base_sorted.groupby("adm2_pcode")[HEAT_VARS].mean().reindex(h22.adm2_pcode)
h22["heat_loss"] = heat_loss(h22.reset_index(drop=True), cl22.reset_index(drop=True))[0]
w22 = h22.area_ha.fillna(h22.area_ha.median())
g22 = {
    "n_districts": len(h22),
    "observed_mean_anom_pct (area-weighted)": 100 * np.average(h22.anom, weights=w22),
    "predicted_mean_anom_pct (area-weighted)": 100 * np.average(h22.pred, weights=w22),
    "pearson_r": stats.pearsonr(h22.anom, h22.pred)[0], "pearson_p": stats.pearsonr(h22.anom, h22.pred)[1],
    "spearman_rho": stats.spearmanr(h22.anom, h22.pred)[0],
    "RMSE_model": np.sqrt(np.mean((h22.pred - h22.anom) ** 2)), "RMSE_trend": np.sqrt(np.mean(h22.anom ** 2)),
    "share_sign_correct": np.mean(np.sign(h22.pred) == np.sign(h22.anom)),
    "mean_heat_loss_pct": 100 * np.average(h22.heat_loss, weights=w22),
    "heat_loss_1991_2020_mean_pct": 100 * np.average(H0, weights=hzi.area_ha.values),
}
v22 = h22.set_index("adm2_pcode").join(riskdf[["R_now", "H_t", "V_t"]], how="inner")
v22["tercile"] = pd.qcut(v22.R_now, 3, labels=["low", "medium", "high"])
kw = stats.kruskal(*[g.anom.values for _, g in v22.groupby("tercile", observed=False)])
g22.update({"spearman_R_vs_anom": stats.spearmanr(v22.R_now, v22.anom)[0],
            "spearman_R_vs_anom_p": stats.spearmanr(v22.R_now, v22.anom)[1],
            "spearman_H_vs_anom": stats.spearmanr(v22.H_t, v22.anom)[0],
            "spearman_H_vs_anom_p": stats.spearmanr(v22.H_t, v22.anom)[1],
            "kruskal_H": kw.statistic, "kruskal_p": kw.pvalue,
            **{f"mean_anom_pct_{t}_risk": 100 * g.anom.mean() for t, g in v22.groupby("tercile", observed=False)}})
# additional analysis: 2022 losses inside versus outside the significant high-risk cluster (Section 8)
in_hh = v22.index.isin(riskdf.index[riskdf.lisa == "High-High"])
mw = stats.mannwhitneyu(v22.anom[in_hh], v22.anom[~in_hh], alternative="less")
g22.update({"n_highhigh_cluster": int(in_hh.sum()), "mean_anom_pct_highhigh_cluster": 100 * v22.anom[in_hh].mean(),
            "mean_anom_pct_other_districts": 100 * v22.anom[~in_hh].mean(),
            "mannwhitney_U_cluster_lt_other": mw.statistic, "mannwhitney_p_one_sided": mw.pvalue})
val22 = pd.Series(g22)
val22.round(4).to_csv(TAB / "T8_validation_2022.csv", header=["value"])
print(val22.round(3).to_string())
h22.to_csv(TAB / "validation_2022_districts.csv", index=False)

# %% [markdown]
# ### 10.1 Robustness checks
# (a) Hazard ranking with a model that also uses inherited (parent-district) yields; (b) GDHY gridded yields as
# an independent cross-check of the compiled district yield series (1982-2016).

# %%
allrows = data[data.year != HOLDOUT_SEASON].reset_index(drop=True)
a_all, _ = refit_anomaly(allrows, np.ones(len(allrows), bool))
ok = np.isfinite(a_all)
m_inh = make_model("XGB", best["XGB"]).fit(allrows.loc[ok, FEATS].values, a_all[ok])
L_inh = heat_loss(base, clim_heat, models=(m_inh, final["RF"]))[0]
H_inh = base.assign(l=L_inh).groupby("adm2_pcode").l.mean().reindex(risk_d)
rob = {"Spearman hazard: own-only vs incl. inherited": stats.spearmanr(H0, H_inh)[0],
       "Spearman hazard: XGB-only vs RF-only": stats.spearmanr(
           base.assign(l=heat_loss(base, clim_heat, models=(final["XGB"],))[0]).groupby("adm2_pcode").l.mean().reindex(risk_d),
           base.assign(l=heat_loss(base, clim_heat, models=(final["RF"],))[0]).groupby("adm2_pcode").l.mean().reindex(risk_d))[0],
       "Spearman risk: PCA vs equal vulnerability weights": stats.spearmanr(
           riskdf.loc[risk_d, "R_now"], Hs(H0) * Et * minmax(R.V_eq).values)[0]}

gdhy = []
for y in range(1982, 2017):
    f = RAW / "04_crop" / "gdhy_wheat" / "wheat" / f"yield_{y}.nc4"
    with xr.open_dataset(f) as ds:
        da = ds["var"].sel(lat=slice(23, 38), lon=slice(60, 79))
        Wg = weight_matrix(da.lat.values, da.lon.values)
        gdhy.append(pd.DataFrame({"adm2_pcode": PCODES, "year": y,
                                  "gdhy": district_mean(Wg, da.values[None].astype("float64"))[0]}))
gdhy = pd.concat(gdhy).merge(tgt[tgt.own][["adm2_pcode", "year", "yield_t_ha"]], on=["adm2_pcode", "year"]).dropna()
gdhy = gdhy[gdhy.gdhy > 0]
rob["Pearson yield vs GDHY (district-seasons, 1982-2016)"] = stats.pearsonr(gdhy.yield_t_ha, gdhy.gdhy)[0]
gdhy["d_obs"] = gdhy.groupby("adm2_pcode").yield_t_ha.diff()
gdhy["d_gdhy"] = gdhy.groupby("adm2_pcode").gdhy.diff()
gg = gdhy.dropna()
rob["Pearson first differences yield vs GDHY"] = stats.pearsonr(gg.d_obs, gg.d_gdhy)[0]
# (c) heat defined more broadly: stage VPD treated as a heat predictor as well
VPD_VARS = [f"vpd_{s_}" for s_ in STAGE_ORDER]
clim_ext = base.groupby("adm2_pcode")[HEAT_VARS + VPD_VARS].transform("mean")
L_ext = heat_loss(base, clim_ext, heat_vars=HEAT_VARS + VPD_VARS)[0]
H_ext = base.assign(l=L_ext).groupby("adm2_pcode").l.mean().reindex(risk_d)
rob["National heat loss %, heat incl. VPD (vs main)"] = 100 * np.average(H_ext, weights=hzi.area_ha.values)
rob["National heat loss %, main definition"] = 100 * np.average(H0, weights=hzi.area_ha.values)
rob["Spearman hazard: main vs heat incl. VPD"] = stats.spearmanr(H0, H_ext)[0]
rob_tab = pd.Series(rob)
rob_tab.round(4).to_csv(TAB / "S2_robustness.csv", header=["value"])
print(rob_tab.round(3).to_string())

# %%
# final district table (risk, uncertainty, LISA cluster, dominant driver and lever) and surrogate SHAP values
riskdf.round(5).to_csv(TAB / "district_risk.csv")
sv_sur.round(5).to_csv(TAB / "district_risk_drivers_shap.csv")
print("district tables written:", len(riskdf), "districts")
