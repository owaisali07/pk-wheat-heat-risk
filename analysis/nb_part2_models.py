# %% [markdown]
# ## 3. Hazard model and blocked cross-validation (Stage A)
#
# Four predictors of the relative yield anomaly are compared: a trend-only baseline (anomaly = 0), ridge
# regression, random forest and XGBoost. Hyperparameters are tuned in an inner loop (3-fold, grouped by
# season). The district technology trends are refitted on the training seasons of every fold, so test
# seasons never inform the detrending. Three outer designs avoid leakage:
# leave-one-year-out (LOYO), leave-district-group-out (8 spatially contiguous k-means blocks), and forward
# chaining (train on all seasons before a 5-year block, predict the block).

# %%
import itertools

import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if "data" not in globals():
    data = pd.read_parquet(CACHE / "model_table.parquet")
FEATS = CLIM_FEATURES
full = data[data.own].reset_index(drop=True)  # all own observations incl. 2022 (2022 used only in Stage G)
dev = full[full.year != HOLDOUT_SEASON].reset_index(drop=True)


def refit_anomaly(df, fit_mask):
    """Centred Theil-Sen trend per district on rows where fit_mask; returns anomaly and trend for all rows."""
    tr = np.full(len(df), np.nan)
    for pc, ix in df.groupby("adm2_pcode").indices.items():
        f = ix[fit_mask[ix]]
        if len(f) < 8:
            continue
        r = stats.theilslopes(df.yield_t_ha.values[f], df.year.values[f].astype(float))
        t_ = r.intercept + r.slope * df.year.values[ix]
        tf = r.intercept + r.slope * df.year.values[f]
        tr[ix] = t_ * (1 + np.mean((df.yield_t_ha.values[f] - tf) / tf))  # centred: zero mean anomaly on fit rows
    tr = np.where(tr > 0.1, tr, np.nan)
    return (df.yield_t_ha.values - tr) / tr, tr


GRIDS = {
    "Ridge": [{"alpha": a} for a in (1.0, 10.0, 100.0, 1000.0)],
    "RF": [{"max_features": mf, "min_samples_leaf": ml} for mf in (0.33, 0.6) for ml in (15, 40)],
    "XGB": [{"max_depth": md, "min_child_weight": mc} for md in (2, 3) for mc in (20, 60)],
}


def make_model(name, p, n_jobs=N_JOBS, seed=SEED):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))
    if name == "RF":
        return RandomForestRegressor(n_estimators=300, max_features=p["max_features"],
                                     min_samples_leaf=p["min_samples_leaf"], n_jobs=n_jobs, random_state=seed)
    return xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=p["max_depth"],
                            min_child_weight=p["min_child_weight"], subsample=0.8, colsample_bytree=0.7,
                            reg_lambda=1.0, tree_method="hist", n_jobs=n_jobs, random_state=seed)


def tune(name, X, y, groups):
    gkf = GroupKFold(n_splits=3)
    best, best_mse = None, np.inf
    for p in GRIDS[name]:
        mse = []
        for a, b in gkf.split(X, y, groups):
            m = make_model(name, p).fit(X[a], y[a])
            mse.append(np.mean((m.predict(X[b]) - y[b]) ** 2))
        if np.mean(mse) < best_mse:
            best, best_mse = p, np.mean(mse)
    return best


def run_design(df, folds, label):
    """folds: list of (train_mask, test_mask, trend_mask). Returns out-of-fold predictions and params.
    trend_mask = seasons used to fit the district trends (training seasons for LOYO and forward chaining;
    all seasons of the held-out districts for the spatial design, whose test is spatial transfer)."""
    recs, chosen = [], []
    for k, (trm, tem, tdm) in enumerate(folds):
        anom, trend = refit_anomaly(df, tdm)
        ok = np.isfinite(anom)
        tr_i, te_i = np.where(trm & ok)[0], np.where(tem & ok)[0]
        if len(te_i) == 0:
            continue
        X, y = df[FEATS].values, anom
        pred = {"Trend": np.zeros(len(te_i))}
        for name in ("Ridge", "RF", "XGB"):
            p = tune(name, X[tr_i], y[tr_i], df.year.values[tr_i])
            chosen.append({"design": label, "fold": k, "model": name, **p})
            pred[name] = make_model(name, p).fit(X[tr_i], y[tr_i]).predict(X[te_i])
        recs.append(pd.DataFrame({"design": label, "fold": k, "adm2_pcode": df.adm2_pcode.values[te_i],
                                  "year": df.year.values[te_i], "obs": y[te_i], "trend": trend[te_i],
                                  "yield": df.yield_t_ha.values[te_i], **{f"pred_{m}": v for m, v in pred.items()}}))
    return pd.concat(recs, ignore_index=True), pd.DataFrame(chosen)


# district blocks for the spatial design
cent = study.set_index("adm2_pcode").to_crs(32642).geometry.centroid
dpc = dev.adm2_pcode.unique()
km = KMeans(n_clusters=8, n_init=20, random_state=SEED).fit(np.c_[cent[dpc].x, cent[dpc].y])
block = pd.Series(km.labels_, index=dpc)
dev["block"] = dev.adm2_pcode.map(block)

yrs = dev.year.values
FOLDS = {
    "LOYO": [(yrs != y, yrs == y, yrs != y) for y in sorted(dev.year.unique())],
    "Spatial blocks": [(dev.block.values != b, dev.block.values == b, np.ones(len(dev), bool)) for b in range(8)],
    "Forward chaining": [(yrs < a, (yrs >= a) & (yrs <= a + 4), yrs < a) for a in (2001, 2006, 2011, 2016, 2021)],
}
cv_path = CACHE / "cv_predictions.parquet"
if cv_path.exists():
    oof = pd.read_parquet(cv_path)
    params = pd.read_csv(CACHE / "cv_params.csv")
else:
    t0 = time.time()
    res = [run_design(dev, f, lab) for lab, f in FOLDS.items()]
    oof = pd.concat([r[0] for r in res], ignore_index=True)
    params = pd.concat([r[1] for r in res], ignore_index=True)
    oof.to_parquet(cv_path)
    params.to_csv(CACHE / "cv_params.csv", index=False)
    print(f"cross-validation done in {(time.time() - t0) / 60:.1f} min")

# %% [markdown]
# ### 3.1 Skill and Diebold-Mariano tests
# RMSE and R^2 of the anomaly, RMSE and relative RMSE of yield (anomaly converted back with the fold trend),
# and the skill score over the trend baseline, SS = 1 - MSE_model / MSE_trend. Paired model comparisons use
# the Diebold-Mariano test on the season series of district-mean squared-error differences (Newey-West
# variance, lag 1, Harvey-Leybourne-Newbold small-sample correction).

# %%
MODELS = ["Trend", "Ridge", "RF", "XGB"]


def dm_test(e1, e2, h=1):
    d = e1 - e2
    T = len(d)
    dc = d - d.mean()
    gamma = [np.sum(dc[k:] * dc[:T - k]) / T for k in range(h + 1)]
    var = gamma[0] + 2 * sum((1 - k / (h + 1)) * gamma[k] for k in range(1, h + 1))
    dm = d.mean() / np.sqrt(var / T)
    dm *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return dm, 2 * stats.t.sf(abs(dm), df=T - 1)


skill_rows, dm_rows = [], []
for des, g in oof.groupby("design", sort=False):
    mse_trend = np.mean(g.obs ** 2)
    for m in MODELS:
        e = g[f"pred_{m}"] - g.obs
        yhat = g.trend * (1 + g[f"pred_{m}"])
        rmse_y = np.sqrt(np.mean((yhat - g["yield"]) ** 2))
        skill_rows.append({"design": des, "model": m, "n": len(g), "RMSE_anom": np.sqrt(np.mean(e ** 2)),
                           "R2_anom": r2_score(g.obs, g[f"pred_{m}"]) if m != "Trend" else 0.0,
                           "RMSE_yield_t_ha": rmse_y, "rRMSE_yield_pct": 100 * rmse_y / g["yield"].mean(),
                           "R2_yield": r2_score(g["yield"], yhat), "SS_vs_trend": 1 - np.mean(e ** 2) / mse_trend})
    se = g.assign(**{f"se_{m}": (g[f"pred_{m}"] - g.obs) ** 2 for m in MODELS}).groupby("year")[
        [f"se_{m}" for m in MODELS]].mean()
    for a, b in [("XGB", "Trend"), ("RF", "Trend"), ("Ridge", "Trend"), ("XGB", "RF"), ("XGB", "Ridge"), ("RF", "Ridge")]:
        dm, p = dm_test(se[f"se_{a}"].values, se[f"se_{b}"].values)
        dm_rows.append({"design": des, "comparison": f"{a} vs {b}", "DM": dm, "p_value": p, "n_seasons": len(se)})
skill = pd.DataFrame(skill_rows)

# skill at aggregate scale (province-season, area-weighted) and by irrigation class (LOYO)
lo = oof[oof.design == "LOYO"].merge(dev[["adm2_pcode", "year", "area_ha", "adm1_name", "irr_share"]],
                                      on=["adm2_pcode", "year"])
lo["w"] = lo.area_ha.fillna(lo.area_ha.median())
agg_rows = []
for lev, keys in [("district-season", None), ("province-season", ["adm1_name", "year"]), ("national-season", ["year"])]:
    if keys is None:
        g_ = lo
    else:
        g_ = lo.groupby(keys).apply(lambda g: pd.Series({c: np.average(g[c], weights=g.w) for c in
                                                          ["obs"] + [f"pred_{m}" for m in MODELS]})).reset_index()
    for m in ("Ridge", "RF", "XGB"):
        agg_rows.append({"level": lev, "model": m, "n": len(g_), "r": stats.pearsonr(g_.obs, g_[f"pred_{m}"])[0],
                         "SS_vs_trend": 1 - np.mean((g_[f"pred_{m}"] - g_.obs) ** 2) / np.mean(g_.obs ** 2)})
lo["irr_class"] = np.where(lo.irr_share >= 0.8, "irrigated (>=80%)", np.where(lo.irr_share < 0.5, "rainfed (<50%)", "mixed"))
for cl, g_ in lo.groupby("irr_class"):
    for m in ("Ridge", "RF", "XGB"):
        agg_rows.append({"level": f"district-season, {cl}", "model": m, "n": len(g_),
                         "r": stats.pearsonr(g_.obs, g_[f"pred_{m}"])[0],
                         "SS_vs_trend": 1 - np.mean((g_[f"pred_{m}"] - g_.obs) ** 2) / np.mean(g_.obs ** 2)})
agg_skill = pd.DataFrame(agg_rows)
agg_skill.round(4).to_csv(TAB / "T2d_skill_by_scale_and_irrigation.csv", index=False)
print(agg_skill.round(3).to_string(index=False))
dmt = pd.DataFrame(dm_rows)
skill.round(4).to_csv(TAB / "T2_model_skill.csv", index=False)
dmt.round(4).to_csv(TAB / "T2b_diebold_mariano.csv", index=False)
print(skill.round(3).to_string(index=False))
print(dmt.round(3).to_string(index=False))

# %% [markdown]
# ### 3.2 Final models and bootstrap ensemble
# The final XGBoost and random forest are tuned on all development seasons (5-fold grouped by season) and
# refitted 40 times on season-block bootstrap samples; the refits carry model uncertainty into the hazard and
# the Monte Carlo risk analysis. The supplementary EO model adds MODIS NDVI/EVI (2001-2025).

# %%
anom_dev, trend_dev = refit_anomaly(dev, np.ones(len(dev), bool))
dev["anom_fit"], dev["trend_fit"] = anom_dev, trend_dev
dv = dev[np.isfinite(dev.anom_fit)].reset_index(drop=True)
Xd, yd, gd = dv[FEATS].values, dv.anom_fit.values, dv.year.values


def tune5(name, X, y, g):
    gkf = GroupKFold(n_splits=5)
    scores = {}
    for i, p in enumerate(GRIDS[name]):
        scores[i] = np.mean([np.mean((make_model(name, p).fit(X[a], y[a]).predict(X[b]) - y[b]) ** 2)
                             for a, b in gkf.split(X, y, g)])
    return GRIDS[name][min(scores, key=scores.get)]


best = {m: tune5(m, Xd, yd, gd) for m in ("XGB", "RF", "Ridge")}
print("final hyperparameters:", best)
final = {m: make_model(m, best[m]).fit(Xd, yd) for m in ("XGB", "RF", "Ridge")}

B = 40
boot_path = CACHE / "boot_models.json"
seasons_dev = np.unique(gd)


def fit_boot(b):
    r = np.random.default_rng(SEED + b)
    s = r.choice(seasons_dev, size=len(seasons_dev), replace=True)
    ix = np.concatenate([np.where(gd == k)[0] for k in s])
    mx = make_model("XGB", best["XGB"], n_jobs=1, seed=SEED + b).fit(Xd[ix], yd[ix])
    mr = make_model("RF", best["RF"], n_jobs=1, seed=SEED + b).fit(Xd[ix], yd[ix])
    return mx, mr


t0 = time.time()
boots = Parallel(n_jobs=N_JOBS)(delayed(fit_boot)(b) for b in range(B))
print(f"{B} bootstrap refits of XGB and RF in {time.time() - t0:.0f} s")


def ens_predict(X, models=None):
    models = models or (final["XGB"], final["RF"])
    return np.mean([m.predict(X) for m in models], axis=0)


# supplementary model: climate + MODIS (2001-2025 seasons), LOYO
eo_dev = dv[dv.year >= 2001].dropna(subset=EO_FEATURES).reset_index(drop=True)
eo_rows = []
for fs_name, fs in [("climate only", FEATS), ("climate + MODIS", FEATS + EO_FEATURES)]:
    pr = np.zeros(len(eo_dev))
    for y in eo_dev.year.unique():
        a, b = eo_dev.year.values != y, eo_dev.year.values == y
        pr[b] = make_model("XGB", best["XGB"]).fit(eo_dev.loc[a, fs].values, eo_dev.anom_fit.values[a]).predict(
            eo_dev.loc[b, fs].values)
    e = pr - eo_dev.anom_fit.values
    eo_rows.append({"feature_set": fs_name, "n": len(eo_dev), "RMSE_anom": np.sqrt(np.mean(e ** 2)),
                    "R2_anom": r2_score(eo_dev.anom_fit, pr),
                    "SS_vs_trend": 1 - np.mean(e ** 2) / np.mean(eo_dev.anom_fit ** 2)})
eo_skill = pd.DataFrame(eo_rows)
eo_skill.round(4).to_csv(TAB / "T2c_eo_added_skill.csv", index=False)
print(eo_skill.round(3).to_string(index=False))

# %% [markdown]
# ### 3.3 Feature-design experiment (Supplementary Table S3)
# LOYO skill with (a) absolute predictors, (b) within-district anomalies of all predictors and (c) anomalies plus
# absolute late-season heat, each with standard and strong regularisation. Absolute predictors were retained:
# they performed at least as well and are required to read physical temperature thresholds.

# %%
fe_path = CACHE / "feature_experiment.csv"
if not fe_path.exists():
    cols_c = [c for c in FEATS if not c.startswith("irr")]
    A_ = dev[cols_c] - dev.groupby("adm2_pcode")[cols_c].transform("mean")
    A_.columns = [c + "_a" for c in cols_c]
    dx = pd.concat([dev, A_], axis=1)
    late_abs = [f"{v}_{s}" for s in ("joint", "anth", "fill") for v in ("tmax", "kdd30", "kdd34")]
    SETS = {"absolute": FEATS, "anomaly": list(A_.columns) + ["irr_share"],
            "anomaly+late abs heat": list(A_.columns) + late_abs + ["irr_share"]}
    MK = {"XGB std": lambda: make_model("XGB", {"max_depth": 3, "min_child_weight": 20}),
          "XGB strong": lambda: xgb.XGBRegressor(n_estimators=400, learning_rate=0.02, max_depth=2, min_child_weight=60,
                                                 subsample=0.7, colsample_bytree=0.5, reg_lambda=10, n_jobs=N_JOBS,
                                                 random_state=1),
          "RF strong": lambda: make_model("RF", {"max_features": 0.33, "min_samples_leaf": 40})}
    rows_fe = []
    for sname, cols in SETS.items():
        for mname, mk in MK.items():
            o_, p_ = [], []
            for y in np.unique(yrs):
                an, _ = refit_anomaly(dx, yrs != y)
                ok = np.isfinite(an)
                m_ = mk().fit(dx.loc[(yrs != y) & ok, cols].values, an[(yrs != y) & ok])
                o_.append(an[(yrs == y) & ok]); p_.append(m_.predict(dx.loc[(yrs == y) & ok, cols].values))
            o_, p_ = np.concatenate(o_), np.concatenate(p_)
            rows_fe.append({"features": sname, "model": mname, "SS_vs_trend": 1 - np.mean((p_ - o_) ** 2) / np.mean(o_ ** 2),
                            "r": stats.pearsonr(o_, p_)[0]})
    pd.DataFrame(rows_fe).to_csv(fe_path, index=False)
fe = pd.read_csv(fe_path)
fe.round(4).to_csv(TAB / "S3_feature_design_experiment.csv", index=False)
print(fe.round(3).to_string(index=False))
