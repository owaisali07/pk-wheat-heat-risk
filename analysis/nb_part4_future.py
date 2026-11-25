# %% [markdown]
# ### 5.1 Future hazard at 1.5, 2 and 3 degC global warming (NEX-GDDP-CMIP6, delta method)
#
# For each NEX-GDDP-CMIP6 model, the 20-year window in which it reaches a global warming level (GWL; relative
# to 1850-1900, Hauser et al. warming-level table, SSP5-8.5, SSP2-4.5 where SSP5-8.5 is unavailable) is
# compared with its own 1991-2020 climate. Monthly district changes in Tmax and Tmin (additive) and rainfall
# (multiplicative, bounded to 0.2-5) are applied to the observed 1991-2020 daily series; dew point is shifted
# with mean temperature (approximately constant relative humidity) and soil moisture is held at observed
# values. Predictors are recomputed and the heat loss is evaluated against the observed 1991-2020 heat
# climatology. CO2 fertilisation and adaptation are not represented.

# %%
NEX = CLIM / "nex_gddp_cmip6"
GWLS = [1.5, 2.0, 3.0]
wl = pd.read_csv(CLIM / "cmip6_warming_levels" / "cmip6_warming_levels_one_ens_1850_1900.csv", comment="#",
                 skipinitialspace=True)
wl.columns = [c.strip() for c in wl.columns]
gcms_all = sorted(os.path.basename(p) for p in glob.glob(str(NEX / "*")) if os.path.isdir(p))
# daily Tmax and Tmin are required; CESM2, CESM2-WACCM and IITM-ESM provide only pr and tas in NEX-GDDP
gcms = [m for m in gcms_all if glob.glob(str(NEX / m / "historical" / "tasmax_*"))
        and not m.endswith("_gr2")]  # GFDL-CM4_gr2 is GFDL-CM4 on a second grid: counted once
print(f"NEX-GDDP models: {len(gcms_all)} on disk, {len(gcms)} used (daily Tmax/Tmin, one grid per model):",
      sorted(set(gcms_all) - set(gcms)), "excluded")


def gwl_windows(model):
    name = model.replace("_gr2", "")
    out = {}
    for g in GWLS:
        for exp in ("ssp585", "ssp245"):
            r = wl[(wl.model == name) & (wl.exp == exp) & np.isclose(wl.warming_level, g)]
            if len(r) and os.path.isdir(NEX / model / exp) and len(os.listdir(NEX / model / exp)):
                out[g] = (exp, int(r.start_year.iloc[0]), int(r.end_year.iloc[0]))
                break
    return out


PTS = (pts.lat.values, pts.lon.values, pts.w.values, pts.d.values)
MONTHS = [11, 12, 1, 2, 3, 4]


def nex_monthly(model):
    """Monthly (Nov-Apr) wheat-weighted district means of tasmax, tasmin, pr for the needed years."""
    out = CACHE / "nex" / f"{model}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    out.parent.mkdir(exist_ok=True)
    wins = gwl_windows(model)
    fut_exp = "ssp585" if os.path.isdir(NEX / model / "ssp585") and len(os.listdir(NEX / model / "ssp585")) else "ssp245"
    need = {(y, "historical") for y in range(1991, 2015)} | {(y, fut_exp) for y in range(2015, 2021)}
    for g, (exp, a, b) in wins.items():
        need |= {(y, exp) for y in range(a, b + 1)}
    plat, plon, pw, pd_ = PTS
    W, rec = None, []
    for (y, exp) in sorted(need):
        for var in ("tasmax", "tasmin", "pr"):
            fs = (glob.glob(str(NEX / model / exp / f"{var}_day_*_{exp}_*_{y}_v*.nc"))
                  + glob.glob(str(NEX / model / exp / f"{var}_day_*_{exp}_*_{y}.nc")))
            if not fs:
                continue
            with xr.open_dataset(fs[0], engine="h5netcdf") as ds:
                if W is None:
                    lat, lon = ds.lat.values, ds.lon.values
                    il = np.abs(plat[:, None] - lat[None, :]).argmin(1)
                    jl = np.abs(plon[:, None] - lon[None, :]).argmin(1)
                    W = sparse.csr_matrix((pw, (pd_, il * len(lon) + jl)), shape=(len(PCODES), len(lat) * len(lon)))
                mo = ds.time.dt.month.values
                arr = ds[var].transpose("time", "lat", "lon").values.astype("float64")
            dm = district_mean(W, arr)
            for m in MONTHS:
                sel = mo == m
                if sel.any():
                    rec.append(pd.DataFrame({"model": model, "exp": exp, "var": var, "year": y, "month": m,
                                             "adm2_pcode": PCODES, "value": dm[sel].mean(0)}))
    df = pd.concat(rec, ignore_index=True) if rec else pd.DataFrame(
        columns=["model", "exp", "var", "year", "month", "adm2_pcode", "value"])
    df.to_parquet(out)
    return df


t0 = time.time()
nex_all = Parallel(n_jobs=6)(delayed(nex_monthly)(m) for m in gcms)
print(f"NEX-GDDP monthly district means for {len(gcms)} models in {(time.time() - t0) / 60:.1f} min")

delta_rows, used = [], []
for model, df in zip(gcms, nex_all):
    wins = gwl_windows(model)
    piv = df.pivot_table(index=["year", "month", "adm2_pcode"], columns="var", values="value").reset_index()
    if not {"tasmax", "tasmin", "pr"} <= set(piv.columns):
        continue
    basec = piv[piv.year.between(1991, 2020)].groupby(["month", "adm2_pcode"])[["tasmax", "tasmin", "pr"]].mean()
    for g, (exp, a, b) in wins.items():
        futc = piv[piv.year.between(a, b)].groupby(["month", "adm2_pcode"])[["tasmax", "tasmin", "pr"]].mean()
        d = pd.DataFrame({"dtmax": futc.tasmax - basec.tasmax, "dtmin": futc.tasmin - basec.tasmin,
                          "pr_ratio": (futc.pr / basec.pr.replace(0, np.nan)).clip(0.2, 5).fillna(1.0)}).reset_index()
        d["model"], d["gwl"], d["exp"], d["window"] = model, g, exp, f"{a}-{b}"
        delta_rows.append(d)
        used.append({"model": model, "gwl": g, "exp": exp, "start": a, "end": b})
deltas = pd.concat(delta_rows, ignore_index=True)
used = pd.DataFrame(used)
used.to_csv(TAB / "S1_gcm_warming_windows.csv", index=False)
print(used.groupby("gwl").model.nunique().rename("n_models").to_string())
w_area = hz.set_index("adm2_pcode").area_ha
dsum = deltas[deltas.adm2_pcode.isin(risk_d)].groupby(["gwl", "month"]).apply(
    lambda g: pd.Series({"dTmax_C": np.average(g.dtmax, weights=g.adm2_pcode.map(w_area)),
                         "rain_ratio": np.average(g.pr_ratio, weights=g.adm2_pcode.map(w_area))}))
print("wheat-area-weighted multi-model mean change:\n", dsum.groupby(level=[0, 1]).mean().round(2).unstack(0).to_string())

# %%
obs_win = (tmax.index >= "1990-11-01") & (tmax.index <= "2020-04-30")
TX, TN = tmax[obs_win], tmin[obs_win]
TD = dew.reindex(TX.index)
PR = prcp.reindex(TX.index)
SM = sm.reindex(TX.index)
mon = TX.index.month.values
SEAS_BASE = list(range(1991, 2021))


def future_features(d):
    dx = d.pivot(index="month", columns="adm2_pcode", values="dtmax").reindex(columns=PCODES).fillna(0)
    dn = d.pivot(index="month", columns="adm2_pcode", values="dtmin").reindex(columns=PCODES).fillna(0)
    rr = d.pivot(index="month", columns="adm2_pcode", values="pr_ratio").reindex(columns=PCODES).fillna(1)
    ax = dx.reindex(mon).fillna(0).values
    an = dn.reindex(mon).fillna(0).values
    ar = rr.reindex(mon).fillna(1).values
    f = stage_features(TX + ax, TN + an, TD + (ax + an) / 2, PR * ar, SM, SEAS_BASE)
    f = f.merge(irr_share, left_on="adm2_pcode", right_index=True, how="left")
    f["irr_x_kdd30_late"] = f.irr_share * (f.kdd30_anth + f.kdd30_fill)
    return f[f.adm2_pcode.isin(risk_d)].sort_values(["adm2_pcode", "year"]).reset_index(drop=True)


base_sorted = base.sort_values(["adm2_pcode", "year"]).reset_index(drop=True)
clim_sorted = base_sorted.groupby("adm2_pcode")[HEAT_VARS].transform("mean")
fut_path = CACHE / "future_hazard.parquet"
if fut_path.exists():
    fut = pd.read_parquet(fut_path)
else:
    def one(model, g):
        d = deltas[(deltas.model == model) & (deltas.gwl == g)]
        ff = future_features(d)
        l, fo, _ = heat_loss(ff, clim_sorted)
        r = ff[["adm2_pcode", "year"]].copy()
        r["loss"], r["model"], r["gwl"] = l, model, g
        r["tmax_fill"], r["kdd34_fill"], r["kdd34_anth"] = ff.tmax_fill.values, ff.kdd34_fill.values, ff.kdd34_anth.values
        return r
    t0 = time.time()
    fut = pd.concat([one(r.model, r.gwl) for r in used.itertuples()], ignore_index=True)
    fut.to_parquet(fut_path)
    print(f"future losses for {len(used)} model-GWL pairs in {(time.time() - t0) / 60:.1f} min")

Hf = fut.groupby(["gwl", "model", "adm2_pcode"]).loss.mean().rename("H").reset_index()
Hf_med = Hf.groupby(["gwl", "adm2_pcode"]).H.median().unstack(0)
hz = hz.set_index("adm2_pcode")
for g in GWLS:
    hz[f"H_{g}"] = Hf_med[g].reindex(hz.index)
    hz[f"H_t_{g}"] = hz[f"H_{g}"] * hz.trend2025_t_ha * hz.area_ha
hz = hz.reset_index()
nat = {"present": np.average(hz.H, weights=hz.area_ha)}
for g in GWLS:
    nat[f"+{g}C"] = np.average(hz[f"H_{g}"], weights=hz.area_ha)
gcm_spread = Hf.groupby(["gwl", "model"]).apply(lambda g: np.average(g.H, weights=g.adm2_pcode.map(w_area)))
fut_tab = pd.DataFrame({
    "period": ["1991-2020"] + [f"+{g} degC" for g in GWLS],
    "n_models": [np.nan] + [used[used.gwl == g].model.nunique() for g in GWLS],
    "loss_pct_trend_yield": [100 * nat["present"]] + [100 * nat[f"+{g}C"] for g in GWLS],
    "gcm_p10": [np.nan] + [100 * np.percentile(gcm_spread[g], 10) for g in GWLS],
    "gcm_p90": [np.nan] + [100 * np.percentile(gcm_spread[g], 90) for g in GWLS],
    "loss_kt": [hz.H_t.sum() / 1e3] + [hz[f"H_t_{g}"].sum() / 1e3 for g in GWLS],
})
fut_tab.round(3).to_csv(TAB / "T4_hazard_warming_levels.csv", index=False)
print(fut_tab.round(2).to_string(index=False))
hz.to_csv(TAB / "district_hazard.csv", index=False)
