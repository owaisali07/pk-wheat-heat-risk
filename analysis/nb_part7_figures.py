# %% [markdown]
# ## 11. Figures for the manuscript
# Figures 3-9 (Figures 1-2, study area and framework, were produced earlier by `scripts/01_*` and `02_*`).
# All maps use the 2022 OCHA boundaries; districts outside the risk analysis are shown in light grey.

# %%
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

prov = adm.dissolve("adm1_name").reset_index()
gplot = adm.set_index("adm2_pcode").copy()
gplot["geometry"] = gplot.geometry.simplify(0.01, preserve_topology=True)  # ~1 km, keeps vector PDFs small
prov["geometry"] = prov.geometry.simplify(0.01, preserve_topology=True)
W2 = 7.16  # IEEE double-column width (in)


def base_map(ax):
    gplot.plot(ax=ax, color="#eeeeee", edgecolor="#c8c8c8", linewidth=0.15)
    ax.set_xlim(60.6, 77.9)
    ax.set_ylim(23.6, 37.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def map_values(ax, series, cmap, vmin, vmax, title, cblabel, fmt="%.2f", extend="neither"):
    base_map(ax)
    g = gplot.loc[series.index].assign(v=series.values)
    g.plot(column="v", cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, edgecolor="#8a8a8a", linewidth=0.15)
    prov.boundary.plot(ax=ax, color="#1a1a1a", linewidth=0.45)
    sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cb = plt.colorbar(sm_, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=26, extend=extend,
                      format=fmt)
    cb.ax.tick_params(labelsize=6, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label(cblabel, fontsize=6.5)
    ax.set_title(title, loc="left", fontsize=7.5)


def top_legend(ax, frac=0.24, ncol=3, **kw):
    """Single-row legend in headroom added above the data, so it never covers bars, lines or points."""
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + frac * (y1 - y0))
    ax.legend(frameon=False, loc="upper center", ncol=ncol, fontsize=kw.pop("fontsize", 6), handlelength=1.4,
              columnspacing=1.0, borderaxespad=0.2, **kw)


def below_legend(ax, ncol=3, offset=-0.04, **kw):
    """Legend placed outside the axes, under the panel."""
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, offset), ncol=ncol,
              fontsize=kw.pop("fontsize", 6), handlelength=1.2, columnspacing=0.9, **kw)


def save(fig, name):
    # editable text in vector files: TrueType fonts in PDF, real text (not paths) in SVG
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    edit = FIG / "editable"
    edit.mkdir(exist_ok=True)
    fig.savefig(edit / f"{name}.pdf")
    fig.savefig(edit / f"{name}.svg")
    fig.savefig(edit / f"{name}.png", dpi=600)
    fig.savefig(edit / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.show()
    plt.close(fig)


# ---------------- Fig. 3: data and model skill
fig = plt.figure(figsize=(W2, 2.45), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])
ax = fig.add_subplot(gs[0])
nat = panel[panel.own].groupby("year").apply(lambda g: g.prod_t.sum() / g.area_ha.sum())
ann = tgt[tgt.own].groupby("year").apply(lambda g: np.average(g.anom, weights=g.area_ha.fillna(g.area_ha.median())))
ax.bar(ann.index, 100 * ann.values, color=np.where(ann.values < 0, "#c0392b", "#2e86c1"), width=0.8)
ax.set_ylabel("Area-weighted yield anomaly (%)")
ax2 = ax.twinx()
ax2.plot(nat.index, nat.values, color="k", lw=0.9)
ax2.set_ylabel("Yield of reporting districts (t/ha)")
ax.axvline(HOLDOUT_SEASON, color="#555", lw=0.6, ls="--")
ax.set_title("(a) Detrended district yield anomalies", loc="left")
ax = fig.add_subplot(gs[1])
sk = skill[skill.model != "Trend"].pivot(index="design", columns="model", values="SS_vs_trend").loc[
    ["LOYO", "Spatial blocks", "Forward chaining"], ["Ridge", "RF", "XGB"]]
xs = np.arange(len(sk))
for i, (m, c) in enumerate(zip(sk.columns, ["#95a5a6", "#27ae60", "#8e44ad"])):
    ax.bar(xs + (i - 1) * 0.26, sk[m].values, width=0.25, color=c, label=m)
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(xs)
ax.set_xticklabels(["LOYO", "Spatial\nblocks", "Forward\nchaining"])
ax.set_ylabel("Skill score vs trend baseline")
top_legend(ax, frac=0.28)
ax.set_title("(b) Out-of-sample skill", loc="left")
ax = fig.add_subplot(gs[2])
lo = oof[oof.design == "LOYO"]
ax.hexbin(100 * lo.obs, 100 * lo.pred_XGB, gridsize=35, cmap="Greys", mincnt=1, linewidths=0.1)
ax.plot([-70, 70], [-70, 70], color="#c0392b", lw=0.6)
r_ = stats.pearsonr(lo.obs, lo.pred_XGB)[0]
ax.text(0.04, 0.93, f"r = {r_:.2f}, n = {len(lo)}", transform=ax.transAxes, fontsize=6.5)
ax.set_xlim(-70, 70)
ax.set_ylim(-45, 45)
ax.set_xlabel("Observed anomaly (%)")
ax.set_ylabel("Predicted anomaly, XGB (%)")
ax.set_title("(c) LOYO predictions", loc="left")
save(fig, "Fig3_yield_and_skill")

# ---------------- Fig. 4: SHAP and thresholds
fig = plt.figure(figsize=(W2, 4.3), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, 0])
hm = imp[imp.stage != "-"].pivot_table(index="variable", columns="stage", values="mean_abs_shap_XGB").reindex(
    columns=STAGE_ORDER)
hm = hm.loc[hm.sum(1).sort_values(ascending=False).index] * 100
im = ax.imshow(hm.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(5))
ax.set_xticklabels(["Sow", "Till", "Joint", "Anth", "Fill"])
ax.set_yticks(range(len(hm)))
ax.set_yticklabels([{"tmax": "Tmax", "kdd30": "KDD30", "kdd34": "KDD34", "gdd": "GDD", "frost": "Frost days",
                     "vpd": "VPD", "rain": "Rainfall", "dry": "Dry spell", "sm": "Soil moisture"}.get(v, v)
                    for v in hm.index])
cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
cb.set_label("Mean |SHAP| (pp)", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
ax.set_title("(a) Predictor importance (XGB)", loc="left")
for k, s in enumerate(["joint", "anth", "fill"]):
    ax = fig.add_subplot(gs[0, 1 + k] if k < 2 else gs[1, 0])
    j = FEATS.index(f"tmax_{s}")
    ax.scatter(Xd[:, j], 100 * sv_x[:, j], s=1.2, c=dv.irr_share.values, cmap="viridis_r", alpha=0.5, lw=0,
               rasterized=True)
    t_ = thr_x.set_index("stage").loc[s]
    _, lo_ = zero_cross(Xd[:, j], sv_x[:, j])
    ax.plot(lo_[:, 0], 100 * lo_[:, 1], color="#c0392b", lw=1)
    ax.axhline(0, color="k", lw=0.4)
    if np.isfinite(t_.threshold_C) and t_.pct_obs_above >= 1:  # thresholds exceeded by <1 % of seasons are not shown
        ax.axvspan(t_.ci_lo, t_.ci_hi, color="#c0392b", alpha=0.12, lw=0)
        ax.axvline(t_.threshold_C, color="#c0392b", lw=0.6, ls="--")
        ax.text(0.03, 0.06, f"threshold {t_.threshold_C:.1f} °C\n[{t_.ci_lo:.1f}, {t_.ci_hi:.1f}]",
                transform=ax.transAxes, fontsize=6.3)
    ax.set_xlabel(f"Mean Tmax, {dict(joint='jointing (Feb)', anth='anthesis (Mar)', fill='grain filling (Apr)')[s]} (°C)")
    ax.set_ylabel("SHAP (pp of yield)")
    ax.set_title(f"({'bcd'[k]}) SHAP dependence", loc="left")
ax = fig.add_subplot(gs[1, 1])
for s, c in zip(["joint", "anth", "fill"], ["#2e86c1", "#e67e22", "#c0392b"]):
    e, a = ale_res[s]
    ax.plot(e, 100 * a, color=c, lw=1, label={"joint": "Feb", "anth": "Mar", "fill": "Apr"}[s])
ax.axhline(0, color="k", lw=0.4)
ax.set_xlabel("Stage mean Tmax (°C)")
ax.set_ylabel("ALE (pp of yield)")
top_legend(ax, frac=0.22)
ax.set_title("(e) Accumulated local effects", loc="left")
ax = fig.add_subplot(gs[1, 2])
for v, c in [("kdd34_fill", "#c0392b"), ("kdd34_anth", "#e67e22"), ("kdd30_fill", "#7f8c8d")]:
    j = FEATS.index(v)
    lo_ = lowess(sv_x[:, j], Xd[:, j], frac=0.4)
    ax.plot(lo_[:, 0], 100 * lo_[:, 1], color=c, lw=1, label=v.upper().replace("_FILL", " Apr").replace("_ANTH", " Mar"))
ax.axhline(0, color="k", lw=0.4)
ax.set_xlabel("Killing degree days (°C·d)")
ax.set_ylabel("SHAP (pp of yield)")
top_legend(ax, frac=0.36, ncol=2, fontsize=5.6)
ax.set_title("(f) Response to extreme heat", loc="left")
save(fig, "Fig4_shap_thresholds")

# ---------------- Fig. 5: hazard now and at warming levels
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.35), constrained_layout=True)
hzz = hz.set_index("adm2_pcode")
vmax_h = np.nanpercentile(100 * hzz[[f"H_{g}" for g in GWLS]].values, 97)
for ax, col, ttl in zip(axs, ["H"] + [f"H_{g}" for g in GWLS],
                        ["(a) 1991–2020", "(b) +1.5 °C", "(c) +2 °C", "(d) +3 °C"]):
    map_values(ax, 100 * hzz[col], "OrRd", 0, vmax_h, ttl, "Heat loss (% of trend yield)", fmt="%.1f", extend="max")
save(fig, "Fig5_hazard_warming")

# ---------------- Fig. 6: exposure, vulnerability, risk, uncertainty
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.35), constrained_layout=True)
map_values(axs[0], riskdf.E_t, "BuPu", 0, 1, "(a) Exposure", "E (0–1)")
map_values(axs[1], riskdf.V_t, "PuRd", 0, 1, "(b) Vulnerability", "V (0–1)")
map_values(axs[2], riskdf.R_now, "inferno_r", 0, np.nanpercentile(riskdf.R_now, 98), "(c) Livelihood risk",
           "Median R", extend="max")
map_values(axs[3], riskdf.rank_now_hi - riskdf.rank_now_lo, "Greys", 0, np.nanpercentile(riskdf.rank_now_hi - riskdf.rank_now_lo, 98),
           "(d) Rank uncertainty", "90% rank interval width", fmt="%.0f")
save(fig, "Fig6_exposure_vulnerability_risk")

# ---------------- Fig. 7: hotspots and inequality
fig = plt.figure(figsize=(W2, 2.5), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1])
ax = fig.add_subplot(gs[0])
base_map(ax)
cols = {"High-High": "#b2182b", "Low-Low": "#2166ac", "High-Low": "#f4a582", "Low-High": "#92c5de", "ns": "#f7f7f7"}
gl = gplot.loc[risk_d].assign(c=riskdf.loc[risk_d, "lisa"].map(cols).values)
gl.plot(color=gl.c, ax=ax, edgecolor="#8a8a8a", linewidth=0.15)
prov.boundary.plot(ax=ax, color="#1a1a1a", linewidth=0.45)
ax.legend(handles=[Patch(fc=v, ec="#8a8a8a", lw=0.3, label=k if k != "ns" else "not significant")
                   for k, v in cols.items()], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.01),
          ncol=3, fontsize=5.8, handlelength=1.2, columnspacing=0.9)
ax.set_title(f"(a) Risk hotspots (LISA, FDR 5%), I = {gm.I:.2f}", loc="left")
ax = fig.add_subplot(gs[1])
for p, c in zip(["present"] + GWLS, ["k", "#f39c12", "#e74c3c", "#7b241c"]):
    x_, y_ = curves[(p, "relative loss (% of trend yield)")]
    ax.plot(x_, y_, color=c, lw=0.9, label="1991–2020" if p == "present" else f"+{p} °C")
ax.plot([0, 1], [0, 1], color="#999", lw=0.5, ls="--")
ax.set_xlabel("Cumulative share of districts,\npoorest to richest (RWI)")
ax.set_ylabel("Cumulative share of heat loss")
ax.legend(frameon=False, loc="upper left", fontsize=6)
ax.set_title("(b) Concentration curves", loc="left")
ax = fig.add_subplot(gs[2])
iq = ineq[ineq.metric.str.startswith("relative")]
xs = np.arange(len(iq))
ax.bar(xs - 0.18, 100 * iq.share_poorest20, width=0.35, color="#7b241c", label="Poorest 20%")
ax.bar(xs + 0.18, 100 * iq.share_richest20, width=0.35, color="#5d6d7e", label="Richest 20%")
ax.axhline(20, color="#999", lw=0.5, ls="--")
ax.set_xticks(xs)
ax.set_xticklabels(["1991–\n2020", "+1.5 °C", "+2 °C", "+3 °C"])
ax.set_ylabel("Share of total heat loss (%)")
ytop = 100 * max(iq.share_poorest20.max(), iq.share_richest20.max())
for x_, c_ in zip(xs, iq.C):
    top_bar = 100 * max(iq.share_poorest20.iloc[x_], iq.share_richest20.iloc[x_])
    ax.text(x_, top_bar + 0.8, f"C = {c_:.2f}", ha="center", va="bottom", fontsize=6)
ax.set_ylim(0, ytop + 11)
ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=6, handlelength=1.4, borderaxespad=0.2)
ax.set_title("(c) Loss shares by wealth quintile", loc="left")
save(fig, "Fig7_hotspots_inequality")

# ---------------- Fig. 8: policy typology
fig = plt.figure(figsize=(W2, 2.9), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
ax = fig.add_subplot(gs[0])
base_map(ax)
levers = sorted(riskdf.lever.dropna().unique())
pal = dict(zip(levers, ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#f39c12", "#16a085", "#7f8c8d", "#d35400"]))
gt = gplot.loc[risk_d].assign(lever=riskdf.loc[risk_d, "lever"].values, high=top_tercile.reindex(risk_d).values)
for lv, gsub in gt.groupby("lever"):
    if gsub.high.any():
        gsub[gsub.high].plot(ax=ax, color=pal[lv], edgecolor="#555", linewidth=0.2)
    if (~gsub.high).any():
        gsub[~gsub.high].plot(ax=ax, color=pal[lv], alpha=0.35, edgecolor="#999", linewidth=0.15)
prov.boundary.plot(ax=ax, color="#1a1a1a", linewidth=0.45)
ax.legend(handles=[Patch(fc=pal[l], label=l) for l in levers], frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=5.8)
ax.set_title("(a) Dominant risk driver and policy lever (solid: top risk tercile)", loc="left")
ax = fig.add_subplot(gs[1])
sm_ = shap_sur_mean.iloc[::-1]
ax.barh(range(len(sm_)), sm_.values, color="#34495e")
ax.set_yticks(range(len(sm_)))
ax.set_yticklabels(sm_.index)
ax.set_xlabel("Mean |SHAP| on risk index")
ax.set_title(f"(b) Drivers of risk (surrogate CV R² = {sur_r2_cv:.2f})", loc="left")
save(fig, "Fig8_policy_typology")

# ---------------- Fig. 9: 2022 validation
fig = plt.figure(figsize=(W2, 2.45), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.9, 1.05])
ax = fig.add_subplot(gs[0])
cprov = {"Punjab": "#2e86c1", "Sindh": "#e67e22", "Khyber Pakhtunkhwa": "#27ae60", "Balochistan": "#8e44ad", "Islamabad": "#7f8c8d"}
for pv, g in h22.groupby("adm1_name"):
    ax.scatter(100 * g.anom, 100 * g.pred, s=8, color=cprov[pv], label=pv, lw=0)
ax.plot([-60, 40], [-60, 40], color="#999", lw=0.5, ls="--")
ax.axhline(0, color="k", lw=0.3)
ax.axvline(0, color="k", lw=0.3)
ax.set_xlabel("Observed 2022 anomaly (%)")
ax.set_ylabel("Predicted 2022 anomaly (%)")
ax.text(0.03, 0.93, f"r = {g22['pearson_r']:.2f}", transform=ax.transAxes, fontsize=6.5)
# the lower-right quadrant (observed > 0, predicted < -25 %) holds no districts: legend goes there
assert (100 * h22.pred).min() > -25
ax.legend(frameon=False, fontsize=5.5, loc="lower right", borderaxespad=0.3, handletextpad=0.2)
ax.set_title("(a) Hazard model, 2021–22 hold-out", loc="left")
ax = fig.add_subplot(gs[1])
data_t = [100 * g.anom.values for _, g in v22.groupby("tercile", observed=False)]
bp = ax.boxplot(data_t, widths=0.55, patch_artist=True, medianprops=dict(color="k", lw=0.8),
                flierprops=dict(markersize=2))
for patch, c in zip(bp["boxes"], ["#fdebd0", "#f5b041", "#ba4a00"]):
    patch.set_facecolor(c)
    patch.set_linewidth(0.5)
ax.set_xticklabels(["Low", "Medium", "High"])
ax.set_xlabel("Livelihood-risk tercile")
ax.set_ylabel("Observed 2022 anomaly (%)")
ax.axhline(0, color="k", lw=0.3)
y0, y1 = ax.get_ylim()
ax.set_ylim(y0 - 0.25 * (y1 - y0), y1)   # room below the boxes for the cluster note
ax.set_title(f"(b) Risk terciles (Kruskal–Wallis p = {g22['kruskal_p']:.2f})", loc="left")
ax.text(0.03, 0.04, f"LISA high-risk cluster: {g22['mean_anom_pct_highhigh_cluster']:.1f}% vs "
        f"{g22['mean_anom_pct_other_districts']:.1f}%" + chr(10) + f"(Mann–Whitney one-sided p = {g22['mannwhitney_p_one_sided']:.3f})",
        transform=ax.transAxes, fontsize=5.8)
ax = fig.add_subplot(gs[2])
map_values(ax, 100 * h22.set_index("adm2_pcode").anom, "RdBu", -40, 40, "(c) Observed 2022 anomaly", "% of trend yield",
           fmt="%.0f", extend="both")
gplot.loc[riskdf.index[riskdf.lisa == "High-High"]].boundary.plot(ax=ax, color="#00a000", linewidth=0.9)
ax.text(61.0, 36.4, "green outline:" + chr(10) + "high-risk cluster", fontsize=5.8, color="#007000", va="top")
save(fig, "Fig9_validation_2022")
print("figures written to", FIG)

# %% [markdown]
# ## 12. Manuscript tables (LaTeX)
# Result tables are written as booktabs LaTeX fragments to `results/tables/latex/`, so that every number in the
# manuscript comes from this notebook.

# %%
TEX = TAB / "latex"
TEX.mkdir(exist_ok=True)


def tex_table(df, path, fmt):
    lines = []
    for _, r in df.iterrows():
        cells = [fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "--" for c in df.columns]
        lines.append(" & ".join(cells) + r" \\")
    (TEX / path).write_text("\n".join(lines).replace("%", r"\%").replace("_", r"\_"), encoding="utf-8")


t2 = skill.copy()
t2["design"] = pd.Categorical(t2.design, ["LOYO", "Spatial blocks", "Forward chaining"])
t2["model"] = pd.Categorical(t2.model, MODELS)
t2 = t2.sort_values(["design", "model"])
tex_table(t2[["design", "model", "RMSE_anom", "R2_anom", "RMSE_yield_t_ha", "rRMSE_yield_pct", "SS_vs_trend"]], "T2_skill.tex",
          {"RMSE_anom": "{:.3f}", "R2_anom": "{:.3f}", "RMSE_yield_t_ha": "{:.3f}", "rRMSE_yield_pct": "{:.1f}",
           "SS_vs_trend": "{:.3f}"})
tex_table(thr[thr.model == "XGB"][["stage", "threshold_C", "ci_lo", "ci_hi", "slope_pp_per_C_above", "pct_obs_above",
                                   "ALE_threshold_C"]], "T3_thresholds.tex",
          {c: "{:.1f}" for c in ["threshold_C", "ci_lo", "ci_hi", "slope_pp_per_C_above", "pct_obs_above", "ALE_threshold_C"]})
tex_table(fut_tab, "T4_hazard.tex", {"n_models": "{:.0f}", "loss_pct_trend_yield": "{:.2f}", "gcm_p10": "{:.2f}",
                                     "gcm_p90": "{:.2f}", "loss_kt": "{:.0f}"})
tex_table(pca_tab[["indicator", "weight_PCA", "loading_PC1", "loading_PC2"]], "T5_weights.tex",
          {"weight_PCA": "{:.3f}", "loading_PC1": "{:.2f}", "loading_PC2": "{:.2f}"})
tex_table(ineq[["period", "metric", "C", "C_boot_lo", "C_boot_hi", "share_poorest20", "share_richest20"]], "T6_inequality.tex",
          {"C": "{:.3f}", "C_boot_lo": "{:.3f}", "C_boot_hi": "{:.3f}", "share_poorest20": "{:.3f}", "share_richest20": "{:.3f}"})
print(sorted(os.listdir(TEX)))
