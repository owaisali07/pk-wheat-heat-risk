"""Paper 8 Fig. 1: study area map and district baseline table.

The methodology flowchart (Fig. 2) is built in draw.io by 02_methodology_drawio.py.

Inputs (already on disk from earlier papers):
  - OCHA COD-AB Pakistan admin2 (160 districts), Paper 3
  - SPAM2010 v2.0 wheat physical area, all and irrigated technologies, Paper 6
  - ERA5-Land daily Tmax 1991-2020, Paper 3
Outputs:
  - figures/Fig1_study_area.(png|pdf)
  - data/district_wheat_heat_baseline.csv
"""
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from rasterio.transform import from_origin
from rasterstats import zonal_stats

ROOT = Path(os.environ.get("EXTERNAL_DATA_ROOT", "external_data"))  # boundaries, SPAM, ERA5-Land
OUT = Path(os.environ.get("PAPER8_ROOT", "."))
FIG = OUT / "figures"
DAT = OUT / "data"
FIG.mkdir(parents=True, exist_ok=True)
DAT.mkdir(parents=True, exist_ok=True)

ADM2 = ROOT / "Paper 3/Paper1_Seasonal_Clock/data/boundaries/pak_admin2.shp"
SPAM = ROOT / "Paper 6/data_raw/spam2010"
TMAX = ROOT / "Paper 3/Paper1_Seasonal_Clock/data/era5land/tmax"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6})

# --- district attributes -------------------------------------------------
g = gpd.read_file(ADM2)
for tech in ("A", "I"):
    zs = zonal_stats(g, str(SPAM / f"spam2010V2r0_global_A_WHEA_{tech}.tif"), stats=["sum"], nodata=-1)
    g[f"wheat_{tech}_ha"] = [z["sum"] or 0.0 for z in zs]
g["wheat_kha"] = g["wheat_A_ha"] / 1e3
g["irr_share"] = np.where(g["wheat_A_ha"] > 500, 100 * g["wheat_I_ha"] / g["wheat_A_ha"].replace(0, np.nan), np.nan)

acc, n = None, 0
for y in range(1991, 2021):
    ds = xr.open_dataset(TMAX / f"era5land_tmax_{y}.nc", engine="h5netcdf")
    da = ds["tmax"].sel(time=ds["time.month"].isin([3, 4])).mean("time").load()
    acc = da if acc is None else acc + da
    n += 1
    ds.close()
tm = acc / n
if float(tm.mean()) > 200:
    tm = tm - 273.15
tm = tm.sortby("lat", ascending=False)
lat, lon = tm["lat"].values, tm["lon"].values
transform = from_origin(lon[0] - 0.05, lat[0] + 0.05, 0.1, 0.1)
zs = zonal_stats(g, tm.values.astype("float64"), affine=transform, stats=["mean"], nodata=np.nan, all_touched=True)
g["tmax_MA_C"] = [z["mean"] for z in zs]

cols = ["adm2_pcode", "adm2_name", "adm1_name", "area_sqkm", "wheat_kha", "irr_share", "tmax_MA_C"]
g[cols].sort_values(["adm1_name", "adm2_name"]).round(3).to_csv(DAT / "district_wheat_heat_baseline.csv", index=False)
prov = g.dissolve("adm1_name")

# --- Fig. 1 --------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.75), constrained_layout=True)
panels = [("wheat_kha", "YlGn", "(a) Wheat physical area (thousand ha)", None),
          ("irr_share", "Greys", "(b) Irrigated share of wheat area (%)", (0, 100)),
          ("tmax_MA_C", "Reds", "(c) Mean Mar–Apr $T_{max}$, 1991–2020 (°C)", (15, 36))]
for ax, (col, cmap, title, lim) in zip(axs, panels):
    vmin, vmax = lim if lim else (np.nanpercentile(g[col], 2), np.nanpercentile(g[col], 98))
    g.plot(column=col, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, edgecolor="#9a9a9a", linewidth=0.15,
           missing_kwds={"color": "#ececec", "edgecolor": "#bdbdbd", "linewidth": 0.15, "hatch": "////"})
    prov.boundary.plot(ax=ax, color="#1a1a1a", linewidth=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, aspect=28)
    cb.ax.tick_params(labelsize=7, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlim(60.6, 78.0)
    ax.set_ylim(23.5, 37.2)
    ax.set_xticks([62, 66, 70, 74, 78])
    ax.set_yticks([24, 28, 32, 36])
    ax.tick_params(labelsize=6.5, width=0.5, length=2)
    ax.set_xticklabels([f"{v}°E" for v in [62, 66, 70, 74, 78]])
    ax.set_yticklabels([f"{v}°N" for v in [24, 28, 32, 36]])
for name, (x, yv) in {"Punjab": (72.2, 30.6), "Sindh": (68.3, 26.0), "Balochistan": (65.2, 28.6),
                      "KP": (71.6, 34.4)}.items():
    axs[0].text(x, yv, name, fontsize=6.5, ha="center", color="#1a1a1a", style="italic")
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"Fig1_study_area.{ext}", dpi=600)

_ED = Path(os.path.join(os.environ.get("PAPER8_ROOT", "."), r"figures/editable"))
_ED.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
for _ext, _kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600}),
                  ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
    fig.savefig(_ED / f"Fig1_study_area.{_ext}", **_kw)
plt.close(fig)

print(g[["wheat_kha", "irr_share", "tmax_MA_C"]].describe().round(2))
print("national wheat area (thousand ha):", round(g["wheat_kha"].sum(), 1))
print("by province (thousand ha):")
print(g.groupby("adm1_name")["wheat_kha"].sum().round(1))
