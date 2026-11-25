"""Paper 8: download the missing climate and EO layers from Google Earth Engine.

  era5      ERA5-Land daily dew point and volumetric soil moisture (layers 1-2, 0-28 cm),
            1980-2025, 0.1 deg, same grid and monthly file layout as era5land_daily.
  modis     MODIS MOD13Q1 v061 NDVI/EVI (250 m, 16-day), mean over cropland pixels
            (ESA WorldCover 2021 class 40) of each district, good-quality pixels only.
  viirs     VIIRS DNB annual night-time lights (V21 2013-2021, V22 2022-2025), district mean/sum.
  access    Travel time to cities 2015 (Weiss et al. 2018): 1 km GeoTIFF over Pakistan and
            district mean/median.

Resumable: existing outputs are skipped. Run with the p6drought env (has earthengine-api):
  python 04_download_gee.py [era5 modis viirs access]
Parallel workers: set YEARS=<first>-<last> to restrict era5/modis to a year range.
"""
import json
import os
import sys
import time

import ee
import numpy as np
import pandas as pd
import xarray as xr

ee.Initialize(project=os.environ["EE_PROJECT"])  # your Google Earth Engine cloud project
DR = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_raw")
W, E, S, N = 60.0, 79.0, 23.0, 38.0
WHAT = sys.argv[1:] or ["access", "viirs", "modis", "era5"]
# optional shard for parallel workers, e.g. YEARS=2010-2017 (existing outputs are skipped anyway)
YEARS = tuple(map(int, os.environ["YEARS"].split("-"))) if os.environ.get("YEARS") else None

with open(os.path.join(DR, "01_boundaries", "pak_admin2_simplified.geojson"), encoding="utf-8") as f:
    DIST = ee.FeatureCollection(json.load(f))


def retry(fn, tries=5):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if k == tries - 1:
                raise
            print("  retry:", str(e)[:150], flush=True)
            time.sleep(30 * (k + 1))


def native_grid(img, band):
    t = img.select(band).projection().getInfo()["transform"]
    sx, x0, sy, y0 = t[0], t[2], t[4], t[5]
    ix0, ix1 = int(np.floor((W - x0) / sx)), int(np.ceil((E - x0) / sx))
    iy0, iy1 = int(np.floor((N - y0) / sy)), int(np.ceil((S - y0) / sy))
    tx, ty = x0 + ix0 * sx, y0 + iy0 * sy
    grid = {"crsCode": "EPSG:4326",
            "affineTransform": {"scaleX": sx, "shearX": 0, "translateX": tx,
                                "shearY": 0, "scaleY": sy, "translateY": ty},
            "dimensions": {"width": ix1 - ix0, "height": iy1 - iy0}}
    return grid, tx + sx * (np.arange(ix1 - ix0) + 0.5), ty + sy * (np.arange(iy1 - iy0) + 0.5)


def zonal(img, reducer, scale, props):
    """District zonal statistics of one image -> list of dicts."""
    fc = img.reduceRegions(collection=DIST, reducer=reducer, scale=scale, tileScale=4)
    feats = retry(lambda: fc.getInfo())["features"]
    return [{**props, **f["properties"]} for f in feats]


# ---------------------------------------------------------------- ERA5-Land
def era5():
    col = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    bands = ["dewpoint_temperature_2m", "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2"]
    grid, lon, lat = native_grid(col.first(), bands[0])
    od = os.path.join(DR, "02_climate", "era5land_dewpoint_soilmoisture")
    y0, y1 = YEARS or (1980, 2025)
    for start in pd.date_range(f"{y0}-01-01", f"{y1}-12-01", freq="MS"):
        p = os.path.join(od, f"era5land_dew_swvl_PAK_{start:%Y%m}.nc")
        if os.path.exists(p):
            continue
        ic = col.filterDate(str(start.date()), str((start + pd.offsets.MonthBegin(1)).date())).sort("system:time_start")
        dates = pd.to_datetime(retry(lambda: ic.aggregate_array("system:time_start").getInfo()), unit="ms")
        out = {}
        for b in bands:
            img = ic.select([b]).toBands().rename([f"d{i}" for i in range(len(dates))]).unmask(-9999, False)
            arr = retry(lambda: ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": grid}))
            a = np.stack([arr[c] for c in arr.dtype.names]).astype("float32")
            a[a == -9999] = np.nan
            out[b] = (("time", "lat", "lon"), a)
        ds = xr.Dataset(out, coords={"time": dates, "lat": lat, "lon": lon}).sortby("lat")
        ds["dewpoint_temperature_2m"].attrs["units"] = "K"
        for b in bands[1:]:
            ds[b].attrs["units"] = "m3 m-3"
        ds.attrs["source"] = "ECMWF/ERA5_LAND/DAILY_AGGR (GEE); swvl1 0-7 cm, swvl2 7-28 cm"
        ds.to_netcdf(p + ".part", encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
        os.replace(p + ".part", p)
        print("ERA5", start.strftime("%Y-%m"), flush=True)


# ---------------------------------------------------------------- MODIS
def modis():
    od = os.path.join(DR, "03_earth_observation", "modis_mod13q1_ndvi_evi")
    crop = ee.ImageCollection("ESA/WorldCover/v200").first().eq(40)
    col = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate("2000-01-01", "2026-01-01")
    red = (ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True))
    y0, y1 = YEARS or (2000, 2025)
    for y in range(y0, y1 + 1):
        p = os.path.join(od, f"modis_ndvi_evi_cropland_district_{y}.csv")
        if os.path.exists(p):
            continue
        ic = col.filterDate(f"{y}-01-01", f"{y+1}-01-01").sort("system:time_start")
        ids = retry(lambda: ic.aggregate_array("system:index").getInfo())
        rows = []
        for iid in ids:
            im = ee.Image(f"MODIS/061/MOD13Q1/{iid}")
            good = im.select("SummaryQA").lte(1)
            img = im.select(["NDVI", "EVI"]).multiply(0.0001).updateMask(good).updateMask(crop)
            rows += zonal(img, red, 250, {"composite": iid.replace("_", "-")})
        df = pd.DataFrame(rows).drop(columns=["NDVI_count"], errors="ignore")
        df = df.rename(columns={"EVI_count": "n_good_cropland_px"})
        df.to_csv(p + ".part", index=False)
        os.replace(p + ".part", p)
        print("MODIS", y, len(df), flush=True)


# ---------------------------------------------------------------- VIIRS
def viirs():
    od = os.path.join(DR, "03_earth_observation", "viirs_nightlights")
    p = os.path.join(od, "viirs_dnb_annual_district_2013_2025.csv")
    if os.path.exists(p):
        return
    rows = []
    for cid in ["NOAA/VIIRS/DNB/ANNUAL_V21", "NOAA/VIIRS/DNB/ANNUAL_V22"]:
        ic = ee.ImageCollection(cid)
        for iid in retry(lambda: ic.aggregate_array("system:index").getInfo()):
            img = ee.Image(f"{cid}/{iid}").select("average_masked").rename("rad")
            red = ee.Reducer.mean().combine(ee.Reducer.sum(), sharedInputs=True)
            rows += zonal(img, red, 463.83, {"year": int(iid[:4]), "version": cid[-3:]})
            print("VIIRS", iid, flush=True)
    pd.DataFrame(rows).rename(columns={"mean": "radiance_mean_nW_cm2_sr",
                                       "sum": "radiance_sum"}).to_csv(p, index=False)


# ---------------------------------------------------------------- travel time
def access():
    od = os.path.join(DR, "03_earth_observation", "travel_time_cities_2015")
    img = ee.Image("projects/malariaatlasproject/assets/accessibility/accessibility_to_cities/2015_v1_0") \
        .select(0).rename("minutes")
    tif = os.path.join(od, "travel_time_to_cities_2015_PAK.tif")
    if not os.path.exists(tif):
        grid, _, _ = native_grid(img, "minutes")
        data = retry(lambda: ee.data.computePixels({"expression": img.unmask(-9999, False).toFloat(),
                                                    "fileFormat": "GEO_TIFF", "grid": grid}))
        with open(tif, "wb") as f:
            f.write(data)
        print("access tif", flush=True)
    p = os.path.join(od, "travel_time_district.csv")
    if not os.path.exists(p):
        red = ee.Reducer.mean().combine(ee.Reducer.median(), sharedInputs=True)
        pd.DataFrame(zonal(img, red, 927.67, {})).to_csv(p, index=False)
        print("access district", flush=True)


for w in WHAT:
    {"era5": era5, "modis": modis, "viirs": viirs, "access": access}[w]()
print("ALL DONE")
