# Input data

All inputs are public. Download them into `data_raw/` (not versioned) with the folder names below. Large
inputs are not redistributed here; the harmonised and derived products are (see the main README).

| Folder in `data_raw/` | Dataset | Use | Source |
|---|---|---|---|
| `01_boundaries` | OCHA COD-AB Pakistan, 160 districts (admin2), 2022 | Spatial units | Humanitarian Data Exchange (UN OCHA) |
| `02_climate/era5land_*` | ERA5-Land daily Tmax, Tmin, dew point, soil moisture (0–7, 7–28 cm), 0.1° | Hazard predictors | Copernicus Climate Data Store; Google Earth Engine `ECMWF/ERA5_LAND/DAILY_AGGR` (`scripts/04_download_gee.py`) |
| `02_climate/chirps_v2_daily` | CHIRPS v2.0 daily rainfall, 0.05° | Hazard predictors | Climate Hazards Center, UC Santa Barbara |
| `02_climate/nex_gddp_cmip6` | NEX-GDDP-CMIP6 daily tasmax, tasmin, pr, 0.25°, historical and SSP5-8.5 (31 models) | Warming-level deltas | NASA Center for Climate Simulation |
| `02_climate/cmip6_warming_levels` | Years in which each CMIP6 model reaches 1.5, 2 and 3 °C | Warming-level windows | Hauser, cmip_warming_levels (GitHub / Zenodo) |
| `03_earth_observation/modis_mod13q1_ndvi_evi` | MOD13Q1 v061 NDVI/EVI over ESA WorldCover 2021 cropland | Supplementary hazard model | Earth Engine `MODIS/061/MOD13Q1`, `ESA/WorldCover/v200` |
| `03_earth_observation/viirs_nightlights` | VIIRS DNB annual composites | Validation of vulnerability | Earth Engine `NOAA/VIIRS/DNB/ANNUAL_V21`, `_V22` |
| `03_earth_observation/travel_time_cities_2015` | Travel time to cities (Weiss et al., 2018) | Vulnerability indicator | Malaria Atlas Project |
| `04_crop/yields_pbs_crs` | PBS *Crops Area & Production by Districts* (1981-82 to 2008-09); Punjab Crop Reporting Service; MNFSR district volumes | Yield panel | Pakistan Bureau of Statistics; Crop Reporting Service Punjab; Ministry of National Food Security and Research |
| `04_crop/yields_provincial` | *Development Statistics* of Sindh, Khyber Pakhtunkhwa and Balochistan | Yield panel | Provincial Bureaus of Statistics |
| `04_crop/gdhy_wheat` | GDHY v1.2/1.3 gridded wheat yield, 0.5° | Cross-check | Iizumi & Sakai (2020), PANGAEA |
| `04_crop/spam2010` | SPAM2010 v2.0 wheat physical and irrigated area | Weights, irrigated share | IFPRI / Harvard Dataverse |
| `05_socioeconomic/agricultural_census_2024` | 7th Agricultural Census 2024, district tables | Exposure and vulnerability | Pakistan Bureau of Statistics |
| `05_socioeconomic/population_census2023` | 2023 Population and Housing Census, district tables | Exposure and vulnerability | Pakistan Bureau of Statistics |
| `05_socioeconomic/mpi_2016` | District Multidimensional Poverty Index 2014–15 | Vulnerability | Ministry of Planning, Development and Reform, UNDP, OPHI (2016) |
| `05_socioeconomic/relative_wealth_index` | Meta Relative Wealth Index (Chi et al., 2022) | Validation, wealth ranking | Humanitarian Data Exchange |
| crop models | NASA POWER daily radiation and wind; SoilGrids 2.0; NOAA Mauna Loa CO₂ (`apsim_package/inputs/raw/co2_annmean_mlo.csv`); RCMIP concentrations v5.1.0 (Meinshausen et al., 2020) | WOFOST and APSIM inputs | NASA LaRC; ISRIC; NOAA GML; RCMIP (Zenodo) |

The study-area map (`scripts/01_study_area_and_flowchart.py`) reads boundaries, SPAM2010 and ERA5-Land from
`EXTERNAL_DATA_ROOT`. The RCMIP file `rcmip-concentrations-annual-means-v5-1-0.csv` goes into
`apsim_package/inputs/raw/` before you run `apsim_package/prepare_inputs.py`.

## Included derived data

- `data_processed/yield_panel/`: the harmonised district wheat yield panel 1982–2025, the per-source series, overlap
  validation and coverage (see the README there).
- `data_processed/analysis/`: the model table, stage-wise predictors, cross-validation predictions and future hazard.
- `data/district_wheat_heat_baseline.csv`: wheat area and the March–April heat baseline by district.
