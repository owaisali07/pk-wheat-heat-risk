# Analysis code (Stages A–G)

`Paper8_Analysis.ipynb` contains the complete analysis, from the raw inputs in `../data_raw/` to every
table in `../results/tables/` and Figures 3–9 in `../figures/`. It is built from the seven part
files below, which use `# %%` cell markers, by `python build_notebook.py`.

| Part | Notebook sections | Methodology stage |
|---|---|---|
| `nb_part1_data.py` | 1–2 | Yield target; wheat-weighted daily climate (ERA5-Land, CHIRPS) and MODIS; stage-wise predictors |
| `nb_part2_models.py` | 3 | Trend, ridge, random forest and XGBoost; three blocked CV designs; Diebold–Mariano tests; bootstrap ensemble; MODIS added skill |
| `nb_part3_shap_hazard.py` | 4–5 | SHAP, heat thresholds, ALE; heat-attributable losses and hazard (A) |
| `nb_part4_future.py` | 5.1 | NEX-GDDP-CMIP6 deltas at 1.5/2/3 °C warming levels; future hazard (A) |
| `nb_part5_vulnerability.py` | 6 | Exposure (B); PCA vulnerability index and its validity (C) |
| `nb_part6_risk.py` | 7–10 | Monte Carlo risk (D); LISA and concentration index (E); surrogate SHAP typology (F); 2022 validation (G); robustness |
| `nb_part7_figures.py` | 11 | Figures 3–9 |

## Running

**Environment:** Windows Anaconda base environment (Python 3.12) with numpy 1.26, pandas 2.2, scipy 1.13,
scikit-learn 1.5, xgboost 2.1, shap, statsmodels, geopandas 1.0, xarray, h5netcdf, rasterio,
libpysal 4.12 and esda 2.6.

**Run the notebook:**

```
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Paper8_Analysis.ipynb
```

**Caching:** expensive intermediate products are cached in `../data_processed/analysis/`: daily
district climate, predictor tables, CV predictions, NEX-GDDP monthly district means and future
losses. Delete that folder to recompute everything from the raw data. A full recomputation takes
about 1.5 h on a 12-core laptop, most of it nested cross-validation.

**Reproducibility:** randomness is seeded (`SEED = 20260514`). The 2021–22 season is withheld from
all fitting and is used only in Stage G.

## Inputs

All inputs are open data (see `../data_raw/README.md`). The district wheat yield panel is built
by `../scripts/yield_panel/` (see `../data_processed/yield_panel/README.md`).
