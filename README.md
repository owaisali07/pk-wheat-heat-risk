# Wheat heat-stress livelihood risk in Pakistan (explainable GeoAI)

Code, harmonised data and derived results for the paper

> **Explainable GeoAI Locates Structural Wheat Heat-Stress Risk in Pakistan: Integrating Earth Observation, Climate Projections and Socioeconomic Data**
> Owais Ali, Shahida Haji and Adnan Arshad (submitted).

## What the study does

- Builds a harmonised district wheat yield panel for Pakistan (1982–2025, 2022 district boundaries) from official statistics.
- Models relative yield anomalies with XGBoost, random forest and ridge regression, using 47 stage-wise ERA5-Land and CHIRPS predictors.
- Evaluates skill under strictly blocked cross-validation (by year, by region, and forward in time).
- Derives heat thresholds with SHAP and accumulated local effects.
- Estimates heat-attributable losses now and at 1.5, 2 and 3 °C of global warming (31 NEX-GDDP-CMIP6 models).
- Simulates six districts with two crop models (WOFOST 7.3 and APSIM-Wheat) to add CO₂, phenology and adaptation effects.
- Combines hazard, exposure and a census-based vulnerability index into a district livelihood-risk index with Monte Carlo uncertainty.
- Analyses hotspots (local Moran's I), the inequality of losses (concentration index) and the drivers of risk (a policy typology).
- Tests the results against the withheld 2021–22 heatwave season.

## Repository structure

```
analysis/                 full statistical analysis: Paper8_Analysis.ipynb, built from nb_part1..7 (Stages A–G)
scripts/                  study-area map and Fig. 2 (draw.io), Google Earth Engine downloads
scripts/yield_panel/      extraction and harmonisation of the district wheat yield panel (y01–y08)
crop_model/               WOFOST 7.3 (PCSE) pipeline and analysis (Stage H), inputs and outputs
apsim_package/            APSIM-Wheat (APSIM Next Generation) inputs, runner and analysis (Stage H)
data_processed/yield_panel/   harmonised district wheat yield panel 1982–2025 and the per-source series
data_processed/analysis/  model table, predictors, cross-validation predictions, future hazard
data/                     district wheat heat baseline
results/tables/           every table of the paper and supplement (CSV, LaTeX)
figures/                  Figures 1–11 (PDF, PNG; Fig. 2 also as draw.io source)
```

## Main data products

| File | Content |
|---|---|
| `data_processed/yield_panel/wheat_yield_district_panel_1982_2025.csv` | District wheat area, production and yield, seasons 1981–82 to 2024–25, 2022 boundaries, with source and flags |
| `data_processed/yield_panel/README.md` | Sources, harmonisation rules and coverage of the panel |
| `results/tables/district_hazard.csv` | Heat-attributable loss by district, present and at 1.5, 2 and 3 °C |
| `results/tables/district_risk.csv` | Exposure, vulnerability, hazard and livelihood risk (median and 90% rank interval) by district |
| `results/tables/district_socioeconomic_indicators.csv` | Vulnerability and exposure indicators by district |
| `results/tables/district_risk_drivers_shap.csv` | Dominant driver and policy lever by district |
| `crop_model/outputs/wofost_results.csv`, `apsim_package/outputs/production_all.csv` | Crop-model simulations at the six districts |

## Reproducing the results

1. **Environment.** Python 3.12 (see `requirements.txt`). The crop models also need PCSE 6.0 for WOFOST, and APSIM Next Generation for APSIM-Wheat (see `apsim_package/README.md`).
2. **Inputs.** Download the input datasets listed in [`data/README.md`](data/README.md) into `data_raw/`. The ERA5-Land, CHIRPS and MODIS district series can be produced with `scripts/04_download_gee.py`, which needs a Google Earth Engine project: set `EE_PROJECT`.
3. **Paths.** Scripts resolve paths relative to the repository root. Run them from the root, or set `PAPER8_ROOT` to the repository folder. The notebook runs from `analysis/`.
4. **Yield panel.** Run `scripts/yield_panel/y01`–`y08` in order. The finished panel is already included.
5. **Statistical analysis.** Run `python analysis/build_notebook.py`, then execute `analysis/Paper8_Analysis.ipynb`. Expensive intermediate products are cached in `data_processed/analysis/`.
6. **Crop models.**
   - WOFOST: `python crop_model/wofost_pipeline.py`, then `python crop_model/analyse_wofost.py`.
   - APSIM: `python apsim_package/prepare_inputs.py`, then `python apsim_package/run_apsim.py --apsimx <ApsimX folder> --stage production`, then `python apsim_package/analyse_apsim.py`.

## Data availability

All input datasets are public; `data/README.md` lists their sources. This repository contains:
- the harmonised yield panel;
- the derived predictors and results;
- every table and figure of the paper.

Large intermediate files are not included, because the code regenerates them from the public inputs:
- the daily district climate series;
- the NEX-GDDP-CMIP6 district series;
- the RCMIP concentration file.

## Citation

Please cite the paper and this repository (see `CITATION.cff`):

> Ali, O., Haji, S., & Arshad, A. (2026). Explainable GeoAI locates structural wheat heat-stress risk in Pakistan: Integrating Earth observation, climate projections and socioeconomic data (submitted).
>
> Ali, O., Haji, S., & Arshad, A. (2026). Code and data for: Explainable GeoAI locates structural wheat heat-stress risk in Pakistan (v1.0). GitHub. https://github.com/owaisali07/pk-wheat-heat-risk

## Licence

- **Code** (`*.py`, `*.ipynb`): MIT License (`LICENSE`).
- **Data, results and figures** (`data/`, `data_processed/`, `results/`, `figures/`, and the crop-model inputs and outputs): CC BY 4.0 (`LICENSE-CC-BY-4.0.md`).
- **Input datasets** keep their own licences.

## Contact

Owais Ali, Department of Space Science, Institute of Space Technology, Islamabad, Pakistan (owais.ali.bahria@gmail.com).
