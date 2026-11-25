# APSIM-Wheat site simulations

This package runs the APSIM Next Generation wheat model at six representative districts. It provides the
crop-model component of the paper: phenology and yield under the 1991–2020 climate, at global warming levels of
1.5, 2 and 3 °C (31 NEX-GDDP-CMIP6 models, with and without CO2 fertilisation), and with two adaptation options.

`prepare_inputs.py` builds the inputs below from the district weather series, NEX-GDDP-CMIP6 deltas, SoilGrids,
NASA POWER and the CO2 records. The simulation outputs used in the paper are in `outputs/`.

## What is in the package

| Path | Content |
|---|---|
| `inputs/sites.csv` | Six sites, wheat-area-weighted centroids, sowing dates, irrigation, approximate phenology targets |
| `inputs/met/*_obs.met` | Daily weather 1984–2025: ERA5-Land Tmax/Tmin and CHIRPS rain (same district series as the hazard model), NASA POWER radiation, NOAA CO2 |
| `inputs/deltas.csv` | Monthly Tmax/Tmin changes and rain ratios per site, GCM and warming level (from the hazard model's 31 GCMs) |
| `inputs/co2_scenarios.csv`, `co2_baseline.json` | SSP5-8.5 CO2 for each GCM's warming-level window; 1991–2020 mean CO2 |
| `inputs/soils.json` | Six-layer soils from SoilGrids 2.0 with Saxton & Rawls (2006) hydraulic properties |
| `inputs/observed_yield.csv` | District yields for evaluating the hindcast |
| `run_apsim.py` | Builds the simulations from the official APSIM examples, runs them, collects results |
| `check_build.py` | Offline check that the simulation files can be built (no APSIM run) |
| `run_on_colab.ipynb` | The same steps as a Google Colab notebook |
| `analyse_apsim.py` | Tables and figures for the paper from `outputs/` |

## Experiments

| Scenario | Weather | CO2 | Management | Simulations |
|---|---|---|---|---|
| `hist` | observed 1984–2025 | observed | calibrated cultivar, standard sowing | 1 per site |
| `base` | observed 1990–2020 | 1991–2020 mean | standard; early sowing (−15 d); slower cultivar; both | 4 per site |
| `gwl{1.5,2,3}\|<GCM>\|co2` | 1990–2020 + GCM change | SSP5-8.5 window mean | standard | 93 per site |
| `gwl{1.5,2,3}\|<GCM>\|noco2` | same | 1991–2020 mean | standard | 93 per site |
| `gwl{1.5,2,3}\|median\|…` | 1990–2020 + ensemble-median change | median window CO2 | standard, early, slower, both | 12 per site |

Wheat is sown on a fixed date with 250 plants m⁻², 225 mm rows and 150 kg N ha⁻¹ in two splits. Irrigated sites
use automatic irrigation on a 7-day canal rotation (refill to field capacity when 50% of the plant-available water
in the top 600 mm is used). Chakwal is rainfed.

## How to run (Linux, or Google Colab)

```bash
# 1. .NET 10 SDK (no administrator rights needed)
curl -sSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh
bash dotnet-install.sh --channel 10.0 --install-dir $HOME/.dotnet
export PATH=$HOME/.dotnet:$PATH DOTNET_ROOT=$HOME/.dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1

# 2. APSIM Next Generation from source (about 3 minutes)
git clone --depth 1 https://github.com/APSIMInitiative/ApsimX.git
dotnet build ApsimX/Models/Models.csproj -c Release

# 3. Python packages
pip install pandas numpy

# 4. Run, in this order
python run_apsim.py --apsimx ApsimX --stage smoke         # must print "SMOKE TEST PASSED"
python run_apsim.py --apsimx ApsimX --stage calibrate     # about 10 minutes
python run_apsim.py --apsimx ApsimX --stage production    # about 1-3 hours on 2-4 cores
```

Results are written to `outputs/`; then run `python analyse_apsim.py` from the repository root.

If the smoke test fails, check `work/apsimx/smoke.log`; the most likely cause is a change in the APSIM example
files, which `run_apsim.py` uses as templates.

## Notes and limitations

- APSIM-Wheat simulates temperature effects on development (shorter phases), radiation-use efficiency and leaf
  growth, and frost. It does not simulate heat damage to grain set around anthesis. `run_apsim.py` therefore also
  reports heat exposure in the simulated anthesis (±7 days) and grain-filling windows, using the 30 and 34 °C
  thresholds of the hazard model.
- No local phenology observations were available. The cultivar is chosen among APSIM's generic spring types and
  Yecora, by matching approximate anthesis and maturity windows. Shahid et al. (2024, Field Crops Res. 307,
  109265) show that APSIM-Wheat needs species-level changes to thermal time and leaf senescence to reproduce
  Pakistani wheat under high temperature. Those changes are not applied here. Results should be read as relative
  changes, not as calibrated yield forecasts.
- Weather changes are applied October–May; the warming-level deltas cover November–April, so October takes the
  November change and May the April change.

## Model set-up corrections

Two corrections were made during the runs:
- The 0–5 cm and 5–15 cm SoilGrids layers are merged into one 0–15 cm top layer. A 5-cm top layer dried within
  days of sowing in hot weather, so seedlings in early-sown or warmer Sindh seasons failed to establish.
- Irrigated sites now receive a 75-mm irrigation at sowing.

Results: `outputs/production_all.csv`. Tables T11, T12, S21 and S22 and Fig. 11 come from `analyse_apsim.py`.
