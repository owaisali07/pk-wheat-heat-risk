# Crop-model simulations (Stage H) — WOFOST 7.3 via PCSE

WOFOST 7.3, implemented in pure Python by PCSE, is the first of the two crop models (APSIM-Wheat in
`../apsim_package` is the second). Sites, soils, climate changes and CO2 are shared with `../apsim_package/inputs`.

| File | Purpose |
|---|---|
| `wofost_pipeline.py` | Builds the weather (ERA5-Land, CHIRPS, NASA POWER radiation and wind), calibrates phenology (TSUM1/TSUM2) and runs all scenarios: hindcast, baseline, 31 GCMs x 3 warming levels with and without CO2, and adaptation options. Writes `outputs/wofost_results.csv` |
| `calibrate_tmpftb.py` | Yield-level test of the temperature response of photosynthesis (Supplementary Table S18). Not adopted: the benchmark ratio was not reachable |
| `analyse_wofost.py` | Tables T9, T9b, T10, S16, S17 (in `results/tables/`) and `figures/Fig10_crop_model` |
| `test_wofost.py` | Quick single-site test |

Run order: `python wofost_pipeline.py` (about 20 min on 10 cores), then `python analyse_wofost.py`.

**Key settings**

- Crop parameters: WOFOST parameter set `Winter_wheat_101` (WOFOST 7.3 branch), with daylength and vernalisation switched off (IDSL = 0) to represent spring wheat.
- Production level: irrigated sites at potential production; Chakwal at water-limited production with the classic water balance.
- CO2 is passed as a site parameter. Values above 700 ppm (10 of 31 GCMs at GWL 3 °C) are allowed, because the crop parameter set defines CO2 responses up to 2000 ppm.

**Software:** PCSE 6.0.13, installed with `pip install --no-deps` together with pydispatcher, dotmap and traitlets-pcse. numpy 1.26.4 and pandas 2.2.2 were left unchanged.
