# District wheat yield panel, 1981-82 to 2024-25 (2022 boundaries)

**Main file:** `wheat_yield_district_panel_1982_2025.csv`. It has one row per district and
season: 136 districts × 44 seasons = 5,984 rows.

- **Coverage:** the 2022 OCHA admin2 districts of Punjab (36), Sindh (29), Khyber Pakhtunkhwa (35),
  Balochistan (35) and Islamabad (1).
- **Excluded:** Azad Kashmir and Gilgit-Baltistan, because no district wheat statistics are published for them.
- **Rebuild:** run `scripts/yield_panel/y01 … y08` in order. Each script reads the PDFs in
  `data_raw/04_crop` and writes one `src_*.csv` file; `y07` builds the panel.

## Columns
| Column | Meaning |
|---|---|
| `adm2_pcode`, `adm2_name`, `province` | 2022 OCHA district |
| `season`, `year` | Rabi season (e.g. `2021-22`); `year` = harvest year (2022) |
| `area_ha`, `prod_t` | Harvested area (ha) and production (t). Left empty for `inherited` rows. |
| `yield_t_ha` | Production divided by area (t/ha) |
| `source` | `pbs`, `punjab_crs`, `sindh_ds`, `kp_ds`, `bal_ds`, `mnfsr`, `mnfsr_scan` (see below) |
| `flag` | See the flag table below |
| `from_unit` | The unit whose figures were used |
| `qc`, `robust_z` | `implausible_level` (yield < 0.3 or > 6.5 t/ha) or `outlier_vs_local_median` (\|z\| > 5 against a 9-season rolling median). Flagged rows are kept, not dropped. |

**Flags:**

| Flag | Meaning |
|---|---|
| `own` | The district's own figure |
| `own_merged_units` | The sum of units that now lie inside the district (e.g. Rawalpindi + Murree after 2022) |
| `inherited` | The district did not exist yet, or was not reported separately. It takes the yield of its parent district (`from_unit`). |
| `missing` | No data |

## Sources and priority
In each province, the first source in the list that has data is used.

| Province | Priority (seasons) |
|---|---|
| Punjab, Islamabad | Punjab Crop Reporting Service historical series (1990-91 to 2024-25) → PBS (1981-82 to 1989-90) |
| Sindh | MNFSR (2021-22 to 2024-25) → MNFSR 2021-22 volume, scanned pages, transcribed (2020-21) → Sindh BOS *Development Statistics* (2005-06 to 2019-20) → PBS |
| KP | MNFSR (2021-22 to 2024-25) → MNFSR scan (2020-21) → KP *Development Statistics* 2014–2021 editions (2010-11 to 2019-20) → PBS |
| Balochistan | *Development Statistics of Balochistan* (2017-18 to 2023-24) → MNFSR (2024-25) → PBS (to 2008-09) |

**Source abbreviations:**
- **PBS:** *Crops Area and Production by Districts 1981-82 to 2008-09*.
- **MNFSR:** *Crops Area & Production (District wise)*, volumes 2021-22, 2022-23 and 2024-25.

**Validation:** `validation_source_overlaps.csv` (output of `y08`) compares overlapping sources:

| Sources compared | Seasons | Pairs | Result |
|---|---|---|---|
| PBS vs Punjab CRS | 1990-91 to 2008-09 | 660 | r = 1.000, median difference 0.03% |
| MNFSR vs Punjab CRS | 2020-21 to 2024-25 | 185 | r = 1.000 |
| Balochistan DS vs MNFSR | 2021-22 to 2023-24 | 89 | r = 1.000 |
| PBS vs Sindh DS | 2005-06 to 2008-09 | 85 | r = 0.991, 95% within 5% |

The transcribed 2020-21 pages match the printed provincial totals (difference ≤ 0.03 thousand t or thousand ha).

## Corrections applied (documented in the scripts)
- **PBS 1981-82:** six new Punjab districts (Bhakkar, Khushab, Layyah, Okara, Rajanpur, Toba Tek Singh)
  report area but zero production; their production is inside the parent's figure. Their area was
  added to the parent district.
- **PBS page 29:** a production row labelled `2003-05` is read as 2004-05.
- **Sindh DS 2010:** Kashmore 2006-07 area printed as 3,366 ha; corrected to 33,366 ha from the 2009 edition.
- **Balochistan DS 2019:** Mastung 2017-18 area printed as 7 ha; set to missing.
- **MNFSR 2021-22 volume, 2020-21:** the Mansehra and Tor Ghar rows are swapped; swapped back.
- **Dropped:** ex-Frontier Regions and sub-divisions (`FR …`, `SD …`), which are under 1% of KP wheat area.
- **Merged into their 2022 district:**

| Unit | Merged into |
|---|---|
| Murree | Rawalpindi |
| Talagang | Chakwal |
| Wazirabad | Gujranwala |
| Kot Addu | Muzaffargarh |
| Taunsa | Dera Ghazi Khan |
| Tamboo | Nasirabad |
| Bolan | Kachhi |

## Known gaps and caveats
- **Balochistan 2009-10 to 2016-17** and **KP 2009-10** are `missing`. Those
  *Development Statistics* editions are not online: Balochistan's site only has 2019+, and the KP BOS
  site is a JavaScript app that blocks scripted access. Two other sources don't fill the gap:
  - FEWS NET has only division-level data for these years.
  - The Balochistan *Trends Analysis 2010–2023* is province-level only.
- **KP 2010-11 to 2015-16:** the 7 ex-FATA districts are `missing`. The 2014–2017 KP editions cover settled
  districts only; ex-FATA districts are reported from 2016-17.
- **Punjab CRS area revision from 2019-20:** areas of some districts jump (e.g. Islamabad 29 → 85
  thousand acres), apparently a revision in the area estimates. Consider a break dummy at 2019-20.
- **`inherited` rows** are approximations: the parent's yield stands in for the new district. Use the
  `flag` column to exclude them, or to weight them down.
