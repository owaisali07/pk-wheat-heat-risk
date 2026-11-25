"""Build the district wheat yield panel 1981-82 to 2024-25 on the 2022 OCHA admin2 boundaries
(Punjab, Sindh, Khyber Pakhtunkhwa, Balochistan, Islamabad; AJK and Gilgit-Baltistan have no data).

Inputs (data_processed/yield_panel/): src_pbs_28yr, src_punjab_crs, src_sindh_ds, src_kp_ds,
src_balochistan_ds, src_mnfsr, manual_mnfsr_2021-22_scanned_pages (Sindh/KP 2020-21, transcribed
import os
from scanned pages and checked against the printed province totals).

Harmonisation, per source record:
  1. alias  - spelling variants -> 2022 district name (or a historical unit: Karachi, Kohistan, Chitral)
  2. merge  - units that are now part of a 2022 district are summed into it (Murree -> Rawalpindi,
              Talagang -> Chakwal, Wazirabad -> Gujranwala, Kot Addu -> Muzaffargarh,
              Taunsa -> Dera Ghazi Khan, Tamboo -> Nasirabad, Bolan -> Kachhi)
  3. drop   - ex-Frontier Regions / sub-divisions (FR ..., SD ...), < 1% of KP wheat area
Then, per 2022 district and season, the highest-priority source with data is taken. A district with
no own value inherits the yield of its parent (the district it was split from, or the historical
unit it belongs to) in that season: flag 'inherited', area/production left empty so totals are
not double counted.
Output: data_processed/yield_panel/wheat_yield_district_panel_1982_2025.csv (+ coverage table)
"""
import re

import geopandas as gpd
import numpy as np
import pandas as pd

P = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_processed/yield_panel/")
ADM = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_raw/01_boundaries/pak_admin2_simplified.geojson")

# ------------------------------------------------------------------ load sources
SRC = {"pbs": "src_pbs_28yr.csv", "punjab_crs": "src_punjab_crs.csv", "sindh_ds": "src_sindh_ds.csv",
       "kp_ds": "src_kp_ds.csv", "bal_ds": "src_balochistan_ds.csv", "mnfsr": "src_mnfsr.csv"}
parts = []
for k, f in SRC.items():
    d = pd.read_csv(P + f)[["province", "district_src", "season", "area_ha", "prod_t"]]
    parts.append(d.assign(src=k))
m = pd.read_csv(P + "manual_mnfsr_2021-22_scanned_pages.csv")
for s, a, p in [("2020-21", "area_kha_2020_21", "prod_kt_2020_21")]:
    parts.append(pd.DataFrame({"province": m.province, "district_src": m.district_src, "season": s,
                               "area_ha": m[a] * 1000, "prod_t": m[p] * 1000, "src": "mnfsr_scan"}))
raw = pd.concat(parts, ignore_index=True)
# MNFSR 2021-22 volume, 2020-21: Mansehra and Tor Ghar rows are swapped (Tor Ghar 36.6 kha, Mansehra
# 6.1 kha; every other year Mansehra ~36 kha, Tor Ghar ~6 kha)
sw = (raw.src == "mnfsr_scan") & (raw.season == "2020-21")
a = sw & (raw.district_src == "Torgar")
b = sw & (raw.district_src == "Mansehra")
raw.loc[a, "district_src"], raw.loc[b, "district_src"] = "Mansehra", "Torgar"

# ------------------------------------------------------------------ name harmonisation
ALIAS = {  # normalised key -> canonical name (2022 adm2_name, or historical unit)
    # Punjab
    "dgkhan": "Dera Ghazi Khan", "deraghazikhan": "Dera Ghazi Khan", "layyah": "Leiah",
    "mgarh": "Muzaffargarh", "mbdin": "Mandi Bahauddin", "nankanasb": "Nankana Sahib",
    "ryKhan": "Rahim Yar Khan", "rykhan": "Rahim Yar Khan", "rahimyarkhan": "Rahim Yar Khan",
    "ttsingh": "Toba Tek Singh", "muree": "Murree",
    # Sindh
    "abad": "Shaheed Benazir Abad", "shaheedbenazirabad": "Shaheed Benazir Abad", "kot": "Kambar Shahdad Kot",
    "kambarshahdadkot": "Kambar Shahdad Kot", "kambarshahdatkot": "Kambar Shahdad Kot",
    "nferoze": "Naushahro Feroze", "mirpurkhas": "Mirpur Khas", "umerkot": "Umer Kot", "umarkot": "Umer Kot",
    "tmkhan": "Tando Muhammad Khan",
    # KP
    "bajour": "Bajaur", "battagram": "Batagram", "bunir": "Buner", "charsada": "Charsadda",
    "dikhan": "D. I. Khan", "deraismailkhan": "D. I. Khan", "dirlower": "Lower Dir", "dirupper": "Upper Dir",
    "lakki": "Lakki Marwat", "muhmand": "Mohmand", "nwaziristan": "North Waziristan",
    "swaziristan": "South Waziristan", "oriakzai": "Orakzai", "sawabi": "Swabi", "torgar": "Tor Ghar",
    # Balochistan
    "derabughti": "Dera Bugti", "gawadar": "Gwadar", "kabdullah": "Killa Abdullah", "turbat": "Kech",
    "musakhail": "Musakhel", "panjgoor": "Panjgur", "punjgoor": "Panjgur",
}
MERGE = {"Murree": "Rawalpindi", "Talagang": "Chakwal", "Wazirabad": "Gujranwala", "Kot Addu": "Muzaffargarh",
         "Taunsa": "Dera Ghazi Khan", "Tamboo": "Nasirabad", "Bolan": "Kachhi"}
# parent used when a 2022 district has no own value in a season (split year in brackets)
PARENT = {
    "Chiniot": "Jhang", "Nankana Sahib": "Sheikhupura", "Mandi Bahauddin": "Gujrat", "Hafizabad": "Gujranwala",
    "Narowal": "Sialkot", "Pakpattan": "Sahiwal", "Lodhran": "Multan", "Okara": "Sahiwal",
    "Toba Tek Singh": "Faisalabad", "Khushab": "Sargodha", "Bhakkar": "Mianwali", "Leiah": "Muzaffargarh",
    "Rajanpur": "Dera Ghazi Khan", "Chakwal": "Jhelum",
    "Kashmore": "Jacobabad", "Kambar Shahdad Kot": "Larkana", "Jamshoro": "Dadu", "Matiari": "Hyderabad",
    "Tando Allahyar": "Hyderabad", "Tando Muhammad Khan": "Hyderabad", "Mirpur Khas": "Tharparkar",
    "Umer Kot": "Mirpur Khas", "Ghotki": "Sukkur", "Naushahro Feroze": "Shaheed Benazir Abad",
    "Sujawal": "Thatta", **{k: "Karachi" for k in ["Central Karachi", "East Karachi", "Korangi Karachi",
                                                   "Malir Karachi", "South Karachi", "West Karachi"]},
    "Batagram": "Mansehra", "Tor Ghar": "Mansehra", "Shangla": "Swat", "Buner": "Swat", "Upper Dir": "Lower Dir",
    "Hangu": "Kohat", "Karak": "Kohat", "Charsadda": "Peshawar", "Nowshera": "Peshawar", "Swabi": "Mardan",
    "Haripur": "Abbottabad", "Lakki Marwat": "Bannu", "Tank": "D. I. Khan",
    "Kohistan Lower": "Kohistan", "Kohistan Upper": "Kohistan", "Kolai Palas Kohistan": "Kohistan",
    "Chitral Lower": "Chitral", "Chitral Upper": "Chitral",
    "Nushki": "Chagai", "Washuk": "Kharan", "Sherani": "Zhob", "Harnai": "Sibi", "Lehri": "Sibi",
    "Sohbatpur": "Jaffarabad", "Duki": "Loralai", "Shaheed Sikandarabad": "Kalat", "Chaman": "Killa Abdullah",
    "Musakhel": "Loralai", "Killa Saifullah": "Zhob", "Killa Abdullah": "Pishin", "Barkhan": "Loralai",
    "Ziarat": "Sibi", "Jhal Magsi": "Kachhi", "Jaffarabad": "Nasirabad", "Mastung": "Kalat", "Awaran": "Khuzdar",
}
PRIORITY = {  # first source with data wins
    "Punjab": ["punjab_crs", "pbs", "mnfsr"],
    "Islamabad": ["punjab_crs", "pbs", "mnfsr"],
    "Sindh": ["mnfsr", "mnfsr_scan", "sindh_ds", "pbs"],
    "Khyber Pakhtunkhwa": ["mnfsr", "mnfsr_scan", "kp_ds", "pbs"],
    "Balochistan": ["bal_ds", "mnfsr", "pbs"],
}

adm = gpd.read_file(ADM)[["adm2_pcode", "adm2_name", "adm1_name"]]
adm = adm[~adm.adm1_name.isin(["Azad Kashmir", "Gilgit Baltistan"])]
NAMES = set(adm.adm2_name) | {"Karachi", "Kohistan", "Chitral"} | set(MERGE)


def canon(n):
    k = re.sub(r"[^a-z]", "", n.lower())
    if k in ALIAS:
        return ALIAS[k]
    for c in NAMES:
        if re.sub(r"[^a-z]", "", c.lower()) == k:
            return c
    return None


raw["unit"] = raw.district_src.map(canon)
fr = raw.district_src.str.match(r"^(FR|SD) ")
unk = raw[raw.unit.isna() & ~fr].district_src.unique()
assert len(unk) == 0, f"unmatched names: {unk}"
raw = raw[~fr].copy()
raw["unit"] = raw.unit.replace(MERGE)
raw.loc[raw.unit == "Islamabad", "province"] = "Islamabad"
# sum merged units into their 2022 district (per source & season); a unit with no data stays NaN
agg = raw.groupby(["src", "province", "unit", "season"]).agg(
    area_ha=("area_ha", lambda s: s.sum(min_count=1)), prod_t=("prod_t", lambda s: s.sum(min_count=1)),
    n_units=("district_src", "nunique")).reset_index()
agg = agg[agg.area_ha.notna() & agg.prod_t.notna() & (agg.area_ha > 0)]
agg["yield_t_ha"] = agg.prod_t / agg.area_ha

# ------------------------------------------------------------------ choose source per unit-season
agg["rank"] = [PRIORITY[p].index(s) for p, s in zip(agg.province, agg.src)]
best = agg.sort_values("rank").groupby(["unit", "season"], as_index=False).first()
seasons = [f"{y}-{(y + 1) % 100:02d}" for y in range(1981, 2025)]
own = best.set_index(["unit", "season"])

rows = []
for _, r in adm.iterrows():
    for s in seasons:
        rec = {"adm2_pcode": r.adm2_pcode, "adm2_name": r.adm2_name, "province": r.adm1_name, "season": s,
               "year": int(s[:4]) + 1}
        if (r.adm2_name, s) in own.index:
            o = own.loc[(r.adm2_name, s)]
            rec.update(area_ha=o.area_ha, prod_t=o.prod_t, yield_t_ha=o.yield_t_ha, source=o.src,
                       flag="own" if o.n_units == 1 else "own_merged_units", from_unit=r.adm2_name)
        else:
            u = PARENT.get(r.adm2_name)
            while u is not None and (u, s) not in own.index:
                u = PARENT.get(u)
            if u is not None:
                o = own.loc[(u, s)]
                rec.update(yield_t_ha=o.yield_t_ha, source=o.src, flag="inherited", from_unit=u)
            else:
                rec.update(flag="missing")
        rows.append(rec)
panel = pd.DataFrame(rows)

# ------------------------------------------------------------------ quality flags
panel["qc"] = ""
panel.loc[(panel.yield_t_ha < 0.3) | (panel.yield_t_ha > 6.5), "qc"] = "implausible_level"
g = panel.groupby("adm2_pcode").yield_t_ha
med = g.transform(lambda x: x.rolling(9, center=True, min_periods=4).median())
mad = g.transform(lambda x: (x - x.median()).abs().median())
z = (panel.yield_t_ha - med) / (1.4826 * mad.replace(0, np.nan))
panel.loc[(z.abs() > 5) & (panel.qc == ""), "qc"] = "outlier_vs_local_median"
panel["robust_z"] = z.round(2)

out = P + "wheat_yield_district_panel_1982_2025.csv"
panel.to_csv(out, index=False)
print("rows", len(panel), "| flags:", panel.flag.value_counts().to_dict(), "| qc:", panel.qc.value_counts().to_dict())

# coverage table: share of districts with own / inherited data, by province and period
panel["period"] = pd.cut(panel.year, [1981, 1990, 2000, 2009, 2013, 2017, 2020, 2025],
                         labels=["1982-90", "1991-00", "2001-09", "2010-13", "2014-17", "2018-20", "2021-25"])
cov = panel.pivot_table(index="province", columns="period", values="flag", observed=False,
                        aggfunc=lambda f: f"{(f.str.startswith('own')).mean():.0%} own / {(f == 'inherited').mean():.0%} inh")
cov.to_csv(P + "coverage_by_province_period.csv")
print(cov.to_string())

# provincial totals from 'own' records (check against official series)
t = panel[panel.flag.str.startswith("own")].groupby(["province", "season"])[["area_ha", "prod_t"]].sum() / 1e3
print(t.unstack(0).loc[["1981-82", "1990-91", "2000-01", "2008-09", "2012-13", "2016-17", "2019-20",
                        "2022-23", "2024-25"]].round(0).to_string())
print(panel[panel.qc != ""][["adm2_name", "season", "area_ha", "prod_t", "yield_t_ha", "source", "qc"]].to_string())
