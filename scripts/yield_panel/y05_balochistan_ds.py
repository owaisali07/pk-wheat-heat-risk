"""Balochistan district wheat area (ha) and production (t), from 'Development Statistics of
Balochistan' editions 2019-2024 (table 'IRRIGATED AND UN-IRRIGATED AREA AND PRODUCTION OF WHEAT IN
BALOCHISTAN BY DISTRICT', source: Directorate of Crop Reporting Services Balochistan).
Each edition has 2 seasons: [area irr, unirr, total, prod irr, unirr, total] x 2 -> 2017-18 to 2022-23.
Output: data_processed/yield_panel/src_balochistan_ds.csv
"""
import os
import re

import numpy as np
import pandas as pd
from pypdf import PdfReader

D = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_raw/04_crop/yields_provincial/balochistan/")
OUT = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_processed/yield_panel/src_balochistan_ds.csv")
VAL = re.compile(r"^(-|[\d,]+)$")

rows = []
for ed in range(2019, 2025):
    r = PdfReader(D + f"Development_Statistics_Balochistan_{ed}.pdf")
    hit = None
    for i in range(20, 60):
        t = r.pages[i].extract_text() or ""
        u = " ".join(t.split()).upper()
        if "AREA AND PRODUCTION OF WHEAT" in u and "BY DISTRICT" in u and "QUETTA" in u:
            hit = (i, t)
            break
    assert hit, ed
    i, t = hit
    seasons = sorted(set(re.findall(r"20\d\d-\d\d", " ".join(t.split()).split("BY DISTRICT")[-1])))
    assert len(seasons) == 2, (ed, seasons)
    for line in t.split("\n"):
        toks = line.split()
        k = next((j for j, x in enumerate(toks) if VAL.match(x)), None)
        if not k:
            continue
        name, vals = " ".join(toks[:k]), toks[k:]
        if name.lower().startswith("balochistan") or len(vals) < 11:
            continue
        v = [np.nan if x == "-" else float(x.replace(",", "")) for x in vals] + [np.nan] * (12 - len(vals))
        for j, s in enumerate(seasons):
            a, p = v[6 * j + 2], v[6 * j + 5]
            rows.append({"district_src": name, "season": s, "area_ha": a, "prod_t": p,
                         "area_irr_ha": v[6 * j], "edition": ed, "page": i + 1})
    print(ed, "page", i + 1, seasons, sum(x["edition"] == ed for x in rows))

df = pd.DataFrame(rows)
df["district_src"] = df.district_src.replace({"Chaghi": "Chagai", "Dera Bughti": "Dera Bugti"})
df = df.sort_values("edition").groupby(["district_src", "season"], as_index=False).last()
df.loc[(df.area_ha == 0) | (df.prod_t == 0), ["area_ha", "prod_t"]] = np.nan
# typo in the 2019 edition: Mastung 2017-18 area printed as 7 ha (production 7,269 t)
df.loc[(df.district_src == "Mastung") & (df.season == "2017-18"), "area_ha"] = np.nan
df["province"] = "Balochistan"
df["source"] = "Development Statistics of Balochistan (CRS Balochistan)"
df["yield_t_ha"] = df.prod_t / df.area_ha
print(df[(df.yield_t_ha > 5) | (df.yield_t_ha < 0.3)][["district_src", "season", "area_ha", "prod_t"]].to_string())
df[["source", "province", "district_src", "season", "area_ha", "prod_t", "area_irr_ha", "edition", "page"]].to_csv(OUT, index=False)
print(sorted(df.district_src.unique()))
print((df.groupby("season")[["area_ha", "prod_t"]].sum() / 1000).round(1).to_string())
