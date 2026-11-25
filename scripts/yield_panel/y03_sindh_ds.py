"""Sindh district wheat area (ha) and production (t) from Sindh Bureau of Statistics
'Development Statistics of Sindh' editions 2009-2021 (table 'AREA AND PRODUCTION OF WHEAT BY
DISTRICT IN SINDH, yyyy-yy TO yyyy-yy', 3 seasons each). Where editions overlap, the latest
edition (revised figures) is kept. Output: data_processed/yield_panel/src_sindh_ds.csv
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from pypdf import PdfReader

D = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_raw/04_crop/yields_provincial/sindh_bos/")
OUT = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_processed/yield_panel/src_sindh_ds.csv")
EDITIONS = {  # file -> edition year (used to prefer revised figures)
    "Development-Statistics-2009-.pdf": 2009, "Development-Statistics-2010-.pdf": 2010,
    "Develpment-Statistics-2011-with-Graph.pdf": 2011, "DS-2012.pdf": 2012,
    "Development-Statistics-2013-.pdf": 2013, "DS-2014.pdf": 2014,
    "Development-Statistics-2015-final.pdf": 2015, "Development-Statistics-2016-final.pdf": 2016,
    "Development-Statistics-of-Sindh-2017.pdf": 2017, "Development-Statistics-of-Sindh-2018.pdf": 2018,
    "Development-Statistics-of-Sindh-2019.pdf": 2019, "DS-2020_25-01-2022.pdf": 2020,
    "Development_Statistics_-_2021.pdf": 2021,
}
TITLE = re.compile(r"WHEAT\s+BY\s+DISTRICT\s+IN\s+SINDH,?\s*(\d{4})\s*-\s*(\d{2,4})\s+TO\s+(\d{4})\s*-\s*(\d{2,4})", re.I | re.S)
VAL = re.compile(r"^(\.\.|-|[\d,]+(\.\d+)?)$")


def num(t):
    return np.nan if t in ("..", "-") else float(t.replace(",", ""))


rows = []
for f, ed in EDITIONS.items():
    r = PdfReader(D + f)
    hit = None
    for i, p in enumerate(r.pages[90:170], start=90):
        t = p.extract_text() or ""
        flat = " ".join(t.split()).upper()
        if "WHEAT BY DISTRICT" in flat or ("WHEAT" in flat and "DISTRICT" in flat and "LARKANA" in flat
                                            and "TABLE 2.4.12" in flat):
            hit = (i, t)
            break
    if not hit:
        print("NOT FOUND", f)
        continue
    i, t = hit
    seasons = None
    for line in t.split("\n"):
        toks = re.findall(r"\b(?:19|20)\d\d-\d\d\b", line)
        if len(toks) >= 3 and len(toks) == len(line.split()):
            seasons = list(dict.fromkeys(toks))
            break
    if seasons is None:
        print("NO SEASON HEADER", f, i + 1)
        continue
    n, bad = len(seasons), []
    for line in t.split("\n"):
        toks = line.split()
        k = next((j for j, x in enumerate(toks) if VAL.match(x)), None)
        if not k:
            continue
        name, vals = " ".join(toks[:k]), toks[k:]
        if not re.match(r"^[A-Za-z]", name) or name.replace(" ", "").upper().startswith("SINDH"):
            continue
        if len(vals) != 2 * n or not all(VAL.match(v) for v in vals):
            bad.append(line)
            continue
        for j, s in enumerate(seasons):
            rows.append({"district_src": name, "season": s, "area_ha": num(vals[j]),
                         "prod_t": num(vals[n + j]), "edition": ed, "page": i + 1})
    print(f, "page", i + 1, seasons, "rows", sum(1 for x in rows if x["edition"] == ed), "skipped", bad)

df = pd.DataFrame(rows)
CANON = {"jamsharo": "Jamshoro", "jamshoro": "Jamshoro", "sbabad": "Shaheed Benazirabad",
         "sbenazirabad": "Shaheed Benazirabad", "nawabshah": "Shaheed Benazirabad",
         "kambershahdadkot": "Kambar Shahdadkot", "nausheroferoze": "Naushahro Feroze",
         "tandoallahyar": "Tando Allahyar", "tandomuhammadkha": "Tando Muhammad Khan",
         "tandomuhammadkhan": "Tando Muhammad Khan", "tandomohammadkhan": "Tando Muhammad Khan"}
key = df.district_src.str.lower().str.replace(r"[^a-z]", "", regex=True)
df["district_src"] = key.map(CANON).fillna(df.district_src.str.replace(r"\s+", " ", regex=True).str.strip())
df = df.sort_values("edition").groupby(["district_src", "season"], as_index=False).last()
df.loc[(df.area_ha == 0) | (df.prod_t == 0), ["area_ha", "prod_t"]] = np.nan
# typo in the 2010 edition: Kashmore 2006-07 area '3,366' (2009 edition: 33,366 ha)
df.loc[(df.district_src == "Kashmore") & (df.season == "2006-07"), "area_ha"] = 33366
df["province"] = "Sindh"
df["source"] = "Sindh Bureau of Statistics, Development Statistics of Sindh"
df[["source", "province", "district_src", "season", "area_ha", "prod_t", "edition", "page"]].to_csv(OUT, index=False)
print(sorted(df.district_src.unique()))
t = df.groupby("season")[["area_ha", "prod_t"]].sum()
print((t / 1000).round(1).to_string())
