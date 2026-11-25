"""Punjab district wheat area and production from Crop Reporting Service, Punjab:
  WHEAT_HISTORICAL.pdf                      1990-91 to 2021-22 (32 seasons)
  Historical_Wheat_districtwise_2022_25.pdf 2022-23 to 2024-25
Page 1 = area ('000 acres), page 3 = production ('000 tonnes); '--' = district did not exist.
Output: data_processed/yield_panel/src_punjab_crs.csv
"""
import os
import re

import numpy as np
import pandas as pd
from pypdf import PdfReader

D = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_raw/04_crop/yields_pbs_crs/crs/")
OUT = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_processed/yield_panel/src_punjab_crs.csv")
ACRE_HA = 0.40468564224
FILES = {"WHEAT_HISTORICAL.pdf": [f"{y}-{(y + 1) % 100:02d}" for y in range(1990, 2022)],
         "Historical_Wheat_districtwise_2022_25.pdf": ["2022-23", "2023-24", "2024-25"]}
TOK = re.compile(r"^(--|-?\d+(\.\d+)?)$")

rows = []
for f, seasons in FILES.items():
    r = PdfReader(D + f)
    for page, var in [(0, "area_kac"), (2, "prod_kt")]:
        for line in r.pages[page].extract_text().split("\n"):
            toks = line.split()
            i = next((k for k, t in enumerate(toks) if TOK.match(t)), None)
            if not i:
                continue
            name, vals = " ".join(toks[:i]), toks[i:]
            if name.upper() == "PUNJAB" or (len(vals) < len(seasons) and len(seasons) > 3):
                continue
            # districts created in the last season (e.g. Taunsa 2024-25) list fewer values: right-align
            vals = ["--"] * (len(seasons) - len(vals)) + vals if len(vals) < len(seasons) else vals[:len(seasons)]
            for s, v in zip(seasons, vals):
                rows.append({"district_src": name, "season": s, var: np.nan if v == "--" else float(v), "file": f})
df = pd.DataFrame(rows).groupby(["district_src", "season"], as_index=False).agg(
    area_kac=("area_kac", "first"), prod_kt=("prod_kt", "first"), file=("file", "first"))
df.loc[(df.area_kac == 0) | (df.prod_kt == 0), ["area_kac", "prod_kt"]] = np.nan
df["district_src"] = df.district_src.replace({"Cahkwal": "Chakwal"})
df["area_ha"] = df.area_kac * 1000 * ACRE_HA
df["prod_t"] = df.prod_kt * 1000
df["province"] = "Punjab"
df["source"] = "Crop Reporting Service Punjab (district wheat historical series)"
df[["source", "province", "district_src", "season", "area_ha", "prod_t", "file"]].to_csv(OUT, index=False)
print(df.shape, df.district_src.nunique(), "districts")
print(sorted(df.district_src.unique()))
t = df.groupby("season")[["area_ha", "prod_t"]].sum()
t["yield"] = t.prod_t / t.area_ha
print(t.round(0).iloc[[0, 10, 19, 29, 31, 32, 33, 34]].to_string())
