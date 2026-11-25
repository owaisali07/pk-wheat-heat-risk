"""KP district wheat area (ha) and production (t) from 'Development Statistics of Khyber
Pakhtunkhwa' (Table 15, source: Crop Reporting Services KP), editions 2014-2017, 2020, 2021
(3 seasons each -> 2010-11 to 2019-20). Latest edition wins where seasons overlap.
Output: data_processed/yield_panel/src_kp_ds.csv
"""
import os
import re

import numpy as np
import pandas as pd
from pypdf import PdfReader

D = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_raw/04_crop/yields_provincial/kp_development_statistics/")
OUT = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_processed/yield_panel/src_kp_ds.csv")
EDITIONS = {"KP_Development_Statistics_2014.pdf": (2014, 41),
            "KP_Development_Statistics_2015.pdf": (2015, 44),
            "KP_Development_Statistics_2016.pdf": (2016, 42),
            "KP_Development_Statistics_2017.pdf": (2017, 42),
            "KP_Development_Statistics_2020.pdf": (2020, 42),
            "KP_Development_Statistics_2021.pdf": (2021, 44)}
VAL = re.compile(r"^(-|[\d,]+)$")

rows = []
for f, (ed, pno) in EDITIONS.items():
    t = PdfReader(D + f).pages[pno - 1].extract_text()
    assert "WHEAT" in " ".join(t.split()).upper(), f
    seasons = sorted(set(re.findall(r"20\d\d-\d\d", t)))
    assert len(seasons) == 3, (f, seasons)
    lines, pending = [], None
    for line in t.split("\n"):
        toks = line.split()
        if not toks:
            continue
        if pending and VAL.match(toks[0]):  # name on its own line, numbers on the next ('Khyber')
            toks = [pending] + toks
        pending = None
        k = next((j for j, x in enumerate(toks) if VAL.match(x)), None)
        if k is None:
            if len(toks) == 1 and toks[0].isalpha():
                pending = toks[0]
            continue
        if k == 0:
            continue
        name, vals = " ".join(toks[:k]), toks[k:]
        if name.lower().startswith(("pakhtunkhwa", "khyber pakhtunkhwa")) or len(vals) != 9:
            continue
        v = [np.nan if x == "-" else float(x.replace(",", "")) for x in vals]
        for j, s in enumerate(seasons):
            rows.append({"district_src": name, "season": s, "area_ha": v[3 * j], "prod_t": v[3 * j + 1],
                         "yield_pub": v[3 * j + 2], "edition": ed, "page": pno})
    print(f, seasons, sum(r["edition"] == ed for r in rows))

df = pd.DataFrame(rows).sort_values("edition").groupby(["district_src", "season"], as_index=False).last()
df.loc[(df.area_ha == 0) | (df.prod_t == 0), ["area_ha", "prod_t"]] = np.nan
df["province"] = "Khyber Pakhtunkhwa"
df["source"] = "Development Statistics of Khyber Pakhtunkhwa (CRS KP)"
chk = (df.prod_t / df.area_ha * 1000 - df.yield_pub).abs()
print("yield check: max abs diff (kg/ha)", round(chk.max(), 1))
df[["source", "province", "district_src", "season", "area_ha", "prod_t", "edition", "page"]].to_csv(OUT, index=False)
print(sorted(df.district_src.unique()))
print((df.groupby("season")[["area_ha", "prod_t"]].sum() / 1000).round(1).to_string())
