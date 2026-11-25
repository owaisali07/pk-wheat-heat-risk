"""District wheat area and production from MNFSR 'Crops Area & Production (District wise)'
volumes, Table 1 (each volume holds two seasons):
  MNFSR_crops_district_2021-22.pdf -> 2020-21, 2021-22 (only the Punjab page has a text layer;
                                      Sindh/KP/Balochistan pages are scanned images)
  MNFSR_crops_district_2022_23.pdf -> 2021-22, 2022-23
  MNFSR_crops_district_2024_25.pdf -> 2023-24, 2024-25
Rows: name + area s1, area s2, prod s1, prod s2 ('000 ha, '000 t); the 2022-23 and 2024-25 volumes
add a % share after each value. Provinces come in the order Punjab, Sindh, KP, Balochistan, each
block closed by a 'Total' row, which is used to check the district sums.
Output: data_processed/yield_panel/src_mnfsr.csv
"""
import os
import re

import numpy as np
import pandas as pd
from pypdf import PdfReader

D = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_raw/04_crop/yields_pbs_crs/")
OUT = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_processed/yield_panel/src_mnfsr.csv")
VOLS = {"mnfsr_district_volumes/MNFSR_crops_district_2021-22.pdf": (2022, 4),
        "MNFSR_crops_district_2022_23.pdf": (2023, 8),
        "MNFSR_crops_district_2024_25.pdf": (2025, 8)}
ORDER = ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan"]
N = r"(-?[\d,]*\.?\d+)"
TOT = re.compile(r"\bTOTAL\s+" + r"\s+".join([N] * 4), re.I)

rows, checks = [], []
for f, (vol, nv) in VOLS.items():
    rec = re.compile(r"([A-Za-z][A-Za-z .'()\-]*?)\s+" + r"\s+".join([N] * nv) + r"(?=\s|$)")
    r = PdfReader(D + f)
    start = next(i for i, p in enumerate(r.pages[:40])
                 if "WHEAT CROP" in " ".join((p.extract_text() or "").split()).upper())
    seasons, pi, stop = None, 0, False
    for i in range(start, start + 6):
        t = r.pages[i].extract_text() or ""
        if seasons is None:
            seasons = list(dict.fromkeys(re.findall(r"20\d\d-\d\d", t)))[:2]
        for line in t.split("\n"):
            mt = TOT.search(line)
            if mt:
                checks.append((vol, ORDER[pi], [float(x.replace(",", "")) for x in mt.groups()]))
                pi += 1
                if pi == 4:
                    stop = True
                    break
                continue
            for m in rec.finditer(line):
                name = " ".join(m.group(1).split())
                v = [float(x.replace(",", "")) for x in m.groups()[1:]]
                if nv == 8:
                    v = v[0::2]  # drop % shares
                key = name.upper()
                if key.startswith("PAKISTAN") or key.startswith("PROVINCE"):
                    continue
                for j, s in enumerate(seasons):
                    rows.append({"district_src": name, "province": ORDER[pi], "season": s,
                                 "area_kha": v[j], "prod_kt": v[2 + j], "volume": vol, "page": i + 1})
            if stop:
                break
        if stop:
            break
    print(f, "pages", start + 1, "-", i + 1, seasons, "provinces read", pi)

df = pd.DataFrame(rows)
for vol, prov, v in checks:
    sub = df[(df.volume == vol) & (df.province == prov)]
    ss = sorted(sub.season.unique())
    got = [sub[sub.season == s].area_kha.sum() for s in ss] + [sub[sub.season == s].prod_kt.sum() for s in ss]
    ok = np.allclose(got, v, rtol=0.005, atol=0.5)
    print("check", vol, prov, "OK" if ok else "MISMATCH", "printed", v, "sum", [round(g, 2) for g in got])
df = df.sort_values("volume").groupby(["province", "district_src", "season"], as_index=False).last()
df.loc[(df.area_kha == 0) | (df.prod_kt == 0), ["area_kha", "prod_kt"]] = np.nan
df["area_ha"] = df.area_kha * 1000
df["prod_t"] = df.prod_kt * 1000
df["source"] = "MNFSR Crops Area & Production (District wise)"
df[["source", "province", "district_src", "season", "area_ha", "prod_t", "volume", "page"]].to_csv(OUT, index=False)
print((df.groupby(["season", "province"])[["area_ha", "prod_t"]].sum() / 1000).round(1).unstack().to_string())
