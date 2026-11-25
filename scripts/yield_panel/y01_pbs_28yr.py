"""Extract district wheat area and production, 1981-82 to 2008-09, from
PBS 'Crops Area and Production by Districts' (Table 7-10, PDF pages 16-29).

Each page: 28 area rows ('000 ha) followed by 28 production rows ('000 t) for 7-10 districts.
Numbers are placed in columns by their x position (some cells are blank or run together).
Output: data_processed/yield_panel/src_pbs_28yr.csv (long format).
"""
import os
import re

import numpy as np
import pandas as pd
import pdfplumber

PDF = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_raw/04_crop/yields_pbs_crs/pbs_area_production_by_districts_28yrs.pdf")
OUT = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_processed/yield_panel/src_pbs_28yr.csv")

PAGES = {  # pdf page -> (province, district names in column order)
    16: ("Punjab", ["Attock", "Rawalpindi", "Islamabad", "Jhelum", "Chakwal", "Sargodha", "Khushab", "Mianwali", "Bhakkar"]),
    17: ("Punjab", ["Faisalabad", "Toba Tek Singh", "Jhang", "Gujrat", "Mandi Bahauddin", "Sialkot", "Narowal", "Gujranwala", "Hafizabad"]),
    18: ("Punjab", ["Sheikhupura", "Nankana Sahib", "Lahore", "Kasur", "Okara", "Sahiwal", "Pakpattan", "Multan", "Lodhran"]),
    19: ("Punjab", ["Khanewal", "Vehari", "Muzaffargarh", "Layyah", "Dera Ghazi Khan", "Rajanpur", "Bahawalpur", "Rahim Yar Khan", "Bahawalnagar"]),
    20: ("Sindh", ["Khairpur", "Ghotki", "Sukkur", "Naushahro Feroze", "Shaheed Benazirabad", "Jacobabad", "Kashmore", "Shikarpur"]),
    21: ("Sindh", ["Larkana", "Kambar Shahdadkot", "Sanghar", "Tharparkar", "Mirpurkhas", "Umerkot", "Dadu", "Jamshoro"]),
    22: ("Sindh", ["Hyderabad", "Matiari", "Tando Allahyar", "Tando Muhammad Khan", "Badin", "Thatta", "Karachi"]),
    23: ("Khyber Pakhtunkhwa", ["Peshawar", "Charsadda", "Nowshera", "Mardan", "Swabi", "Kohat", "Hangu", "Karak", "Mansehra"]),
    24: ("Khyber Pakhtunkhwa", ["Battagram", "Abbottabad", "Haripur", "Kohistan", "Malakand", "Swat", "Buner", "Shangla", "Lower Dir"]),
    25: ("Khyber Pakhtunkhwa", ["Upper Dir", "Chitral", "Dera Ismail Khan", "Tank", "Bannu", "Lakki Marwat", "Mohmand", "Khyber", "Kurram"]),
    26: ("Khyber Pakhtunkhwa", ["Orakzai", "Bajaur", "North Waziristan", "South Waziristan", "FR Peshawar", "FR Kohat", "FR Bannu", "FR Dera Ismail Khan"]),
    27: ("Balochistan", ["Quetta", "Pishin", "Killa Abdullah", "Chagai", "Loralai", "Musakhel", "Barkhan", "Zhob", "Killa Saifullah"]),
    28: ("Balochistan", ["Sibi", "Ziarat", "Kohlu", "Kachhi", "Dera Bugti", "Nasirabad", "Jaffarabad", "Tamboo", "Bolan"]),
    29: ("Balochistan", ["Mastung", "Khuzdar", "Awaran", "Kharan", "Lasbela", "Kech", "Panjgur", "Gwadar", "Jhal Magsi", "Kalat"]),
}
YR = re.compile(r"^(19|20)\d\d-\d\d$")
NUM = re.compile(r"^-?\d+(\.\d+)?$")


def split_runs(w):
    """'21612.317375.0' -> two words with interpolated x positions."""
    parts = re.findall(r"\d+\.\d", w["text"]) if w["text"].count(".") > 1 else [w["text"]]
    if len(parts) == 1:
        return [w]
    width = (w["x1"] - w["x0"]) / len(w["text"])
    out, x = [], w["x0"]
    for p in parts:
        out.append({"text": p, "x0": x, "x1": x + width * len(p)})
        x += width * len(p)
    return out


rows = []
with pdfplumber.open(PDF) as pdf:
    for pno, (prov, names) in PAGES.items():
        words = pdf.pages[pno - 1].extract_words(keep_blank_chars=False, use_text_flow=False)
        lines = {}
        for w in words:
            lines.setdefault(round(w["top"]), []).append(w)
        # merge lines whose tops differ by <=2 pt
        keys = sorted(lines)
        merged, cur = [], None
        for k in keys:
            if cur is not None and k - cur[0] <= 2:
                cur[1].extend(lines[k])
            else:
                cur = [k, list(lines[k])]
                merged.append(cur)
        table = []
        for _, ws in merged:
            ws = sorted(ws, key=lambda w: w["x0"])
            if not YR.match(ws[0]["text"]):
                continue
            nums = []
            for w in ws[1:]:
                for s in split_runs(w):
                    if NUM.match(s["text"]):
                        nums.append(s)
            table.append((ws[0]["text"], nums))
        assert len(table) == 56, (pno, len(table))
        k = len(names)
        full = [n for _, n in table if len(n) == k]
        right = np.median(np.array([[w["x1"] for w in n] for n in full]), axis=0)
        for i, (yr, nums) in enumerate(table):
            vals = [np.nan] * k
            if len(nums) == k:
                vals = [float(w["text"]) for w in nums]
            else:
                for w in nums:
                    j = int(np.argmin(np.abs(right - w["x1"])))
                    vals[j] = float(w["text"])
            var = "area_kha" if i < 28 else "prod_kt"
            for d, v in zip(names, vals):
                rows.append({"province": prov, "district_src": d, "season": yr, var: v, "page": pno})

df = pd.DataFrame(rows)
df = df.groupby(["province", "district_src", "season"], as_index=False).agg(
    area_kha=("area_kha", "first"), prod_kt=("prod_kt", "first"), page=("page", "first"))
# 0.0 means the district did not exist yet (its area was reported under the parent district)
mm = (df.area_kha == 0) != (df.prod_kt == 0)
print("area/prod zero mismatch:"); print(df[mm][["district_src", "season", "area_kha", "prod_kt"]].to_string())
df["zero_mismatch"] = mm
# A new district with area but zero production: its production is inside the parent's figure,
# so add its area to the parent to keep the parent's yield right.
PARENT = {"Bhakkar": "Mianwali", "Khushab": "Sargodha", "Layyah": "Muzaffargarh", "Okara": "Sahiwal",
          "Rajanpur": "Dera Ghazi Khan", "Toba Tek Singh": "Faisalabad"}
for _, r in df[mm & (df.area_kha > 0)].iterrows():
    if r.district_src in PARENT:
        sel = (df.district_src == PARENT[r.district_src]) & (df.season == r.season)
        df.loc[sel, "area_kha"] += r.area_kha
df.loc[(df.area_kha == 0) | (df.prod_kt == 0), ["area_kha", "prod_kt"]] = np.nan
# page 29 production block labels the 2004-05 row '2003-05' (it follows the 2003-04 row)
df["season"] = df.season.replace({"2003-05": "2004-05"})
df = df.sort_values("area_kha").groupby(["province", "district_src", "season"], as_index=False).first()
df["area_ha"] = df.area_kha * 1000
df["prod_t"] = df.prod_kt * 1000
df["source"] = "PBS Crops Area & Production by Districts 1981-82 to 2008-09"
df[["source", "province", "district_src", "season", "area_ha", "prod_t", "zero_mismatch", "page"]].to_csv(OUT, index=False)
print(df.shape, "missing area", df.area_kha.isna().sum(), "missing prod", df.prod_kt.isna().sum())

# check against the province totals table (page 15) is done in y09_validate.py
tot = df.groupby(["province", "season"])[["area_ha", "prod_t"]].sum() / 1000
print(tot.unstack(0).loc[["1981-82", "1990-91", "2000-01", "2008-09"]].round(1).to_string())
