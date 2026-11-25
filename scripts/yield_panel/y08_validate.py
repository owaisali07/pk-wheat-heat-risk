"""Cross-check overlapping sources (same 2022 district, same season) and FEWS NET.
Writes data_processed/yield_panel/validation_source_overlaps.csv and prints a summary.
"""
import os
import itertools
import re
import sys

import numpy as np
import pandas as pd

sys.argv = ["y07"]  # reuse the harmonised source table from y07 without re-running its outputs
P = os.path.join(os.environ.get("PAPER8_ROOT", "."), r"data_processed/yield_panel/")
ns = {}
src = open(os.path.join(os.environ.get("PAPER8_ROOT", "."), r"scripts/yield_panel/y07_harmonise.py"), encoding="utf-8").read()
exec(src.split("# ------------------------------------------------------------------ choose source")[0], ns)
agg = ns["agg"]

rows = []
for (u, s), g in agg.groupby(["unit", "season"]):
    for a, b in itertools.combinations(g.itertuples(), 2):
        rows.append({"unit": u, "season": s, "src_a": a.src, "src_b": b.src, "yield_a": a.yield_t_ha,
                     "yield_b": b.yield_t_ha, "area_a": a.area_ha, "area_b": b.area_ha})
ov = pd.DataFrame(rows)
ov["pair"] = ov.src_a + " vs " + ov.src_b
ov["yield_pct_diff"] = 100 * (ov.yield_b - ov.yield_a) / ov.yield_a
ov["area_pct_diff"] = 100 * (ov.area_b - ov.area_a) / ov.area_a
ov.to_csv(P + "validation_source_overlaps.csv", index=False)
summ = ov.groupby("pair").agg(n=("unit", "size"), seasons=("season", lambda s: f"{s.min()}..{s.max()}"),
                              r_yield=("yield_a", lambda x: np.corrcoef(x, ov.loc[x.index, "yield_b"])[0, 1]),
                              med_abs_yield_diff_pct=("yield_pct_diff", lambda x: x.abs().median()),
                              share_within_5pct=("yield_pct_diff", lambda x: (x.abs() <= 5).mean()))
print(summ.round(3).to_string())
print("\nlargest disagreements (|yield diff| > 25%):")
print(ov[ov.yield_pct_diff.abs() > 25].sort_values("yield_pct_diff")[
    ["unit", "season", "pair", "yield_a", "yield_b", "area_a", "area_b"]].round(2).to_string())
