"""Assemble the analysis parts (# %% cell markers) into one Jupyter notebook: Paper8_Analysis.ipynb."""
import re
from pathlib import Path

import nbformat

HERE = Path(__file__).parent
PARTS = ["nb_part1_data.py", "nb_part2_models.py", "nb_part3_shap_hazard.py", "nb_part4_future.py",
         "nb_part5_vulnerability.py", "nb_part6_risk.py", "nb_part7_figures.py"]

cells = []
for p in PARTS:
    src = (HERE / p).read_text(encoding="utf-8")
    for chunk in re.split(r"^# %%", src, flags=re.M):
        if not chunk.strip():
            continue
        if chunk.startswith(" [markdown]"):
            body = chunk[len(" [markdown]"):].strip("\n")
            md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(md.strip()))
        else:
            cells.append(nbformat.v4.new_code_cell(chunk.strip("\n")))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbformat.write(nb, HERE / "Paper8_Analysis.ipynb")
print(f"Paper8_Analysis.ipynb: {len(cells)} cells")
