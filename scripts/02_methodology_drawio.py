"""Paper 8 Fig. 2: methodology flowchart as an editable draw.io diagram (black only).

Writes figures/Fig2_methodology.drawio and exports PDF and PNG with the draw.io
desktop CLI. Edit the .drawio file in draw.io directly for manual tweaks, or edit
this script and re-run it.

Sizing: the page is about 1120 px wide and is printed at the IEEE text width
(7.16 in), so 16 px text prints at about 7.4 pt, above the 7 pt minimum.
"""
import os
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(os.path.join(os.environ.get("PAPER8_ROOT", "."), r"figures"))
DRAWIO = Path.home() / "AppData/Local/Programs/draw.io/draw.io.exe"
FONT = "Times New Roman"

BOX = (f"rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#000000;strokeWidth=1;"
       f"fontFamily={FONT};fontSize=16;fontColor=#000000;align=center;verticalAlign=middle;spacing=6;")
KEY = BOX.replace("strokeWidth=1", "strokeWidth=3")
PANEL = ("rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#000000;strokeWidth=1;dashed=1;"
         "dashPattern=4 3;")
HEAD = (f"text;html=1;align=center;verticalAlign=middle;fontFamily={FONT};fontSize=16;fontStyle=1;"
        f"fontColor=#000000;")
FOOT = BOX.replace("fontSize=16", "fontSize=15")
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;endSize=7;"
        "strokeColor=#000000;strokeWidth=1.2;jettySize=auto;orthogonalLoop=1;")

ROWS = [90, 270, 450]
RH = 160
cells, n = [], [1]


def add(value, style, x, y, w, h):
    n[0] += 1
    cid = f"c{n[0]}"
    cells.append(f'<mxCell id="{cid}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" '
                 f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return cid


def link(src, dst, exit_y=0.5, entry_y=0.5):
    n[0] += 1
    style = f"{EDGE}exitX=1;exitY={exit_y};exitDx=0;exitDy=0;entryX=0;entryY={entry_y};entryDx=0;entryDy=0;"
    cells.append(f'<mxCell id="e{n[0]}" style="{style}" edge="1" parent="1" source="{src}" target="{dst}">'
                 f'<mxGeometry relative="1" as="geometry"/></mxCell>')


# column panels (x, width, inner box x, inner box width, header)
cols = [(10, 250, 22, 226, "INPUT DATA"),
        (290, 270, 302, 246, "RISK COMPONENTS (A–C)"),
        (590, 200, 602, 176, "INTEGRATION (D)"),
        (820, 290, 832, 266, "OUTPUTS (E–H)")]
for x, w, _, _, label in cols:
    add("", PANEL, x, 40, w, 590)
    add(label, HEAD, x, 48, w, 30)

inp = [
    ("<b>Climate and Earth<br>observation</b><br>ERA5-Land: T<sub>max</sub>, T<sub>min</sub>,<br>"
     "dew point, soil moisture<br>CHIRPS rainfall · MODIS<br>NEX-GDDP-CMIP6 (31 GCMs)"),
    ("<b>Crop statistics</b><br>District wheat area,<br>production and yield,<br>1982–2025 (PBS, CRS,<br>"
     "BoS, MNFSR) · SPAM2010"),
    ("<b>Socioeconomic data</b><br>MPI 2014–15<br>Agricultural Census 2024<br>Population Census 2023"
     "<br>RWI · VIIRS · travel time"),
]
comp = [
    ("<b>A. Hazard model</b><br>Detrended yield anomalies<br>Stage-wise heat degree-days,<br>VPD, rain, "
     "soil moisture<br>XGBoost / RF, blocked CV<br>SHAP/ALE thresholds;<br>heat loss → H<sub>d</sub>"),
    ("<b>B. Exposure</b><br>Wheat share of<br>cultivated land<br>Rural wheat-dependent<br>households"
     " → E<sub>d</sub>"),
    ("<b>C. Vulnerability index</b><br>Poverty, farm size,<br>rainfed share, tenancy,<br>literacy, access"
     "<br>PCA weights; checked<br>against RWI → V<sub>d</sub>"),
]
outp = [
    ("<b>E. Hotspots and inequality</b><br>Local Moran's I (FDR)<br>Concentration curves<br>and index"
     "<br>Losses borne by poorest 20%"),
    ("<b>F. Decision support</b><br>District SHAP drivers<br>→ policy lever: varieties,<br>irrigation, "
     "insurance,<br>social protection"),
    ("<b>G. Validation</b><br>2021–22 season withheld<br>Predicted vs observed anomalies<br>"
     "Risk terciles, cluster vs losses"),
    ("<b>H. Crop-model projections</b><br>WOFOST + APSIM-Wheat, 6 districts<br>CO<sub>2</sub>, phenology,<br>"
     "sowing date, cultivar → RQ4"),
]
I = [add(v, BOX, cols[0][2], y, cols[0][3], RH) for v, y in zip(inp, ROWS)]
C = [add(v, BOX, cols[1][2], y, cols[1][3], RH) for v, y in zip(comp, ROWS)]
span = ROWS[2] + RH - ROWS[0]
OH = (span - 3 * 10) / 4                        # four output boxes in the same vertical span
OROWS = [ROWS[0] + k * (OH + 10) for k in range(4)]
O = [add(v, BOX, cols[3][2], y, cols[3][3], OH) for v, y in zip(outp, OROWS)]
D = add("<b>D. Livelihood risk</b><br><br>R<sub>d</sub> = H<sub>d</sub> × E<sub>d</sub> × V<sub>d</sub>"
        "<br>(scaled to the present-<br>climate range)<br><br>Monte Carlo,<br>10,000 draws over<br>weights, "
        "hazard and<br>climate models<br><br>Median risk and<br>90% rank interval<br><br>Present climate;<br>"
        "GWL 1.5, 2 and 3 °C",
        KEY, cols[2][2], ROWS[0], cols[2][3], span)

# row centres as fractions of the tall risk box, so links run straight
frac = [round((y + RH / 2 - ROWS[0]) / span, 4) for y in ROWS]

link(I[0], C[0])
link(I[1], C[0], 0.25, 0.82)
link(I[1], C[1])
link(I[2], C[1], 0.25, 0.82)
link(I[2], C[2])
ofrac = [round((y + OH / 2 - ROWS[0]) / span, 4) for y in OROWS]
for k in range(3):
    link(C[k], D, 0.5, frac[k])
for k in range(4):
    link(D, O[k], ofrac[k], 0.5)

add("<b>Cross-cutting:</b> open data with logged provenance · blocked cross-validation (year, region, "
    "forward-chaining) · Monte Carlo uncertainty · 31-model CMIP6 ensemble · CO<sub>2</sub> fertilisation "
    "and adaptation: statistical model excludes them, crop model (H) simulates them at six districts", FOOT, 10, 650, 1100, 56)

xml = ('<mxfile host="drawio"><diagram id="p8" name="Methodology">'
       '<mxGraphModel dx="1120" dy="720" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
       'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1120" pageHeight="720" background="#FFFFFF" '
       'math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
src = OUT / "Fig2_methodology.drawio"
src.write_text(xml, encoding="utf-8")

for fmt, extra in (("pdf", ["--crop"]), ("png", ["--scale", "4"])):
    subprocess.run([str(DRAWIO), "--export", "--format", fmt, *extra, "--border", "8",
                    "--output", str(OUT / f"Fig2_methodology.{fmt}"), str(src)], check=True)
print("written", src)
