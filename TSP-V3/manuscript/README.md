# IEEE TSP V3 manuscript package

This directory contains the contribution-led IEEE Transactions on Signal Processing manuscript that integrates the fresh-seed MaMuJoCo confirmation while preserving the original `TSP/` manuscript unchanged.

Build `main.tex` with the local `build.ps1` script.
The confirmed MaMuJoCo evidence is generated from `../results/marl_mamujoco_calibrated_confirmation_20260930/summary.csv` and is not altered by the manuscript build.

The build regenerates every experimental figure used by the manuscript from
versioned summaries.  `figure_style.py` is the single source of truth for the
Times New Roman typography, line and confidence-band widths, gray-grid opacity,
baseline colors, panel dimensions, and five-tick charged-budget axis.
