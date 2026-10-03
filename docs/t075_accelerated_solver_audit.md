# T-075 safeguarded warm-start solver audit

T-075 tested a warm-started accelerated projected-gradient solver without
changing T-074's statistical controller. Sixty-four deterministic random QPs
and two trajectory fixtures pass objective, feasibility, risk-path, rollback,
and accounting tests. On one trajectory fixture, total iterations fall from
155 to 137.

The runtime gate fails. The four-old-seed calibration exceeded 120 seconds
before completion, versus 53.15 seconds for the frozen T-074 implementation.
Repeated objective safeguards, residual projections, and FISTA state updates
outweigh the iteration reduction in Python. The process was terminated without
creating a result file or grid summary. No tolerance was relaxed and no retry
was performed.

T-075 is therefore not admissible for a pilot. The next action keeps T-074's
scientific controller and solver unchanged, but evaluates endpoints in a
deterministically ordered local process pool under a new identifier. This
separates algorithmic per-decision cost from serial grid orchestration cost.
