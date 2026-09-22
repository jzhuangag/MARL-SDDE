# T-022 EXP-018A failure diagnosis

EXP-018A remains a 5/7 preregistered failure. This diagnosis is post hoc and
does not change its decision.

The positive mechanism signal is real but scoped: the registered median/p90
variance-factor errors were 7.30%/32.38%, and the complete-stream pairwise
sharing rates matched rho. The failure is not evidence for a successful formal
experiment because two mandatory gates failed.

G6 conflated monotonicity with resolvability. At rho 0.5, the theoretical
`q=16` and `q=32` factors differ by only 2.94%; at rho 0.9 they differ by
0.35%. The observed strict-order fractions for this contrast were 55.47% and
59.38%, while the calibration error was actually smallest at rho 0.9. A future
design must not require strict empirical ordering for a contrast below its
predeclared practical-effect threshold.

G5 compared three independently selected but identically distributed q=1
streams. Its expectation is invariant, but a 64-seed comparison of three
sample variances produced median spread 0.327756. A prospective design may use
one shared q=1 common-random-number baseline across rho, because rho has no
pairwise meaning at q=1; this must be frozen before new outcomes.

The next scientifically valid action is an outcome-free redesign under a new
identifier. EXP-018A must not be rerun with altered thresholds or treated as
formal evidence. Any new design should retain all q/rho cells for calibration,
restrict directional tests to theoretically separated contrasts, make output
summaries path-independent, allocate new seeds, and undergo a new power audit.
