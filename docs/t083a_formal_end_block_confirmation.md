# T-083A formal causal end-block confirmation preregistration

T-083A freezes 128 formal seeds with zero overlap with T-071A design seeds and
T-082A pilot seeds.  The controller source hash, 96-cell primary population,
336 controls, continuous static comparator, and all scientific thresholds are
unchanged after the independent pilot.

The formal run contains 55,296 paired endpoints and remains local CPU only.
`endpoints.csv`, `cells.csv`, and `summary.json` are the byte-reproduced
scientific artifacts; runtime remains isolated in `execution.json`.  Any F1--F14
failure stops formal confirmation and all nonlinear/GPU escalation without
changing seeds, thresholds, or subgroups.

Even if all gates pass, T-083A confirms only the registered affine Markov
system and separated signal/mixing class.  A later safety-shield design and
standard RL benchmark require separate outcome-free preregistration.
