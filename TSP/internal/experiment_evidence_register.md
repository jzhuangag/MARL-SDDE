# Experiment evidence register

This register records the positive evidence admitted to the paper.

| Experiment | Frozen evidence | Primary result | Reproduction |
|---|---|---|---|
| EXP-007A | 32 paired seeds, 134,784 rows | Effective participation at `q=32` is 30.996 for independent paths and 1.111 at correlation 0.9; resource-optimal participation moves from 16 to 1 | Byte-exact |
| EXP-010B | 32 paired seeds, 1,152 policy runs, 12 scenarios | All 3 numerical and 6 scientific gates pass; all 12 empirical upper means lie below the finite-time bound | Eight artifacts byte-exact |
| EXP-016B formal | 192 paired seeds, 2,752,512 rows | Layer-A risk improvement 54.4267%; affine-TD transfer improvement 10.8523%; 77 of 96 scenario families satisfy the directional practical-effect criterion | Four core artifacts byte-exact |
| TSP-CURVE-001 | 16 development and 64 disjoint confirmation seeds, 157,440 confirmation rows | Parameter-error AUC reduction 9.9087%; discounted-return-estimation AUC reduction 11.3211%; 9/12 cells improve and 3/12 tie | Complete raw hashes and deterministic aggregate retained |
| TSP-MARL-CONF-001 | 8 confirmation training seeds, 8 disjoint probe seeds, 48 MAPPO runs | Dependence-matched action selected in 16/16 controller runs; shared-stream and equal-mixture team-return AUC gains over fixed `q=8` are 4.9957% and 2.3195% | Gate JSON, curve CSV, and PDF byte-identical |

The manuscript uses the experiments as separate proof instruments:

- EXP-007A tests the statistical mechanism.
- EXP-010B tests the Lyapunov certificate and joint action response.
- EXP-016B tests the value of conditioning information acquisition on downstream learning benefit.
- TSP-CURVE-001 tests resource-indexed convergence and the transfer from parameter error to fixed-policy return-estimation error against a development-selected strong fixed-participation baseline.
- TSP-MARL-CONF-001 tests the observable dependence-selection mechanism on cooperative policy learning under equal message and environment budgets.

No exploratory outcome is promoted to confirmatory evidence in the TSP draft.
The TSP-CURVE-001 summary was recovered from complete immutable trajectory metrics after a JSON serialization failure; no scientific trajectory was rerun.
