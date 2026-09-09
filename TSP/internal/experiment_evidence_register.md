# Experiment evidence register

This register records the positive evidence admitted to the paper.

| Experiment | Frozen evidence | Primary result | Reproduction |
|---|---|---|---|
| EXP-007A | 32 paired seeds, 134,784 rows | Effective participation at `q=32` is 30.996 for independent paths and 1.111 at correlation 0.9; resource-optimal participation moves from 16 to 1 | Byte-exact |
| EXP-010B | 32 paired seeds, 1,152 policy runs, 12 scenarios | All 3 numerical and 6 scientific gates pass; all 12 empirical upper means lie below the finite-time bound | Eight artifacts byte-exact |
| EXP-016B formal | 192 paired seeds, 2,752,512 rows | Layer-A risk improvement 54.4267%; affine-TD transfer improvement 10.8523%; 77 of 96 scenario families satisfy the directional practical-effect criterion | Four core artifacts byte-exact |

The manuscript uses the experiments as separate proof instruments:

- EXP-007A tests the statistical mechanism.
- EXP-010B tests the Lyapunov certificate and joint action response.
- EXP-016B tests the value of conditioning information acquisition on downstream learning benefit.

No exploratory outcome is promoted to confirmatory evidence in the TSP draft.
