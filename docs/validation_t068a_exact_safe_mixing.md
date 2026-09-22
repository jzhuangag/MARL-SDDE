# T-068A exact shadow-anchored mixing validation

## Decision

T-068A is a reproducible partial-positive mandatory-gate failure.  P1--P4,
P6--P9, and the external clean-reproduction gate P12 pass.  P5, P10, and P11
fail.  No sampled T-068 pilot, formal experiment, standard benchmark, GPU job,
or HPC4 job is authorized.

The positive mechanism ceiling is material: after charging twelve of 240 actor
transitions to sensing, the exact recipient-wise safe oracle improves geometric
terminal risk by 5.3224% over the cellwise best common fixed mixing strength and
strictly improves 79.9383% of cells.  Its geometric ratio to the no-probe local
learner is 0.647924, and every cell is within the preregistered 1.05 no-harm
tolerance.

Those values do not establish the preregistered phase claim.  The best common
two-phase schedule improves the cellwise best common fixed strength by only
0.9278%, below the frozen 5% threshold.  Among active cells, only 36.5741% have
larger early than late mean mixing, below 40%.  All six alpha values occur, but
the exact oracle never resets to the charged shadow, so the frozen requirement
that both fallback states occur also fails.

## Frozen provenance

- Parent research commit: `be99b685a2554b7cdabaf76ba6ec0cfe3c7cfd47`.
- Independent preregistration commit: `1cd1761`.
- Configuration SHA-256:
  `C33914498EC711DB657D0668C2EE321C633C5108F0115461D8B9933772A88A28`.
- Runner SHA-256:
  `32C10FAA7173E8E4745CE219A4F2CDDCF3458E910F93D92E8E5978039AD6E330`.
- Exact-moment core SHA-256:
  `DE2C4C83DD7DE5A8C6B36C30669E07A5AA2534E7AF885290D5E8A9C04D63A41F`.
- Workload: 648 cells, 43 policies per cell, 27,864 policy rows, and
  668,736 block-level moment propagations.
- Hardware: local CPU only.

## Gate ledger

| Gate | Frozen requirement | Observed | Status |
|---|---|---:|---|
| P1 | 648 finite cells and 27,864 finite policy rows | exact | pass |
| P2 | zero mixing equals same-data shadow | exact | pass |
| P3 | safe oracle never exceeds charged shadow at a checkpoint | 100% | pass |
| P4 | exact message/environment budgets | 100% | pass |
| P5 | two-phase aggregate improvement at least 5% | 0.9278% | **fail** |
| P6 | two-phase strict improvement in at least 50% cells | 62.0370% | pass |
| P7 | fully charged safe-oracle improvement at least 3% | 5.3224% | pass |
| P8 | safe oracle strict improvement in at least 40% cells | 79.9383% | pass |
| P9 | aggregate no worse than no-probe local and 95% within 1.05 | 0.647924; 100% | pass |
| P10 | early mixing exceeds late mixing in at least 40% active cells | 36.5741% | **fail** |
| P11 | at least four alpha values and both shadow states | six alpha; no reset | **fail** |
| P12 | clean rerun byte-identical | all three artifacts exact | pass |

## Structural interpretation

The result rejects the simple universal collaborate-then-personalize story.
In homogeneous cells, selected mixing generally increases as the common error
contracts.  In higher-heterogeneity cells, recipients with different targets
receive very different mixing strengths.  The observed 5.3224% value may
therefore arise from recipient-specific personalization rather than temporal
adaptation.

That distinction is claim-critical because the strong fixed comparator in
T-068A uses one common scalar alpha for all four recipients.  The next allowed
work is an independently preregistered exact CPU audit against a fixed
recipient-specific alpha vector.  It must precede any sampled controller.  A
positive T-068A oracle ceiling cannot bypass this comparator.

The absence of shadow reset is not retroactively treated as harmful: alpha zero
can retain a previously improved collaborative model, so a reset need not be
optimal.  Nevertheless P11 remains failed exactly as registered.

## Reproduction

The clean output directory is ignored and separate from the primary result.
The following SHA-256 values are byte-identical across both runs:

| Artifact | SHA-256 |
|---|---|
| `policies.csv` | `4D5BA23006F24607565579F5741A1FA6CA88CACAE3BB7EE8C4C2B0D66F5565A3` |
| `cells.csv` | `F79F9D34BEA89EEDD243E3DB555E51133CD29227DE2A9CC05D551D0A0DA99BD8` |
| `summary.json` | `51C314B24A71D5765A0E22566BDE5D087E5EE397A60D1C7B4F323B6E68D57699` |

The full repository regression completed with `585 passed, 7 skipped in
788.62 s`.  Targeted T-068 tests completed with `12 passed`.
