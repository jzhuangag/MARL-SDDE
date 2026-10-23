# T-041A exact phase-map validation

## Decision

**PASS: P1--P10 all pass.** T-041A validates the prospective exact
finite-resource phase map and authorizes only a separately committed
fresh-seed CPU tabular/linear-TD preregistration. It does not authorize a
nonlinear GPU pilot or formal benchmark.

Preregistration commit: 3bc29c5

Frozen configuration SHA-256:
9f5149f4dacbe0603968bcc0e303139df4d3d98fdc6ac055008449a36ac3bdfd

Frozen runner SHA-256:
a5cd92a1388a79b6e2316f5e3b209737d7e9293fd95ce0ec48334842989aea7d

## Registered population

- 270 scenarios and 1,620 exact action rows;
- three drift geometries, three delays, three temporal coefficients, two step
  sizes, \(q\in\{1,4,16\}\), and \(b\in\{1,4\}\);
- no seeds, sampling uncertainty, confidence intervals, GPU, or HPC4;
- initial run: 387.2 seconds on local CPU;
- clean reproduction: 361.7 seconds on local CPU.

## Gate ledger

| Gate | Result | Registered value |
|---|---:|---:|
| P1 finite/stable/budget-valid | PASS | all 1,620 rows |
| P2 speedup direction | PASS | 1.000, threshold 0.95 |
| P3 saturation equality | PASS | maximum relative error 0 |
| P4 reversal direction | PASS | 1.000, threshold 0.90 |
| P5 nontrivial best-q support | PASS | \(\{1,16\}\) |
| P6 binding-ray construction | PASS | environment for speedup/saturation; message for reversal |
| P7 independent scalar identity | PASS | maximum relative error \(3.37\times10^{-15}\) |
| P8 positive and no-value regions | PASS | 108 positive cells; 54 no-value cells |
| P9 byte-identical clean rerun | PASS | rows and summary identical |
| P10 provenance hashes | PASS | configuration, runner, rows, summary recorded |

The 108 speedup-family comparisons all improve by at least 5%. The 54
saturation-family comparisons have exactly equal T-037 risk at the primary
stride. Every one of the 108 reversal-family comparisons makes \(q=16\) at
least 5% worse than \(q=1\). The best-action support contains both endpoints,
so the grid does not collapse to a universal all-agent or one-agent choice.

## Reproduction

Primary artifacts:

- experiments/dependence_delay_linear/results/t041a_exact_phase/rows.csv
- experiments/dependence_delay_linear/results/t041a_exact_phase/summary.json

Clean-rerun artifacts:

- experiments/dependence_delay_linear/results/t041a_exact_phase_reproduction/rows.csv
- experiments/dependence_delay_linear/results/t041a_exact_phase_reproduction/summary.json

Hashes:

| Artifact | SHA-256 |
|---|---|
| rows.csv | a4e26c8774bd3723ea16939778d61dc6657cc581f398e49520b919d32518bbeb |
| summary.json | d06cb5553eb681cc431279d8aa39f7ae5173336ff185d2ab10d0a9653ca36bf3 |

Both corresponding files are textually byte-identical across the two
directories.

## Scientific interpretation

This is stronger than a proxy scan: every risk is the T-037 finite-horizon
identity, delay enters the companion dynamics and environment charge, and the
T-035 scalar formula supplies an independent numerical identity. The result
shows that the theorem can distinguish speedup, saturation, and reversal on a
prospectively frozen vector grid.

It is also deliberately limited. The three families were constructed from
the theorem's resource regimes; T-041A does not show that an unchanged-law
standard RL task naturally occupies all three. It does not close the
multiplicative Markov-TD Poisson remainder, learn the regime online, or
validate a nonlinear controller. Those claims require the next prospective
CPU transfer and, only after its gate, a possible GPU experiment.

## Integrity audit

No registered scenario, threshold, comparison, matrix, budget, or runner line
was changed after the preregistration commit. The first run and clean rerun
used the same frozen hashes. No old experiment outcome entered a registered
risk calculation, and no failed cell was excluded.
