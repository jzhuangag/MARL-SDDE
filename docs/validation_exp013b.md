# EXP-013B validation: realizable nonlinear Markov TD

## Material Passport

- Artifact: confirmatory validation record
- Registration: `3bb903f386b24e69b4e2b6ceb20aa046bdacee88`
- Outcome: failed, three of five frozen gates passed
- Evidence scope: controlled realizable neural TD; not a standard MARL
  benchmark

## Execution audit

The frozen protocol specifies 32 seeds, four correlations, two delays, and
four participation levels, for \(32\times4\times2\times4=1{,}024\) runs.

The first serial execution was launched with a 30-minute command-wrapper
timeout.  At approximately 28 minutes it had not written any artifact and was
stopped before the wrapper could terminate it.  This exposed an execution
envelope error, not a model or statistical failure.

The unchanged registered runner was then scheduled as four disjoint,
eight-seed CPU chunks:

- 20270701--20270708;
- 20270709--20270716;
- 20270717--20270724; and
- 20270725--20270732.

Every chunk produced 256 rows, had an empty error log, contained no duplicate
configuration key, and reported all 256 runs finite.  The merge program
rejects a non-contiguous seed set, duplicate key, or row count other than
1,024 before calling the registered `analyze` function.  Scheduling therefore
changed neither the data-generating process nor the frozen analysis.

Environment:

- Python 3.8.12;
- PyTorch 1.11.0+cpu;
- NumPy 1.20.3;
- pandas 1.4.1; and
- Windows 10.0.19045.

## Frozen-gate results

| Gate | Frozen criterion | Result | Pass |
|---|---|---:|:---:|
| Numerical validity | all 1,024 runs finite | 1,024/1,024 | yes |
| Independent-agent benefit | 99% UCL of \(q=32/q=1<.70\) | ratio .0622; UCL .1119 | yes |
| High-correlation small-subset benefit | 99% UCL of \(q=4/q=32<.85\) | ratio .8319; UCL 1.2935 | no |
| Oracle participation shift | median \(q_0\ge16,\ q_{.9}\le4\) | \(32\rightarrow4\) | yes |
| Both delays preserve magnitude | low-\(\rho\) ratios \(<.80\), high-\(\rho\) ratios \(<.90\) | low: .0674/.0573; high: .9087/.7616 | no |

The preregistered overall conclusion is therefore **failure (3/5 gates)**.
The \(D=0,\rho=.9\) point misses its .90 threshold by .0087, but the failure
is not merely threshold rounding: the cluster-bootstrap upper limit 1.2935
does not exclude no benefit.

Median resource-oracle participation is 32, 16, 16, and 4 at correlations
0, .25, .5, and .9.  This is a clear population-level shift, but the high
correlation choice is heterogeneous across seeds.

## Descriptive failure audit

This section was not preregistered and is labeled descriptive.

At \(\rho=.9\), \(q=4\) beats \(q=32\) in 16/32 seeds for \(D=0\) and 19/32
for \(D=8\).  The per-seed ratios are heavy-tailed: their maxima are 10.56 and
6.37, respectively.  Fixed \(q=1\), 4, and 16 all fail to obtain a reliable
99% advantage over \(q=32\).

The more basic loss-of-speedup interaction is strong.  The geometric
\(q=32/q=1\) MSE ratio changes from .0622 at \(\rho=0\) to 1.1487 at
\(\rho=.9\).  Their ratio is 18.48; a descriptive seed-cluster bootstrap gives
a 99% lower limit of 8.51.  This supports the nonlinear *Beyond Linear
Speedup* mechanism, but it does not rescue the failed fixed-participation
claim.

The implementation audit found no missing row, duplicate key, non-finite
loss, inconsistent budget, or seed overlap.  The failure is consistent with
nonlinear optimization-state heterogeneity: correlation determines the
available variance reduction, while the current parameter state determines
whether fewer updates or lower gradient variance is more valuable.

## Reproduction

Re-running the deterministic merge, registered bootstrap, and plotting code
from the four raw chunks produced byte-identical artifacts:

| Artifact | SHA-256 |
|---|---|
| `metrics.csv` | `F6C88F04C302DF06D8249ED22E746C388F9BEBED087120CED10BE3F0ADD5D30C` |
| `oracle_choices.csv` | `252416E473B881CBEFF124E169CD88FA199B657E2CA2C96E976FFB7594EEEFAC` |
| `summary.json` | `404733F5F95E5445C38D115F535A0BC529D55EA7DC5D639FFAD407F61E9A8C81` |
| `realizable_td_confirmation.png` | `CCC24BDBA789BABCD2B4A0D10A341F73C1A9D0A962BC54D851E0A3815566E71C` |

## Research decision

Retain the theorem-backed nonlinear claim that cross-agent correlation limits
parallel variance reduction by
\(\rho+(1-\rho)/q\).  Retain the controlled evidence that the useful range of
participation collapses as correlation increases.

Do not claim that a correlation-only fixed \(q\) reliably improves nonlinear
TD.  The next algorithmic gate must be state- and risk-aware: it should use
observable gradient-noise and progress statistics to decide whether the
current regime values extra updates or extra averaging.  That controller
should be tested on standard multi-agent or distributed Markov-learning
benchmarks with GPU/HPC4 rather than by further tuning this synthetic teacher.

