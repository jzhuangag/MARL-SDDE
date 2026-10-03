# EXP-011B validation: dual-anytime correlation/mixing controller

## Decision

**PASS, with one scientific-gate failure.**  Both validity gates pass and four
of five scientific gates pass, meeting the preregistered decision rule.

The result establishes safe predictable adaptation when pair sharing is
observable.  It does not support a claim that correlation adaptation improves
every delayed cell or that the controller is uniformly near an informed
reference.

## Formal execution

- Scenarios: 18
- Fresh formal seeds per scenario: 32
- Dual-controller runs: 576
- Correlation-blind runs: 576
- Predictable action rows: 11,520
- Base seed: 20261231
- Resource per run: 20,000
- Initial charged pair pilot: 1,280
- GPU: not used

The earlier four-seed implementation pilot and the stopped pre-evidence
Clopper--Pearson process are excluded.  The formal run uses the corrected
beta-binomial mixture e-process throughout.

A clean same-seed execution reproduced both CSV files and the figure
byte-for-byte.  Parsed `summary.json` objects are identical; their hashes
differ only because the canonical metadata-label patch introduced one final
newline.  The machine-readable comparison is retained with the canonical
artifacts and does not classify this formatting-only difference as a numerical
reproduction failure.

## Validity gates

| Gate | Threshold | Result |
|---|---:|---:|
| Joint time-uniform coverage | \(\ge97.5\%\) | \(100\%\) (576/576) |
| Conditional exact safety | every updating action has radius \(<1\) | pass; maximum \(0.992693\) |

The confidence sequence is valid at arbitrary adaptive sample sizes; this is
stronger than checking fixed-sample confidence intervals only at the observed
decision times.

## Scientific gates

| Gate | Threshold | Result |
|---|---:|---:|
| Participation response | median \(q_{.9}<q_0\) | pass: \(1<9.5\) |
| High-correlation advantage | improve at least 5/6 cells | **fail: 4/6** |
| Independent-data efficiency | worst dual/blind ratio \(\le1.5\) | pass: \(1.2266\) |
| High-persistence mixing refinement | at least 5/6 cells | pass: 6/6 |
| Correlation refinement | at least 9/18 cells | pass: 15/18 |

At \(D=0\), the dual controller beats the correlation-blind policy in all
three \(\rho=.9\) persistence cells.  At \(D=2\), delay already pushes both
methods toward \(q=1\); the two low/medium-persistence errors are almost tied,
and the strict advantage gate fails in those cells.  This is a mechanistic
interaction, not a safety failure: when delay alone makes single-agent
participation optimal, estimating cross-agent sharing has little remaining
control value.

## Controller limitation

The median error ratio to the true-parameter, no-pilot informed controller is
not uniformly small; its maximum is 38.27 in a high-persistence delayed cell.
This reference is not a dynamic-programming oracle, and the ratio is
descriptive.  The result confirms the earlier EXP-009D conclusion that
finite-budget estimation near \(p=1\) carries a large inverse-mixing-gap
penalty.  No uniform near-oracle claim is made.

## Mainline decision

Retain:

1. the correlation-limited minimax lower bound;
2. the affine finite-gap delayed Markov upper bound;
3. predictable dual-anytime safety;
4. low-complexity joint selection of \((q,b,\eta)\); and
5. the empirical result that correlation adaptation matters most before delay
   has already collapsed participation to one.

The remaining CPU theory extension is a latent-correlation confidence
sequence, because observable pair-sharing metadata will not exist in every
MARL environment.  The next breadth milestone is a nonlinear multi-agent
Markov benchmark, which is the first stage likely to benefit materially from a
GPU.
