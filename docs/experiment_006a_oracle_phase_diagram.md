# Experiment 006A: Oracle participation phase diagram

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: UNVERIFIED
- Version Label: preregistration_v1

## Research question

Does the finite-budget delayed stochastic approximation model contain a
non-degenerate and robust region in which the oracle-optimal number of
participating agents changes with Markov-noise dependence?

EXP-006A is an oracle diagnostic. It does not train or evaluate an adaptive
controller. Its purpose is to determine whether a subsequent low-complexity
participation controller has a valid target to learn.

## Oracle objective

For each cell, the oracle searches

\[
q\in\{1,2,4,8,16,32\}
\quad\text{and}\quad
\eta\in\operatorname{geomspace}(0.0025,0.08,17)
\]

using the same stable finite-horizon delayed-transition risk proxy as
EXP-005B/C. The initial delayed state history is constant at amplitude \(e\).
For each \(q\), the oracle first minimizes over \(\eta\); the runner-up is the
best action having a different \(q\) from \(q^\star\). The oracle records the
minimizing \((q^\star,\eta^\star)\), its risk, this different-\(q\) runner-up
risk, and their relative margin. Ties retain the existing registered ordering:
lower risk, then larger \(q\), then smaller \(\eta\).

## Frozen grid

- dependence paths:
  - global: \((\rho_g,\rho_c)=(s,0)\);
  - clustered: \((0,s)\);
  - balanced: \((s/2,s/2)\);
- dependence strength:
  \(s\in\{0,0.1,0.2,\ldots,0.8\}\);
- finite decision budget:
  \(B\in\{250,500,1000,2000,4000,8000\}\);
- current error amplitude:
  \(e\in\{0.01,0.03,0.1,0.3,1.0\}\);
- maximum delay:
  \(D\in\{0,4,16,32\}\);
- per-update overhead:
  \(c\in\{0,4,16\}\).

This gives \(3\times9\times6\times5\times4\times3=9720\) oracle cells.
All other loadings, clusters, autoregressive coefficients, curvature, delay
profile, stability tolerance, and candidate grids remain unchanged from
EXP-005C.

## Track-level definitions

A track fixes dependence path, budget, error amplitude, maximum delay, and
overhead, then varies \(s\) from 0 to 0.8.

- **responsive track**: \(q^\star(s)\) is nonincreasing,
  \(q^\star(0)\ge16\), and \(q^\star(0.8)\le q^\star(0)/4\);
- **well-separated responsive track**: a responsive track whose endpoint at
  \(s=0.8\) has runner-up relative risk margin at least 5%;
- **delay-sensitive group**: after fixing path, strength, budget, error, and
  overhead, at least two of the four delay values choose different
  \(q^\star\).

The relative margin is

\[
\frac{R_{\mathrm{second}}-R_{\mathrm{best}}}
{\max(R_{\mathrm{best}},10^{-15})}.
\]

## Actionable rectangle

For overhead \(c=4\), an actionable rectangle is any adjacent two-by-two
rectangle in the registered budget-error grid, for either the global or
clustered path, such that for both \(D=4\) and \(D=16\):

1. every corner has \(q^\star(0)\ge16\);
2. every corner has \(q^\star(0.8)\le8\);
3. every high-dependence corner has at least 2% runner-up margin.

The rectangle is searched mechanically after evaluating the full registered
surface; no interpolation or post-hoc grid refinement is allowed.

## Go/no-go gates

All gates must pass before agent-number adaptation is retained as a candidate
main contribution:

1. **non-degenerate surface**: at least three distinct \(q^\star\) values each
   occupy at least 1% of all cells, and no single \(q^\star\) occupies more
   than 85%;
2. **correlation responsiveness**: at least 15% of all tracks are responsive,
   and at least two dependence paths individually have at least 10%
   responsive tracks;
3. **decision margin**: at least 50% of responsive tracks are
   well-separated;
4. **delay relevance**: at least 10% of registered delay groups are
   delay-sensitive;
5. **actionable region**: at least one registered actionable rectangle exists;
6. **numerical validity**: every cell has a stable feasible action, finite
   risks, nonnegative margins, and exact row count 9720.

Failure of any gate means the current finite-budget proxy is not yet a sound
foundation for an ICML-level adaptive-participation contribution. The result
will not be repaired by changing thresholds, adding grid points, or excluding
cells after execution.

## Execution

- working directory: `experiments/dependence_delay_linear`;
- smoke:
  `python run_oracle_phase.py --output-dir results/smoke/oracle_phase --smoke`;
- primary:
  `python run_oracle_phase.py --output-dir results/oracle_phase`;
- local Windows CPU; no GPU;
- hard timeout: 10 minutes.

## Expected outputs

- `oracle_surface.csv`;
- `track_summary.csv`;
- `delay_summary.csv`;
- `actionable_rectangles.csv`;
- `summary.json`;
- phase-diagram and participation-frequency figures.

## Execution outcome

The primary scan completed all 9,720 cells in 87.6 seconds. Five of six gates
passed. The only failure was delay relevance: 5.31% of delay groups changed
\(q^\star\), below the registered 10% threshold. The overall registered verdict
is therefore **FAIL**.

Correlation responsiveness was nevertheless strong and non-degenerate:
37.22% of tracks were responsive, 79.60% of those had at least 5% decision
margin, all three dependence paths exceeded 33% responsive coverage, and ten
actionable rectangles were found. A deterministic rerun completed in 86.2
seconds and all seven artifacts matched byte-for-byte.

The validated interpretation is narrower than the failed combined gate:
correlation- and state-adaptive participation has a robust oracle target, while
delay primarily shifts the transition surface rather than frequently changing
the pointwise optimal agent count. See `validation_exp006a.md`.
