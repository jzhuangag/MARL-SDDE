## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: development experiment and mechanism audit
- Origin Date: 2026-09-07
- Verification Status: DEVELOPMENT RUN AND BYTE-EXACT REPRODUCTION COMPLETE
- Version Label: forecast_reversal_development_v1

# Markov forecast-reversal development audit

## Evidence boundary

This experiment was designed after inspecting earlier two-state development
smokes.  Its eight seeds (`97100--97107`) are therefore development data, not
confirmation or formal paper evidence.  It does not change any preserved
experiment, frozen gate, or historical negative result.

The purpose is a formulation-level kill test: determine whether finite-horizon
Markov prediction creates enough intrinsic value for a joint policy-cache-edge
and packet-weight Lyapunov controller to beat a strong online scheduler.  The
test is CPU-only and uses no standard-environment return.

## Analytic construction

The public context is a three-state directed cycle.  With cycle probability
`c`, its transition matrix is

\[
P_c=\begin{bmatrix}
1-c&c&0\\0&1-c&c\\c&0&1-c
\end{bmatrix}.
\]

The three candidate teammate-factor scores are

\[
S=\begin{bmatrix}
0.05&0.11&-0.16\\
-0.16&0.10&0.06\\
0.16&-0.23&0.07
\end{bmatrix}.
\]

Every row has zero mean under the uniform stationary law, so no fixed edge has
a favorable stationary mean built into the construction.  The packet horizon
is two.  Its exact factor value is

\[
V_a(s;c)=\frac12\{(2-c)S_{a,s}+cS_{a,s+1}\},
\tag{1}
\]

with state indices interpreted cyclically.  Pairwise value differences are
affine in `c`; this yields a closed-form rank-change boundary rather than a
post-hoc regime label.  At state zero, factors two and zero exchange order at

\[
c_\star=\frac{0.11}{0.225}=0.488\overline{8}.
\]

The state-one factor-zero/factor-one boundary is
`0.01/0.115=0.0869565`.  Consequently the complete future-optimal map becomes
`(0,1,2)` once `c>c_star`.  At `c=0.05`, the finite-horizon optimal mapping equals the
instantaneous state-myopic mapping `(2,0,2)`.  At `c=0.95`, the future-optimal
mapping is `(0,1,2)`, with margins `(0.10375,0.09925,0.15725)` and rank reversal
in two of three states.  This gives an analytic phase variable rather than an
empirically named regime.

Every event produces the common owner update.  A selected optional edge adds
one teammate factor to that trajectory; the resulting packet is applied after
a uniformly sampled bounded delay.  The plug-in controller uses only past
state transitions and factor scores, minimizes the joint Lyapunov edge--weight
index, and updates the communication virtual queue.  Scores for all local
factors are computed from the same centralized-training trajectory; this is a
full-information local-factor interface with `O(Delta)` computation, not free
extra environment data or extra cache messages.

## Matrix and comparators

- cycle probability: `0.05, 0.50, 0.80, 0.95`;
- maximum receipt delay: `1, 4, 8`;
- optional-message budget: `0.25, 0.50` per event;
- horizon: `4,096` launch events, two Markov transitions per event;
- seeds: eight development seeds;
- total: 24 scenarios, 10 policies, 1,920 endpoints.

The strong envelope contains the instantaneous state-myopic scheduler, each of
the three fixed edges, initial-state fixed edge, round-robin, random edge, and
no optional refresh.  It is selected by geometric average potential over all
eight development seeds within each scenario.  The exact-model joint
Lyapunov controller is an oracle diagnostic, not an implementable comparator.

## Primary result

All 1,920 endpoints are finite; environment-transition counts match within
every scenario/seed; and the post-run audit found zero communication-budget
violations.  The strong envelope selected `state_myopic` in all 24 scenarios.

| Quantity | Development result |
|---|---:|
| plug-in / strong geometric cumulative-risk ratio | `0.75172765` |
| exact dynamic oracle / strong ratio | `0.67150156` |
| all strictly improved cells | `21 / 24` |
| high-prediction-value (`c>=0.8`) plug-in / strong | `0.58680740` |
| high-prediction-value strictly improved cells | `12 / 12` |
| favorable return--message frontier-dominant points | `12 / 12` |

Thus the plug-in controller improves the full-matrix geometric cumulative risk
by `24.83%`; in the analytically favorable phase it improves by `41.32%`.
The three losses all occur at `c=0.05` and budget `0.50`, where future-optimal
and myopic rankings agree.  This is evidence for a phase-specific mechanism,
not universal dominance.

Across favorable cells, the recovered fraction of exact dynamic-oracle
headroom ranges from about `0.8185` to `0.9097`.  Actual message counts can be
above or below the strong baseline's realized count because both remain under
the same public upper budget and suppress zero-weight transmissions.  Paper
plots must therefore show both learning risk and actual policy bytes; they may
not describe this solely as an equal-message result.

The two registered budget points also permit a stricter Pareto check.  For each
of the 12 favorable `(c,delay,budget)` plug-in points, at least one
state-myopic point at the same `(c,delay)` uses no fewer actual messages and has
higher cumulative risk.  All 12 points pass.  This is the relevant
communication-efficiency evidence; the primary claim need not depend on
wall-clock speed.

## Provenance

- output: `experiments/policy_dependency_sync/results/forecast_reversal_dev_20260907`;
- configuration SHA-256:
  `f2bd7233cee207cfd9862346549092fa250e9a60be685e18fa5e5f0711b6180d`;
- endpoint SHA-256:
  `927a426ffff4ac26ca9fc4bb48ac8675a884fe5df82809f4b6fb71c99adff98a`.

Generated endpoints and summaries are ignored local artifacts.  They are not
committed as paper evidence.

The initial four-worker reproduction attempt stopped producing CPU work after
its workers exited and hung before creating an output directory.  It was
interrupted and is not evidence.  A fault-isolated rerun used four independent
two-seed single-worker chunks followed by a deterministic merge.  All chunks
exited zero.  The merged endpoint and summary files are byte-identical to the
primary:

- endpoints SHA-256:
  `927a426ffff4ac26ca9fc4bb48ac8675a884fe5df82809f4b6fb71c99adff98a`;
- summary SHA-256:
  `0e465e948a0a4b1eeaf7a31f69247c73dbcd7686d717a476f68250e66a7e89ff`.

## Decision

The mechanism passes the development kill test and may proceed to a separately
committed confirmation preregistration with untouched seeds.  The confirmation
must freeze the current controller, analytic matrix, strong state-myopic
baseline, cumulative-risk estimand, phase split, budget checks, source hashes,
and stop rule before execution.  It must preserve the expected low-`c` losses
rather than changing the matrix to manufacture universal no-harm.

This does not yet authorize Pursuit or any GPU run.  Standard MARL remains
conditional on the confirmation and on a controlled-Markov learned-critic
error interface for the plug-in bound.
