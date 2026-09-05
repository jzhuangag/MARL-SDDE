# EXP-017B static design: probe-separated participation

## Status

This document is an outcome-free design record, **not** an EXP-017B
preregistration and not authorization to run a pilot. T-019 assigned no new
seed and submitted no job. A later independent commit must freeze the full
runner, analyzer, seeds, thresholds, and population before any GPU outcome.

The design keeps the EXP-017A scientific boundary: fixed-policy nonlinear
neural TD on standard Markov tasks, known mixing or an independently certified
separated certificate, no unrestricted unknown-mixing claim, no global
occupation-optimality claim, and no general nonlinear MARL claim.

## Required controller architecture

### 1. Probe and learning participation are separate actions

At public, outcome-independent probe times, execute a correlation-only probe
with `q_probe=4` and `b_probe=1`. The proposed static schedule probes the first
eight controller blocks and then every 32nd block. Probe observations update
only the dependence certificate; their gradients are discarded and cannot
update the value network.

Every probe is charged exactly like a learning action:

- message charge: `server_overhead + q_probe * parameter_count * 4`;
- environment charge: `b_probe`;
- separately reported agent transitions: `q_probe * b_probe`.

Because q_probe is at least two, every executed probe supplies
`q_probe(q_probe-1)/2 = 6` pairwise trials. Probe timing does not depend on the
learning q, loss, held-out error, true rho, or outcomes. Hence pairwise
evidence grows even if learning participation is q=1, eliminating the
EXP-017A absorbing loop.

### 2. Public strong fallback, never silent q=1

Until the preregistered dependence-evidence condition is met, learning uses a
public nontrivial fixed-q table derived descriptively from the EXP-017A strong
fixed arms:

| Task | Message-binding | Environment-binding |
|---|---:|---:|
| Acrobot | q=4 | q=16 |
| CartPole | q=4 | q=32 |

These choices are pilot-derived design inputs, not formal winners. They must
be frozen before new seeds are assigned. Once evidence is sufficient, the
learning action may choose among q in `{1,4,16,32}`, but the independent probe
schedule continues; selecting q=1 can no longer halt identification.

### 3. Replace the EXP-017A surrogate

True-rho oracle failure shows that correlation information alone cannot repair
the old analytic score. EXP-017B must not reuse it as the decision objective.
The preregistration must specify a learning-value statistic based only on
online training quantities and public resource/certificate inputs, with an
information-only comparator retaining identical probe evidence. Held-out
terminal error, Bellman evaluation data, source assignments, and another
policy's outcomes remain forbidden controller inputs.

### 4. Vectorized and cached scoring

The 12 `(q,b)` candidate features, message costs, inverse-q terms, mixing
powers, and the three delay-trace summaries are computed once per static
configuration and cached. Per-block evaluation is a vector operation over the
12-row table. No Hessian inverse, covariance inverse, preconditioner, or
matrix-valued online controller state is permitted.

## Mandatory future preregistration gates

At minimum, a later outcome-free EXP-017B registry must freeze:

1. new seeds disjoint from `20550101` and `20550102`;
2. exact probe schedule, q_probe, b_probe, and full resource charging;
3. proof and CPU tests showing probe trials increase for every learning-q
   trajectory, including persistent q=1;
4. the public fallback table above and a static assertion that no-evidence
   fallback is never q=1;
5. a complete replacement learning-value surrogate and taint audit;
6. vectorized/cached candidate and delay features;
7. the same standard tasks, marginal-preserving dependence construction,
   mixing claim boundary, correlations, delay traces, and dual budgets unless
   a separately justified design amendment is committed before seeds;
8. a paired no-harm gate against the task × budget public strong fixed-q
   baseline: proposed controller geometric terminal-error ratio at most 1.05
   and CVaR90 ratio at most 1.10, with all registered resource-validity and
   metrics-completeness gates also passing;
9. information-only, public-fallback, oracle, fixed-q, single-agent,
   always-all, correlation-blind, mixing-blind, no-delay, and probe-cost
   ablations;
10. runner/analyzer hashes and a rule that any mandatory failure stops formal
    without seed, threshold, or population changes.

The numeric no-harm margins above are design proposals, not post-pilot
EXP-017A reinterpretations. They become operative only if prospectively
frozen in an independent EXP-017B preregistration.

## Current authorization decision

T-019's static design tests pass and the descriptive fixed-q diagram varies
systematically with rho, delay, and especially budget. This permits preparing
an independent GPU-pilot preregistration. It does **not** permit submission of
that pilot from the current commit. No formal seed list exists, no GPU output
is authorized, and all EXP-017A negative artifacts and gates remain immutable.
