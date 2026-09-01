# Perishable-update phase-conditioned CPU pilot preregistration

## Material Passport

- Artifact ID: `PUB-PHASE-PILOT-20260901`
- Artifact type: experiment preregistration
- Workflow: ARS experiment-agent, plan-to-run handoff
- Stage: Stage 2 mechanism confirmation
- Parent commit: `ba1aeb9f8ec820743366e586df81b3d886d07650`
- Population outcomes observed while preparing this artifact: none
- Next gate: mandatory static audit and independent preregistration commit

## Purpose

This pilot tests one consequence of the perishable-update theory rather than another unrelated controller variant.
In asynchronous multi-agent policy optimization, an accepted proposal changes the validity of proposals that are still in flight.
The theory predicts that this cross-agent expiry cost is negligible in a low-load phase and creates strict value for dynamic scheduling in a separated high-load phase.
The executable question is whether the same closed-form Lyapunov backpressure rule realizes both sides of that phase statement against a deliberately strong static comparator envelope.

This is a mechanism-confirmation study on an exactly analyzable cooperative quadratic potential game with Markovian completion times.
It is not a standard reinforcement-learning benchmark and cannot by itself support a general nonlinear MARL claim.

## Frozen model and population

Each cell has `n` distinct scalar policy parameters and potential

\[
\Phi(\theta)=-\frac12\theta^\top A\theta,
\]

where `A` has unit diagonal and common off-diagonal coupling `c`.
Every agent repeatedly computes a unilateral exact-gradient proposal.
Its completion time follows a seeded two-state Markov latency process.
An accepted coordinate displacement changes every pending proposal's gradient by the known cross-sensitivity `c`, so freshness debt is exact rather than estimated.

The outcome-independent phase rays are:

| Phase | Coupling `c` | Slow latency |
|---|---:|---:|
| low | 0.05 | 2 |
| transition | 0.45 | 5 |
| high | 0.85 | 10 |

They are crossed with `n in {3,5}`, persistent or bursty latency, and sparse, alternating, or dense initialization.
This gives 36 cells, 12 per phase.
The wall-clock horizon is 64 and the Lyapunov potential weight is `V=8`.

The 48 common-random-number pilot seeds are exactly `910001` through `910048` inclusive.
No formal seed has been allocated.

## Frozen controller and comparators

The tested controller is the closed-form perishable-update backpressure (PUB) rule derived in `perishable_update_phase_theory.md`.
At each completion time, it recomputes the exact pending-proposal debt, evaluates the one-dimensional Lyapunov drift bound, jointly chooses admission and step size, and orders simultaneous completions by the smallest certified bound.
It scans no learned hyperparameter and uses `O(n)` scalar arithmetic per ready proposal.

The strong comparator family contains 42 policies:

- seven fixed-step asynchronous accept-all policies;
- seven fresh serial policies;
- seven synchronous barrier policies;
- nine age-decay policies;
- twelve relative-debt threshold policies.

For each cell, the analyzer selects the comparator with the smallest mean normalized regret over all 48 pilot seeds.
This outcome-aware cellwise envelope is intentionally favorable to the comparator and is not presented as a deployable algorithm.

## Outcomes and aggregation

The primary endpoint is normalized wall-clock potential regret.
The secondary endpoint is final full-gradient norm.
The analyzer first averages each policy within a cell, selects the strong comparator described above, and then reports phase-level geometric PUB/comparator ratios and unweighted cell fractions.
No p-value or null-hypothesis significance test is used.

The complete run has 36 cells, 48 seeds and 43 policies, for 74,304 deterministic output rows.
Runtime measurements are deliberately excluded from the scientific JSON because they are machine-load dependent and would invalidate exact byte reproduction.
Arithmetic operation counts provide the preregistered complexity diagnostic.

## Mandatory gates

All gates are joint and immutable after the preregistration commit.

1. All 74,304 rows exist and all registered numeric endpoints are finite.
2. High-load geometric normalized-regret ratio is at most 0.85.
3. At least 60% of high-load cells improve normalized regret by at least 5%.
4. Transition-phase geometric normalized-regret ratio is at most 0.95.
5. Low-load geometric normalized-regret ratio is at most 1.03.
6. Median regret gain is strictly ordered high greater than transition greater than low.
7. High-load median PUB acceptance rate lies in `[0.05,0.95]`, excluding accept-all and reject-all collapse.
8. High-load geometric final-gradient ratio is at most 1.05.
9. The five-agent to three-agent arithmetic-operations-per-event ratio is at most `5/3+0.05`.
10. PUB produces zero realized potential-decreasing accepted steps in the exact quadratic model.
11. A clean rerun is byte-identical to the original trajectories file.
12. Seeds and source provenance match this preregistration.

Failure of any gate stops formal preregistration and nonlinear/GPU escalation for this mechanism.
Passing every gate permits only a new, separately committed formal preregistration with disjoint seeds.
This pilot is never formal paper evidence.

## Taint boundary and stopping rule

Prior role-switch, participation, T-083A, EXP-017A and T-020 outcomes are excluded from the population and analysis.
Their negative records remain unchanged.
The phase rays are fixed from the analytic separation theorem, not selected from old favorable cells.
The pilot must not be rerun with modified cells, thresholds, seeds or comparators to rescue a failed gate.

The machine-readable authority is `docs/perishable_update_phase_pilot_manifest.json`.
