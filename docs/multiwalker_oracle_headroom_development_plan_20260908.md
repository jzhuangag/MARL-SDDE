## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: prospective development-experiment plan
- Origin Date: 2026-09-08
- Verification Status: FROZEN FOR DEVELOPMENT SEEDS ONLY; NOT A FORMAL PREREGISTRATION
- Version Label: multiwalker_oracle_headroom_development_plan_v1

# Multiwalker equal-resource oracle-headroom development plan

## Question

Before fitting a Lyapunov controller, does the actual Multiwalker transition
kernel contain enough state- and edge-dependent policy-profile value to make
dynamic one-edge synchronization scientifically meaningful?

This is a development kill test.  Counterfactual outcomes are not supplied to
a deployable method.  Its seeds can never become confirmation or formal
seeds, and passing does not establish a learning improvement.

## Exact counterfactual construction

The five current actors generate one public all-current behavior prefix.  At
each event `t`, the owner is `t mod 5`.  Every evaluated policy sees the same
public environment state and current actor versions, but maintains its own
recipient-specific policy caches.

Multiwalker's Box2D wrapper cannot be safely copied or pickled.  Each
counterfactual therefore resets with the same environment seed and exactly
replays the complete public action prefix.  The null branch is executed twice
and its actions, observations, one-step reward, and discounted return must be
bit-exact.  A candidate then substitutes either no cache, one physical-chain
neighbor cache, or (for the complete-refresh comparator) both neighbor caches.

The branch estimand is the discounted mean team reward over eight steps,

\[
Y_t(a)=\sum_{h=0}^{7}0.99^h\bar r_{t+h}(a).
\]

The one-step oracle receives only `r_t(a)`; the privileged development oracle
receives `Y_t(a)`.  All other policies use launch-observable quantities only.

## Actor-version stress and resource constraint

Current actor versions follow a deterministic bounded output-layer update with
scales `0.01` and `0.04`.  This is a controlled policy-version stress, not an
RL optimizer or a claimed benchmark solution.  Scale `0.04` is the declared
active freshness phase; scale `0.01` is a weak-motion control.

Each one-edge cache copy is charged the exact serialized actor parameter
payload.  Budget rates are `0.25` and `0.50` edges per launch.  Every policy
must satisfy the prefix constraint

\[
N_{\rm edge}(t)\le \lfloor b(t+1)\rfloor
\]

and receives 40 common-state launches.  A complete neighbor refresh costs its
full number of edges; it cannot be executed when the prefix capacity is
insufficient.

## Comparator family

The privileged eight-step one-edge oracle is compared with the envelope of:

- a privileged one-step reward oracle;
- observable current-versus-cached action gap;
- observable parameter gap;
- cache age;
- round robin and seeded random;
- fixed left and fixed right edge;
- complete local-neighbor refresh;
- no optional refresh.

The envelope is selected per seed, motion scale, and budget after all methods
are evaluated.  This intentionally makes the development gate harder than a
comparison to one fixed heuristic.

## Frozen development matrix and gates

- seeds: `95700--95707`;
- walkers: 5;
- events: 40;
- counterfactual horizon: 8;
- discount: 0.99;
- motion scales: `0.01`, `0.04`;
- budget rates: `0.25`, `0.50`;
- policies: 11;
- expected result rows: 352.

All seven gates must pass:

1. every return and aggregate is finite;
2. repeated null replay is bit-exact;
3. every policy respects its prefix budget;
4. active-phase oracle improvement over no refresh is positive;
5. active aggregate headroom over the strong online envelope is at least 10%
   of oracle improvement over no refresh;
6. oracle beats the strong envelope in at least 75% of active seed-budget
   cells;
7. median active absolute headroom is at least 0.1% of the corresponding
   no-refresh return magnitude.

Failure stops this Multiwalker formulation before critic fitting.  Passing
authorizes an independent-seed oracle confirmation plan and a separate
selected-packet critic nonvacuity design; it does not authorize a controller
efficacy claim, formal experiment, GPU, or HPC4.
