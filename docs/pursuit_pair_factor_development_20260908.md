## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: privileged estimator-class development audit
- Origin Date: 2026-09-08
- Verification Status: DECLARED HIGH-DIMENSIONAL HEAD FAILED; NO EFFICACY AUTHORIZATION
- Version Label: pursuit_pair_factor_development_v1

# Pursuit pair-factor development result

## Decision

The compatible pair-factor algebra remains valid, but the declared
high-dimensional linear pair head does not pass its held-out conditional-mean
test.  This development result authorizes neither an online controller
experiment nor a GPU benchmark.

It also does not refute the pair-factor identity.  The fit used 875
coefficients with only 90 training edge rows.  The correct conclusion is that
small privileged counterfactual datasets cannot identify this head, not that a
factor critic trained from the much larger stream of completed transitions is
impossible.

## Immutable development protocol

The final completed run used fresh seeds after two runtime-only attempts had
been stopped without writing an output file:

- train: `93600--93603`;
- validation: `93604--93605`;
- test: `93606--93607`;
- 32 launch events per seed;
- eight-step packets;
- four independent counterfactual replicates per candidate;
- drift fraction `0.20`;
- ridge candidates `{0.01, 0.1, 1, 10, 100}` selected only on validation;
- four seed workers; elapsed wall time `809.82 s`.

The output is ignored development material at
`tmp/policy_dependency_sync/pursuit_pair_factor_development_v3.json`, SHA-256
`ADCCDA959C31FEF1E0CCA2BEDDA871A87CBC2151A8331EB5EB1B3A54E18B8266`.
The source-bundle hash embedded in that output is
`f1c51235475f33736a7043b2255205297ebad7eab97a58c9d191022dffae0d71`.

The first serial scale attempt used seeds `93400--93415` and was stopped at the
declared 40-minute runtime boundary.  The second used `93500--93507`, completed
the train collection, and was stopped before validation when the lack of
per-seed parallelism was diagnosed.  Neither attempt wrote an artifact or
exposed a fitted test result; none of these seeds is reused.

## Held-out result

The design contained 90 train, 44 validation, and 29 test edge rows, with 875
columns.  Validation selected ridge `100`.  On the untouched test split:

| Quantity | Value |
|---|---:|
| Conditional edge-effect `R^2` | -1.068439 |
| Mean absolute error | 0.0008032 |
| Target standard deviation | 0.0009410 |
| Nonzero sign accuracy | 0.458333 |
| Best action accuracy, including null | 0.450000 |
| Launches with a strictly positive oracle edge | 0.600000 |
| Mean absolute edge effect | 0.0004841 |

Every selected real branch still matched its copied-state counterfactual audit
exactly: the maximum selected reward/alignment discrepancy was zero.  Thus the
failure is not caused by a branch simulator mismatch.

## Interpretation and stop rule

The run is a failure of the declared 875-dimensional linear estimator at the
available counterfactual sample scale.  It is not admissible to increase its
feature dimension, add layers, search additional ridge values, or reuse the
test seeds.  Generic context regression and privileged high-dimensional pair
regression are now both stopped.

The only scientifically distinct next estimator is a centralized pair critic
trained on selected, fully charged completed transitions.  It must exploit
parameter sharing and a low-dimensional local factor representation, and it
must be evaluated on independent conditional-mean audit episodes.  Its critic
training signal cannot contain unexecuted counterfactual branches.  Before a
standard-MARL efficacy run, it must establish:

1. positive held-out prediction or ranking value over a null-only decision;
2. a nonvacuous simultaneous factor/statistical error allowance;
3. measured `O(Delta)` candidate-scoring cost at fixed action count;
4. equal-resource oracle headroom over strong online baselines.

If the selected-packet factor critic fails these conditions, Pursuit is stopped
as the principal benchmark for this mechanism.  Renaming the same estimator or
weakening the comparator is not an admissible continuation.
