# T-030 final decision

## Stop the current practical participation-algorithm ICML line

The stop decision rests on three independent prospective results:

1. T-029 found a 9.50% variance-resource oracle ceiling, showing that the
   underlying phase diagram exists.
2. EXP-019A showed that this ceiling transfers only 0.242% to the registered
   finite-time TD learning improvement: realized AUC gain was 0.01665% and
   terminal gain was 0.02865%.
3. T-030 showed that neither the audited Euclidean Theorem 4 certificate nor
   any preregistered diagonal weighted-MSVE extension can prove a 5% ideal
   improvement; the positive-theta metrics are not even strongly monotone.

Accordingly, no more seeds, controller variants, task tuning, Asterix pilot,
or GPU scaling are scientifically justified for this selector.

## ICML assessment

The package is currently below the ICML 2027 bar.  It has a coherent and
potentially publishable theory core, but the executable participation rule has
negligible prospective learning benefit on the first exact standard-task
transfer test.  More compute would strengthen the negative estimate, not
create a missing effect.

An ICML attempt now requires a genuinely new algorithmic idea and a new
experiment family, not EXP-019A/T-030 amendments.  A possible future idea is a
theorem-derived transient-to-variance switch, but it must first obtain a new
nonvacuous proof and pass a fresh outcome-free analytic gate.  It is not
authorized under the current identifier.

## Recommended paper main line now

Use a theory-first title such as

> Correlation- and Delay-Limited Speedup under Multi-Agent Markov Data

and organize the paper around:

1. the exact `q/[1+(q-1)rho]` attainable speedup law;
2. delayed affine Markov-TD finite-time convergence under certified thinning;
3. dual-budget participation/identification opportunity cost;
4. the unrestricted unknown-mixing impossibility and separated positive
   regime;
5. EXP-016B and EXP-018B as prospective formal mechanism validation;
6. T-029/EXP-019A/T-030 as an honest boundary showing why variance-only
   participation does not automatically yield practical learning gains.

This is better matched to TSP, UAI, or AISTATS than to ICML in its current
form.  The online controller, preconditioning, actor-critic, and nonlinear
convergence must not be headline claims.  The SDDE/Lyapunov-Krasovskii view is
kept as explanatory geometry, while discrete-time theorems carry the formal
guarantees.
