# T-020 final decision

## Decision: permanently stop EXP-017B

EXP-017B must not be preregistered or run. Its corrected strong fallback must
include q=1 when q=1 wins the task×budget pilot selection. Against that
fallback, the cellwise fixed-q oracle improves aggregate geometric terminal
error by only 0.3846%, strictly improves only 21/72 cells, reaches 2% in 6/72,
and reaches 5% in none. The required static gate is 5% aggregate and 60%
directional cells; both fail before accounting for any learning degradation
from probes.

The old `ratio <= 1.05` no-harm gate is insufficient because always executing
the fallback passes with ratio one. A nontriviality gate is mandatory, but the
current benchmark lacks enough oracle value for any controller to satisfy a
meaningful one broadly.

## Gate ledger

| Gate | Result | Consequence |
|---|---:|---|
| Correct task×budget fallback includes q=1 | pass | Acrobot/message and CartPole/message use q=1 |
| Full probe message/environment accounting | pass | usable updates fall by about 0.9% |
| Static oracle aggregate improvement ≥5% | **fail** | observed optimistic ceiling 0.3846% |
| Strict oracle improvement in ≥60% cells | **fail** | 21/72 = 29.17% |
| Existing no-harm gate nontrivial | **fail** | permanent fallback passes trivially |
| Outcome-free stochastic benchmark candidate passes all conditions | **fail** | FrozenLake and MinAtar retain missing 5%/mixing certificates |
| Learning-value formula and observable inputs specified | pass as design | not a preregistration |
| Complete surrogate/no-harm theorem | **fail/open** | ReLU smoothness and time-uniform confidence certificate absent |
| CPU unit tests | pass, 10/10 | algebra only |
| GPU/Slurm execution | not run | required by T-020 |

## What is preserved

- EXP-017A code, seeds, G1–G12 results, negative conclusion, and HPC4
  artifacts remain unchanged.
- T-019 proof, phase diagram, static design, and provenance remain unchanged.
- The T-019 non-q1 fallback remains as historical design provenance; T-020
  records why it is not the correct strong baseline.
- Both pilot seeds remain descriptive design data, never formal evidence.

## Conditions for any future nonlinear GPU pilot

A future effort must use a new experiment identifier, not EXP-017B. It may be
preregistered only after all of the following are simultaneously complete in
an outcome-free commit:

1. a standard intrinsically stochastic fixed-policy task and unchanged-law
   cross-agent coupling;
2. public task/model parameters and theoretical budget rays yielding an
   internal participation optimum;
3. a static, full-probe-cost task×budget oracle ceiling of at least 5%
   aggregate improvement and at least 60% directionally improved cells;
4. a fully specified observable learning-value surrogate;
5. a proved confidence/smoothness certificate supporting its conditional
   no-harm shield;
6. CPU tests for marginal invariance, probe charging, saturation, horizon,
   fallback, nontriviality, taint, and vectorized complexity;
7. new seeds, runner/analyzer hashes, mandatory gates, and a rule that any
   failed gate stops formal without alteration.

T-020 satisfies none of the authorization conditions collectively. No new
number is assigned and no GPU pilot is authorized.
