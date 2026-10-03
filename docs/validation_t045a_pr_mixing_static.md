# T-045A validation: final PR/mixing standard-task scan

## Decision

T-045A is a reproducible static failure. EXP-020B is forbidden, and the
frozen stop rule ends further standard-task redesign under the current ICML
mainline. No sampled trajectory, seed, confidence interval, pilot, formal
run, GPU, or HPC4 job was used.

Mixing-normalized PR averaging repairs part of T-043A's estimator mismatch:
CliffWalking now moves from q16 under weak correlation to q1 under strong
correlation, with q4 in transition cells. But the effect is too small and not
shared by FrozenLake, so it fails the practical adaptation gates.

## Frozen results

| Gate | Observed | Verdict |
|---|---:|---|
| M1 task/kernel validity | exact inherited hashes | PASS |
| M2 finite/stable PR rows | 1,728/1,728 | PASS |
| M3 independent message speedup | 1.000 | PASS |
| M4 correlated message reversal | 0.500 | **FAIL** |
| M5 correlated environment no-value | maximum 0.000 | PASS |
| M6 per-task q support | Cliff `{1,4,16}`, Frozen `{1,16}` | PASS |
| M7 aggregate oracle improvement | 0.7044% | **FAIL** |
| M8 strict message-cell improvement | 29.17% | **FAIL** |
| M9 no outcome leakage | zero trajectories | PASS |
| M10 reproduction/provenance | byte-identical | PASS |

The result is 7/10 gates. It may not be rounded to 5%, pooled with T-043A,
or rescued by changing the task weights.

## Task-level diagnosis

CliffWalking has the intended direction: every `rho=0` message cell selects
q16, every `rho=0.9` or `rho=1` message cell selects q1, and three `rho=0.1`
cells select q4. Nevertheless its oracle improvement over the strongest
fixed-q comparator is only 1.404%, below the 5% practical gate.

FrozenLake selects q16 in all message cells. Its projected fixed point has
very small norm relative to its stationary TD noise, so even PR averaging and
mixing-scaled steps do not make additional perfectly correlated updates
valuable within the frozen horizons. Its adaptation ceiling is exactly zero.

The aggregate oracle/fixed geometric ratio is 0.992956, and only 29.17% of
message cells improve strictly. This is an oracle ceiling; a learned selector
would be weaker after paying identification cost.

## ICML consequence

T-034's hard stop required a standard task with at least 5% positive value
and a no-value regime. T-045A's strongest task reaches only 1.404%. Because
T-045A was explicitly the final standard-task feasibility attempt, the
current correlation-adaptive participation line is not allowed to proceed to
another standard-task design, EXP-020B, a nonlinear transfer, or GPU formal
experiments.

The exact phase, minimax, adaptation-cost, Poisson, and PR-average theorems
remain valuable for a sharply scoped theory/TSP manuscript. A future ICML
attempt would have to be a genuinely new research question with an
independent preregistration—not a renamed continuation of this controller.

## Reproduction

- preregistration commit: `b79d886`;
- primary/reproduction runtimes: 11.858/13.010 seconds;
- rows SHA-256:
  `87608012dee3bdfa762b9d0e09d139c4b8a2e26647ad2319e0f889c7cb47ef89`;
- task constants SHA-256:
  `034103f50a17c2638d61ebd120bbd59cba13e9a635d5d4f71fa4900ed20afae8`;
- summary SHA-256:
  `114b0d74ddc52f7ceb5f5675c2ca6b84399f04e3f3c689487eeec3170ea1d567`.

All three artifacts are byte-identical in the isolated reproduction.
