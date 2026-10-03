# T-045A preregistration: final PR/mixing standard-task scan

## Why this is a new theorem-driven screen

T-043A permanently failed and EXP-020A remains forbidden. Its registered
constant-step terminal estimator, and a subsequent descriptive AUC alignment
check of the same estimator, both gave zero message-regime adaptation value.
T-044 identifies the estimator mismatch and proves the exact finite-horizon
Polyak--Ruppert (PR) averaged risk. T-045A tests that theorem-facing estimator;
it does not amend, overwrite, or relabel T-043A.

The step size is no longer normalized only by the drift norm. It is frozen as

\[
\eta=c\,(1-\lambda)/\|A\|_{\rm op},
\qquad c\in\{0.1,0.25,0.5,1\},
\]

where \(\lambda\) is the exact SLEM certificate of the public regenerative
task kernel. This scaling follows the mainline's mixing dependence and the
T-017 restriction to known or independently certified mixing. It is not fit
from T-043A rows. The estimator averages the last half of the iterates.

## Frozen invariants

FrozenLake/CliffWalking versions, kernels, policy, terminal/reset treatment,
features, task-kernel hashes, rho levels, delays, q catalogue, budget rays,
budgets, strongest-fixed-q grouping, and practical thresholds are unchanged
from T-043A. The only new action coordinate is the registered mixing-step
multiplier, and the only new risk estimator is the T-044 tail PR average.

The scan has 144 scenarios, 12 actions per scenario, and 1,728 deterministic
rows. It uses no T-043A result file, sampled trajectory, seed, confidence
interval, pilot, formal run, GPU, or HPC4 resource.

## Final gates

M1--M10 require exact task validity; all finite/stable PR windows; at least 80%
independent message cells with a 5% q16 gain over q1; at least 90% perfectly
correlated message cells selecting q1; at most 1% oracle value in every
perfectly correlated environment cell; q1/q16 support on both tasks; at least
5% aggregate message oracle value over the strongest fixed q; at least 40%
strict message-cell improvement; zero outcome leakage; and byte-identical
reproduction with complete SHA-256 provenance.

This is the final standard-task feasibility attempt. Any failed mandatory
gate forbids EXP-020B and stops further standard-task redesign under the ICML
mainline. A pass would authorize only a separately committed fresh-seed local
CPU learning preregistration, never formal seeds or GPU.

Frozen SHA-256 provenance:

- configuration: `09af7386c8416dbc24e8a8fb2d3ea3ccc49867ecdabdf3e0af2ad6c3fb23a92c`;
- runner: `8382a00acbe8f38b90b4c0dd23085577a68abbb2e6b47b8650d511d5d0e4cc02`;
- T-044 exact PR core: `a8146c149d3fa2d0c4e385ffe811bde56dd0d154c1dd5887a48a4f17559fb355`.
