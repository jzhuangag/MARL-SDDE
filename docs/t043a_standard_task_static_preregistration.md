# T-043A preregistration: outcome-free standard-task phase scan

## Purpose

T-041A establishes that the exact additive theorem has nonempty speedup,
saturation, and reversal regions. T-043A is the next, deliberately weaker
screen: it asks whether constants extracted from two unmodified public finite
Markov tasks place a prospective standard-task experiment in those regions.
It generates no sampled learning trajectory and is not empirical evidence.

The tasks are Gymnasium 1.0.0 `FrozenLake-v1` (4x4, slippery) and
`CliffWalking-v0`, both under a frozen uniform-random policy. Terminal
transitions receive zero bootstrap and then reset from the exact upstream
initial distribution, yielding a continuing regenerative chain with an exact
finite kernel. Features are deterministic Fourier columns, orthonormalized
under the exact stationary law. No task outcome, seed, or earlier failed
controller selects the features.

## Valid cross-agent dependence

A future sampled experiment may couple fresh inverse-CDF uniforms across
agents. Every agent always evolves its own current state through the exact
policy-induced kernel. A common uniform and private uniforms have identical
uniform marginals, so mixing them preserves every single-agent law. Unlike
EXP-007A's old pair-level construction, transition pairs from unrelated paths
may not be spliced. At `rho=1`, common initialization, action, transition, and
reset uniforms make the trajectories identical; at `rho=0`, all draws are
private.

## Registered analytic surrogate

For each task, T-043A computes the exact regenerative kernel, stationary law,
projected TD fixed point, symmetric-drift minimum, drift operator norm,
stationary TD-innovation trace second moment, and Markov SLEM. Those constants
enter the T-035 scalar delayed finite-horizon risk with the exact
equicorrelation factor `rho+(1-rho)/q` and the T-037 dual-budget update count.

This surrogate is a necessary design screen. It is not the exact risk of the
multiplicative projected-TD recursion, and a pass cannot be reported as task
performance. T-042 states the precise missing small-gain step.

## Frozen scope and gates

The machine-readable configuration freezes 144 scenarios and nine actions per
scenario (1,296 rows): two tasks, two resource rays, three budgets, three
delays, four dependence levels, three participation counts, and three
normalized step sizes. S1--S10 require valid task constants, finite stable
rows, at least 80% independent/environment speedup directions, exact
high-correlation no-value cells, at least 90% high-correlation/message
reversal, q=1 and q=16 support on both tasks, at least 5% message-ray oracle
value over the strongest fixed q, at least 40% strict cell improvement, zero
sampled outcomes, and byte-identical reproduction with SHA-256 provenance.

Any failed gate forbids EXP-020A preregistration. Tasks, policies, budgets,
thresholds, and cells cannot be amended after this commit to rescue the scan.
Passing only authorizes a separately committed fresh-seed CPU learning
preregistration; it never authorizes formal seeds, GPU, or HPC4.

Frozen configuration SHA-256:
`bb37cd590f1c620c5ae386f9516ce2be5347534451c2a95d35b09f697df504cf`.
Frozen runner SHA-256:
`0a38c19000c6b60d6624cf0d82e1b1e0612eaddcaf8b7bce89fef8e31bbd09eb`.

## Source verification

The task definitions are pinned to the locally tested Gymnasium 1.0.0 package.
The official Gymnasium documentation specifies FrozenLake's 16-state,
four-action slippery transition model and CliffWalking's 48-state,
four-action grid with its reward and terminal conditions. These upstream task
definitions are descriptive sources; the actual experiment provenance is the
pinned package version plus the recorded extracted-kernel hashes.
