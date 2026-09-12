# T-070A exact nonstationary collaboration-graph preregistration

## Purpose

T-070A tests whether time variation creates collaboration value that cannot be
captured by a strong time-invariant personalized graph. It is an exact affine
Markov-moment feasibility scan, not sampled or formal evidence.

Frozen configuration SHA-256:
`08a0621afc6c28fca79c64eaa1c9e5cb89e3bf2c00a4b6d4533b2b01827ebb43`.

The experiment is outcome-free at this commit. No T-070A result directory may
exist before this preregistration is committed.

## Frozen model

Four recipients follow scalar strongly monotone affine TD dynamics. At a
decision block, each recipient either remains local or mixes with one delayed
donor using weight 0.5 or 1.0. This gives seven actions per recipient and
\(7^4=2401\) time-invariant recipient-specific graphs.

The public target patterns are

\[
p^A=(-1,-1,1,1),\qquad p^B=(-1,1,-1,1).
\]

The stationary control uses only \(p^A\). The single-switch schedule changes
from \(p^A\) to \(p^B\) after block 11. The alternating schedule uses
\(p^A,p^B,p^A\) for three eight-block segments. Thus the beneficial donor of
agents 1 and 2 changes without changing the action catalogue or marginal noise
law.

## Comparison and charging

The primary comparator is the cellwise best of all 2,401 static personalized
graphs, selected by exact cumulative risk and given zero sensing cost. The
dynamic ceiling is therefore compared against a deliberately strong static
baseline.

The dynamic oracle receives exact moments but is nevertheless charged two real
probe transitions at each of six decision blocks. Probe samples are not
learning updates. A same-data local shadow receives the same reduced learning
horizon. Safety is checked recipient-wise at every decision checkpoint.

The primary metric is personalized parameter mean-square error averaged over
all 24 blocks. Terminal error is descriptive because a terminal-only static
oracle can specialize to the final segment and does not measure tracking.

## Static-separation lemma motivating the scan

Let \(\ell_t(w)\) be a \(\mu\)-strongly convex instantaneous transfer risk on a
convex graph domain, and let \(w_t^*\) minimize \(\ell_t\). Then every static
graph \(w\) satisfies

\[
\sum_{t=1}^T\left[\ell_t(w)-\ell_t(w_t^*)\right]
\geq \frac{\mu}{2}\sum_{t=1}^T\|w-w_t^*\|^2.
\]

Minimizing the right-hand side over static \(w\) yields a positive lower bound
whenever the optimal graph varies. T-070A does not claim this continuous-domain
lemma as a theorem for the finite catalogue; it tests whether the registered
Markov model exhibits a practically material discrete analogue after delay and
probe charging.

## Frozen gates

The exact gates R1--R12, grid, workload, schedules, and budgets are defined in
`t070a_exact_nonstationary_graph_preregistration.json`. Any failed mandatory
gate stops sampled implementation under this experiment identifier. Thresholds
must not be changed after observing T-070A results.
