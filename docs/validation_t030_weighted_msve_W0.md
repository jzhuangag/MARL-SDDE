# T-030 W0 diagonal weighted-MSVE validation

## Decision

W0 fails: none of the five prospectively frozen diagonal metrics passes the
5% optimistic nonvacuity gate.  Under the preregistered rule, W1/W2, another
sampled CPU experiment, MinAtar Asterix, HPC4, GPU, and formal seeds remain
unauthorized.

## Provenance

- preregistration commit:
  `385530939411dc51e12baf6191726284ec212187`;
- configuration SHA-256:
  `84cc7fde97b1dea73cf23d579f55734fcb6237979451ff1b4cc8009e463d4ebf`;
- scientific trajectories: zero;
- result SHA-256:
  `7ed544bd1bde9b56836fa51371d49ac5fd3a2ba8e08431eb2475a3c49aeefa92`;
- isolated reproduction: byte-identical;
- complete experiment tests: `303 passed, 7 skipped`.

## Frozen metric results

| theta in W=diag(pi^theta) | monotonicity | optimistic maximum improvement | W0 |
|---:|---:|---:|---:|
| 0 | 3.47860e-5 | **0.0105612%** | fail |
| 0.25 | -9.10973e-6 | 0 | fail |
| 0.5 | -1.54467e-4 | 0 | fail |
| 0.75 | -4.43571e-4 | 0 | fail |
| 1 | -9.50301e-4 | 0 | fail |

The theta-zero row reproduces the Euclidean audit.  Every positive theta has
negative symmetric drift monotonicity for the unchanged plain TD update on
this nonreversible chain, so the weighted Theorem 4 stability premise already
fails before adding noise, mixing error, or delay.

## Scientific interpretation

This result does not invalidate delayed affine convergence or the exact
correlation variance identity.  It rules out the prospectively defined
low-memory diagonal-metric certificate as a practical fixed-participation
selector on Blackjack.  A dense/ad-hoc Lyapunov metric was explicitly not
allowed as a retrospective rescue: it would obscure the low-complexity story
and would not preserve coordinate-projection nonexpansiveness automatically.

Combined with EXP-019A's 0.01665% realized AUC gain, W0 removes the current
evidence basis for an ICML claim that the proposed participation rule yields
meaningful finite-time learning improvement.  The defensible project is now
a narrower theory paper: correlation-limited speedup, delayed affine Markov
convergence, finite-budget adaptation/opportunity lower bounds, and the
unrestricted unknown-mixing impossibility.  SDDE remains an interpretation
layer, not the discrete convergence proof.
