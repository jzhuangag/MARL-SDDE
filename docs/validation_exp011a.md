# EXP-011A validation: correlation-limited minimax phase diagram

## Decision

**PASS.**  The formal deterministic run passes all three numerical gates and
all six scientific gates.  The result supplies a sharp minimax foundation for
the phrase *beyond linear speedup*: correlated agents cannot provide uniform
\(q\)-fold improvement, even in a zero-delay, one-step-mixing subclass.

## Numerical audit

| Gate | Threshold | Result |
|---|---:|---:|
| Fisher formula versus dense inverse | relative error \(\le10^{-12}\) | \(2.83\times10^{-16}\) |
| Risk ratio versus speedup formula | relative error \(\le10^{-12}\) | \(2.28\times10^{-16}\) |
| Adaptive-budget identity | relative error \(\le10^{-12}\) | \(0\) |

## Scientific audit

| Gate | Result |
|---|---|
| Independent-agent speedup equals \(q\) | pass |
| Correlated speedup obeys \(\min\{q,1/\rho\}\) ceiling | pass |
| Optimal \(q\) is non-increasing in correlation | pass |
| Optimal \(q\) is non-decreasing in per-round overhead | pass |
| Endpoint transition \(32\to1\) from \(\rho=0\) to \(.99\) | pass |
| 32-agent speedup is at most two for \(\rho\ge.5\) | pass |

The exact 32-agent speedup is \(1.9394\) at \(\rho=.5\) and \(1.1073\) at
\(\rho=.9\).  Thus adding 31 agents yields only a 10.7% improvement at the
high-correlation point.

## Resource-optimal participation phase

The table reports the selected candidate
\(q^\star\in\{1,2,4,8,16,32\}\).

| \(\rho\) | \(h=1\) | \(h=4\) | \(h=16\) | \(h=64\) |
|---:|---:|---:|---:|---:|
| 0 | 32 | 32 | 32 | 32 |
| .01 | 8 | 16 | 32 | 32 |
| .05 | 4 | 8 | 16 | 32 |
| .10 | 4 | 8 | 16 | 32 |
| .25 | 2 | 4 | 8 | 16 |
| .50 | 1 | 2 | 4 | 8 |
| .75 | 1 | 1 | 2 | 4 |
| .90 | 1 | 1 | 1 | 2 |
| .99 | 1 | 1 | 1 | 1 |

The phase is not “always use fewer agents under correlation.”  Larger
per-round overhead makes it rational to amortize each round over more agents,
whereas stronger common noise reduces the information gained from doing so.
This interaction is exactly described by
\(q_{\rm cont}^\star=\sqrt{h(1-\rho)/\rho}\), clipped to the feasible range.

## Reproducibility

A second clean execution produced byte-identical copies of all five core
artifacts:

| Artifact | SHA-256 |
|---|---|
| `optimal_participation_phase.png` | `3C8C9EA3F980EB5625F1709DA139DE5A3ABE4CCA777A9798C18DD3682364E701` |
| `phase_grid.csv` | `1190503A44C6D71A1A36C99180910E6998C0179AA7DCBE3740C9537320ECC233` |
| `selected_participation.csv` | `C4522015DA04EB8CB4DCC1A35AD3F21EA5BEE769F56C50750E2E79C202E28199` |
| `speedup_ceiling.png` | `4D30CC4CBC30B4A087548902AFDDB5258E5AF5F349F49E7CE03DA570AB3FB30C` |
| `summary.json` | `F21CF7E6AFAC34CDCE63EE7A92FC095B6358A8D2B58E58DD52DFC491964D950A` |

## Claim boundary and next decision

The lower bound closes the main conceptual gap behind the project title.  It
does not by itself make the work ICML-ready.  The affine finite-gap upper bound
and this minimax lower bound currently meet only at the structural level; the
online estimator still needs a simultaneous confidence theorem and an
end-to-end predictable-controller experiment.  A nonlinear multi-agent
Markov benchmark is then needed for empirical breadth.  The unthinned
Poisson-equation extension and an SDDE-to-discrete approximation theorem are
valuable extensions, but neither should block the next controller milestone.
