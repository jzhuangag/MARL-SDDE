# T-029 exact Blackjack static-scan validation

## Decision

All eight frozen T-029 gates pass.  The result authorizes a separately
preregistered local-CPU Blackjack learning pilot.  It does not authorize
formal seeds, MinAtar, HPC4, GPU execution, or a nonlinear convergence claim.

## Frozen provenance

- parent/preregistration commit:
  `f56b3b8a69695cc9df9db225437b055485bc37fe`;
- configuration SHA-256:
  `c2f0001a2144e93d3ed37983e3fc8baf702b0d1e114891e837f0ae6d9c289e7b`;
- scientific trajectories: `0`;
- exact arm rows: `432`;
- exact comparison cells: `72`, including 36 prospectively active message
  cells and 36 prospectively inactive environment cells.

## Exact-kernel result

The reachable continuing observation chain is finite and stochastic.  The
registered epsilon-soft policy provides a reset-minorization floor of 0.1.
The exact worst-state total-variation distance first falls below 0.05 at
stride 5; the observed distance there is `0.01099600785576694`.

## Frozen value gates

| Quantity | Frozen gate | Result |
|---|---:|---:|
| Aggregate oracle improvement | >=5% | **9.5041%** |
| Strict active-cell improvement | >=60% | **30/36 = 83.3333%** |
| Inactive boundary behavior | 36/36 | **36/36** |
| Distinct oracle q values | >=3 | **6: 1,2,4,8,16,32** |
| Environment q >= message q | 36/36 | **36/36** |
| Message q non-increasing in rho | 6/6 paths | **6/6** |

All exact-validity, mixing, value, direction, diversity, and no-taint gates
pass.  No cell, threshold, q value, or fallback was changed after observing
the scan.

## Reproduction

The scan was run twice in isolated output directories.  Both `summary.json`
files have SHA-256
`e6a14a1ba541d5411962b29521b90764a575e00f41ba1c6331a53bc66d9bd6e5`
and are byte-identical.  The complete experiment test suite reports
`288 passed, 7 skipped`.

## Interpretation boundary

The 9.50% quantity is a prospective mechanism ceiling computed from the
registered correlation-limited risk proxy and exact dual-budget horizon.  It
is not a realized TD-learning improvement.  The exact Blackjack kernel
certifies the task and mixing layer, but a sampled CPU pilot must still show
that fixed-q selection transfers to normalized value-prediction error.  Even
a successful Blackjack pilot remains a theorem-aligned calibration result;
an independently gated external nonlinear benchmark is still required for a
competitive ICML package.
