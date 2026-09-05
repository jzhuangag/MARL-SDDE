# T-041A preregistration: exact finite-resource phase map

## Status

This is an outcome-free preregistration. At this commit, only static
validation and workload estimation are permitted. No registered risk row,
phase summary, or scientific result directory may exist before the
preregistration commit.

Configuration SHA-256:
`9f5149f4dacbe0603968bcc0e303139df4d3d98fdc6ac055008449a36ac3bdfd`

Runner SHA-256:
`a5cd92a1388a79b6e2316f5e3b209737d7e9293fd95ce0ec48334842989aea7d`

## Question

Do the exact T-037 vector risk identity and the T-038 Gaussian Markov design
produce the preregistered speedup, saturation, and reversal directions under
full message/environment/delay accounting, without fitting a proxy to the
outcomes?

## Frozen population

- 270 scenarios: three matrix geometries, three delays, three temporal
  coefficients, two step sizes, and five family-specific correlation values.
- six actions per scenario: \(q\in\{1,4,16\}\) and
  \(b\in\{1,4\}\).
- 1,620 exact analytic rows.
- no random seeds, Monte Carlo confidence intervals, or GPU execution.

The speedup and saturation families are environment-binding, which keeps the
primary \(b=1\) horizon equal across participation levels. The reversal family
is message-binding and starts far from the fixed point, so participation cost
can reduce contraction enough to reverse the variance benefit. Delay is
charged to the environment budget and also enters the lifted recursion.

## Frozen gates

P1--P10 are defined verbatim in
docs/t041a_exact_phase_preregistration.json. Any failure stops the positive
T-041A claim. Thresholds, cells, matrix families, and comparisons may not be
changed after observing registered risks.

## Claim boundary

Passing T-041A would validate the exact analytic phase classifier, not general
multiplicative TD, a learned controller, or a standard nonlinear benchmark.
It would authorize only a separately preregistered fresh-seed CPU
tabular/linear-TD transfer. GPU remains unauthorized.
