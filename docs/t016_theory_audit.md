# T-016 theory audit

## Decision

**Stop gates B and D. Do not preregister or run EXP-016A.**

AC-7 and the instance-dependent adaptive Pareto lower-bound route are closed
for the registered known-mixing Gaussian model. A theorem-derived fallback
rule now provides explicit per-scenario safety slack and proves finite-risk,
error, short-fallback, and conditional long-horizon commit guarantees.
However, a universal constant/log matching result for the adaptive
controlled-belief lower bound was not proved, so AC-9 remains open. Fully
unknown mixing also lacks uniform coverage near \(\lambda=1\), so AC-8
remains open and the main claim must be narrowed.

## Frozen evidence

EXP-015A remains an honest 7/8 pilot failure. Its fallback threshold remains
`0.80`; its observed value remains `0.777778` and is neither rounded upward
nor retuned. No seeds were added. The phase transition remains informative
but cannot authorize formal work. All-agent individual feedback remains
informative, and no incompatibility theorem was asserted.

## Proved in T-016

- AC-7: exact stopped adaptive change of measure, including distinct
  hypothesis filters, changing \(q\), irregular \(b\), selection, randomized
  common action kernels, stopping, dual budgets, and delayed usability.
- Both directional information constraints for every \(\delta\)-correct
  decision.
- Instance-dependent opportunity-cost lower bound as a controlled Kalman
  belief occupation program retaining \(q,b,\lambda,\theta,h\), both budgets,
  and \(D\).
- An explicit \(\epsilon_{\rm safe}\) and strict fallback inequality for a
  finite known-mixing probe/action catalogue.
- Fixed-design error, excess-risk, safety-deficit, automatic short fallback,
  and conditional long-budget commit bounds.
- Interior covariance identifiability of \((\theta,\lambda)\), plus the
  non-uniform boundary obstruction at \(\lambda=1\).

## Assumption boundary

The observation theorem uses a stationary Gaussian scalar common factor,
known unit private noise, two simple \(\theta\) hypotheses, public
\(\lambda\in[0,1)\), finite predictable actions, and a common policy kernel.
Stopping is pathwise budget-bounded; the more general localized statement
requires uniform integrability. The fallback theorem additionally requires a
finite stability-screened catalogue and valid finite-budget risk bounds.
Delay enters through resource use and usable horizon, not through the
observation covariance. No nonlinear convergence or SDDE approximation is
claimed.

## Verification

The new test module performs:

- 70-digit conditional-KL comparison;
- pathwise Kalman versus dense Gaussian likelihood comparison;
- full-vector spatial reduction and brute dense checks over every branch of
  a small selection-dependent, dimension-changing adaptive tree;
- Monte Carlo checks of both expected directional KL identities;
- likelihood-ratio martingale and Ville optional-stopping checks;
- boundaries for \(q=1,\lambda=0,\lambda=1\), exact dual-budget exhaustion,
  \(D\) beyond horizon, and strict fallback ties.

The clean repository-wide run completed with `148 passed in 6.89 s`; 14 of
these are the new T-016 likelihood and boundary audits. The exact commit is
recorded in the final handoff after push.

## Minimum unresolved gap

The smallest next theoretical gap is to relate a predictable
information-directed/track-and-stop allocation to the controlled-belief
occupation optimum uniformly over a compact known-mixing parameter set,
including a constant or logarithmic performance factor. Only after that
result closes AC-9 may a separately committed CPU EXP-016A preregistration be
considered. A later unknown-mixing extension must either prove uniform
composite coverage on \(\lambda\le1-\gamma\) or explicitly consume a
separate mixing certificate.
