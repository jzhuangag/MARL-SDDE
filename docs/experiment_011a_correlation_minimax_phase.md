# EXP-011A: correlation-limited minimax phase diagram

## Material Passport

- Artifact: preregistration and validation protocol
- Role: evidence for Theorem 5 and the phrase *beyond linear speedup*
- Execution: deterministic CPU audit
- Random seeds: none
- Evidence rule: this file is fixed before the first formal run

## Question

Does the exact Gaussian Markov subclass independently verify that shared
cross-agent correlation:

1. caps speedup at a constant even when more agents are available; and
2. creates a finite resource-optimal participation level?

This experiment validates an analytic minimax statement.  It is not presented
as a nonlinear MARL benchmark.

## Fixed design

Normalize total one-agent variance to one:

\[
\sigma_c^2=\rho,\qquad \sigma_e^2=1-\rho.
\]

Use

\[
\rho\in\{0,.01,.05,.1,.25,.5,.75,.9,.99\},
\quad
h\in\{1,4,16,64\},
\]

and candidate agent counts

\[
q\in\{1,2,4,8,16,32\}.
\]

The resource budget is \(B=128000\).  Every grid point records the closed-form
Fisher information, a direct dense-matrix inverse, exact minimax risk,
effective speedup, information per cost, and the adaptive-budget lower bound.

## Pre-registered gates

All three numerical gates and at least five of six scientific gates must pass.

### Numerical gates

1. Maximum relative error between the closed-form and dense-inverse Fisher
   information is at most \(10^{-12}\).
2. Maximum relative error between the risk ratio and the effective-speedup
   formula is at most \(10^{-12}\).
3. Maximum relative error between the selected information efficiency and
   the reported adaptive-budget lower bound is at most \(10^{-12}\).

### Scientific gates

1. At \(\rho=0\), speedup equals \(q\) at every candidate \(q\).
2. At every \(\rho>0\), speedup is no larger than
   \(\min\{q,1/\rho\}\).
3. For each overhead, selected \(q^\star\) is non-increasing in \(\rho\).
4. For each positive correlation, selected \(q^\star\) is non-decreasing in
   overhead.
5. Every overhead selects \(q^\star=32\) at \(\rho=0\) and \(q^\star=1\) at
   \(\rho=.99\).
6. For \(\rho\ge .5\), 32-agent speedup is no larger than two.

## Interpretation rule

A pass establishes a sharp counterexample to uniform linear speedup and
validates the resource-optimal participation phase transition in Theorem 5.
It does not establish that the finite-gap affine controller is empirically
optimal, nor that the theorem-selected \(q\) transfers unchanged to nonlinear
deep MARL.
