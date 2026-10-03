# T-065 discrete foundation audit

## Decision

The claim-critical stability route is now discrete.  The executed delayed
Markov recursion will be analysed directly with a lifted common quadratic
Lyapunov function.  An SDDE or Lyapunov--Krasovskii limit is not required for
the controller, its budget guarantee, or its primary finite-time theorem.  It
may be added later only as a separately proved interpretation or asymptotic
corollary.

This is an outcome-free foundation audit.  It contains no controller outcome,
scientific seed, benchmark comparison, or formal evidence.

## Observable paired-block identity

At a predictable iterate, two disjoint Markov blocks produce

\[
\widehat F^{(1)}=F+\epsilon_1,\qquad
\widehat F^{(2)}=F+\epsilon_2.
\]

Conditional independence and zero conditional means imply

\[
\mathbb E\langle\widehat F^{(1)},\widehat F^{(2)}\rangle=\|F\|^2,
\quad
\mathbb E\frac{\|\widehat F^{(1)}-\widehat F^{(2)}\|^2}{2}
=\operatorname{tr}\operatorname{Cov}(\epsilon_1).
\]

For a stationary unit-variance AR(1) chain with lag-one correlation `lambda`,
the exact scalar variance of a length-`L` block mean is

\[
\kappa_L(\lambda)=\frac{1}{L^2}\left[L+2\sum_{k=1}^{L-1}
(L-k)\lambda^k\right].
\]

The implementation tests both the exact variance formula and the two residual
identities using independently generated Markov blocks.  These blocks must be
charged to the same message and actor-transition budgets in every future
runner; the current test establishes the statistical identity, not that later
accounting automatically does so.

## Discrete delayed certificate

For

\[
e_{t+1}=e_t-\eta A e_{t-D},
\qquad z_t=(e_t,e_{t-1},\ldots,e_{t-D}),
\]

let `C_D(eta)` be the corresponding lifted companion matrix.  Offline, solve

\[
\max_{P,m}\ m
\quad\text{s.t.}\quad
P\succeq\epsilon I,\ \operatorname{tr}P=1,
\ P-C_D(\eta_j)^\top P C_D(\eta_j)\succeq mI
\]

at the two endpoints `eta_j in {eta_min, eta_max}`.  Since `C_D(eta)` is affine
in `eta` and `C -> C^T P C` is matrix convex for `P >= 0`, the two endpoint
constraints imply the same bound throughout the interval.  The dense grid is
only a numerical audit.

For `A=diag(0.5,1)`, `eta in [0.005,0.02]`, the CLARABEL solution gives:

| Delay | SDP margin | smallest eig(P) | worst grid drift eig | worst radius |
|---:|---:|---:|---:|---:|
| 0 | 0.00332778 | 0.333612 | -0.00332778 | 0.997500 |
| 1 | 0.00163643 | 0.00407347 | -0.00163643 | 0.997494 |
| 2 | 0.00107312 | 0.00345978 | -0.00107312 | 0.997487 |
| 4 | 0.000623273 | 0.00291533 | -0.000623273 | 0.997475 |
| 8 | 0.000324955 | 0.00244897 | -0.000324955 | 0.997448 |

The decreasing margin correctly exposes the delay penalty.  A Lyapunov matrix
constructed only at the midpoint gain failed for delays 4 and 8 in a
preliminary numerical check, so midpoint stability is not substituted for a
common-certificate argument.

## Complexity and interpretation

The SDP dimension is `(D+1)d` and is solved once per public drift/delay class,
not online.  Online work remains the `O(d)` residual summaries plus the `O(1)`
continuous joint minimizer and at most two integer evaluations from T-064.
There is no online covariance matrix, LMI, QP, SOCP, Hessian inverse, or scan
over all participation levels.

## Current authorization

The foundation gates authorize a separately frozen CPU mechanism pilot.  They
do not yet establish finite-time Markov TD risk, sensor confidence sequences,
pathwise dual-budget feasibility, or superiority over strong fixed and
one-dimensional adaptive baselines.  Those remain mandatory before a
paper-facing experiment.
