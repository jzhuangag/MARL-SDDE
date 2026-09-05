# T-036 exact affine finite-state Markov moment theorem

## Purpose

T-036 closes the exact finite-horizon moment recursion for an exogenous
finite-state affine Markov jump system with bounded parameter delay.  Unlike
an iid substitution, it retains dependence between the current Markov mode
and the complete delayed iterate state.  The result is an exact theorem and a
proof benchmark; its joint-mode dimension is not proposed as an online
algorithm.

## Setup

Let \(Z_t\) be a finite exogenous Markov chain with transition matrix \(P\).
Let \(x_t\) contain the current error and its \(D\) delayed values.  Conditional
on \((Z_t,Z_{t+1})=(a,b)\), augment the affine update as

\[
\tilde x_{t+1}=G_{ab,t}\tilde x_t,
\qquad
\tilde x_t=[x_t^\top,1]^\top.
\]

The transition law is independent of the iterate, and the conditional update
matrix has a finite second moment.  It may contain a random Jacobian, an
additive temporal-difference innovation, and the delay shift.

## Theorem: exact mode-conditioned full risk

Define unnormalized mode-conditioned moments

\[
m_t^a=\mathbb E[\tilde x_t\mathbf1\{Z_t=a\}],
\qquad
M_t^a=\mathbb E[\tilde x_t\tilde x_t^\top
                 \mathbf1\{Z_t=a\}].
\]

Then

\[
m_{t+1}^b
=\sum_a p_{ab}\,
  \mathbb E[G_{ab,t}m_t^a\mid a,b],
\]

and, when the conditional matrix randomness is independent of the past state
given \((a,b)\),

\[
M_{t+1}^b
=\sum_a p_{ab}\,
  \mathbb E[G_{ab,t}M_t^aG_{ab,t}^\top\mid a,b].
\]

For deterministic conditional updates, these equations are an immediately
computable exact recursion.  Summing over \(b\) gives the unconditional mean,
second moment, and covariance at every finite horizon.

### Proof

Multiply the augmented recursion and its outer product by
\(\mathbf1\{Z_t=a,Z_{t+1}=b\}\), take conditional expectations, and use the
exogeneity of the Markov transition.  Conditional independence permits the
past moment to leave the expectation over \(G_{ab,t}\).  Summing over the
source mode gives each target-mode recursion.  No step replaces the current
sample matrix by its stationary mean, so the sample/iterate dependence is
retained exactly.

## Verification

The implementation is tested against complete path enumeration.  A symmetric
two-state sign chain with persistence \((1+\lambda)/2\) has autocovariance
\(\lambda^{|s-t|}\); for delays 0, 1, and 3 its exact second moment agrees with
the independent T-035 impulse-response formula to numerical precision.  A
separate test confirms that mode-conditioned iterate means differ, explicitly
ruling out a hidden iid assumption.

## Consequence for T-034

The result supplies an exact nonvacuity benchmark for finite-state tabular
tasks and can classify speedup, saturation, and reversal without a
variance-only proxy.  It does not yet provide a dimension-free finite-time
bound for large joint mode spaces or predictable adaptive policies.  That
extension still requires a Poisson-equation/martingale argument with explicit
mixing, delay, and cross-agent covariance constants.
