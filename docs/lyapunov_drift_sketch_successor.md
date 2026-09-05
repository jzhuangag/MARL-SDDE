# Lyapunov drift sketch: the only admissible coupled-timescale successor

Date: 2026-09-05.

Status: **abstract design interface only; one final CPU upper-bound gate is
required before estimator construction.**

## Unified question

In asynchronous cooperative CTDE, an arriving packet offers two coupled
actions: policy progress and centralized-critic repair.  Applying the actor
update moves the critic target; delaying it preserves critic accuracy but
wastes useful policy information.  The research problem is therefore not
generic learning-rate tuning:

> Can the learner compress the conditional consequence of an asynchronous
> actor--critic packet into a constant-dimensional Lyapunov drift sketch, and
> use it to control policy progress and target tracking jointly under a fixed
> trajectory budget?

Execution is decentralized and unchanged.  Only centralized training uses
the sketch.  Wall-clock remains secondary; primary accounting is environment
transitions, actor updates and critic computation.

## Sealed, fully charged split

Each birth snapshot generates three conditionally independent reset splits
`S1,S2,U`.  The two sensor splits cross-fit the drift coefficients; `U` is the
only split used for the applied actor/critic direction.  The controller commits
`(alpha,beta)` after observing `S1,S2` and the version path but before opening
`U`.  Hence the action is predictable with respect to the update innovation.
Every trajectory in all three splits is charged, and an equal-cost baseline
uses the entire same budget for its update.

The true conditional upper drift has only five coefficients,

\[
 D_k(u)=h_k^\top u+\tfrac12u^\top Q_ku,
 \quad u=(\alpha,\beta),\quad Q_k\succeq0.             \tag{1}
\]

They encode actor descent, critic correction, actor curvature/variance,
moving-target cross curvature and critic curvature/variance.  Known
version-path motion contributes deterministic surcharges.  Estimating these
five scalars is statistically different from certifying every coordinate of a
neural gradient, but the reduction itself is not claimed as novelty.

## Robust two-dimensional action

Suppose cross-fitting produces `h_hat,Q_hat` with

\[
 |\widehat h-h|\le e_h,\qquad
 \|\widehat Q-Q\|_{\rm op}\le e_Q.                    \tag{2}
\]

For nonnegative actions,

\[
 \overline D_k(u)=(\widehat h+e_h)^\top u
 +\tfrac12u^\top(\widehat Q+e_QI)u                    \tag{3}
\]

upper-bounds (1).  The action is the exact box minimizer of (3).  Since zero
is feasible, the certified drift is nonpositive on the declared coefficient
event.  For any fixed comparator `v` in the same box,

\[
 D_k(u_k)-D_k(v)
 \le 2e_h^\top v+e_Q\|v\|^2.                          \tag{4}
\]

The online control cost is a constant-size QP after the required dot products.
It uses no Hessian inverse, meta-gradient unroll or scan over learning rates.

An expectation-only variant may replace simultaneous confidence by a
mean-square coefficient error and obtain decision regret of order `1/m` when
`Q_k` is uniformly positive definite.  It must be presented as expected
finite-time convergence—not per-packet no-harm.

## Novelty boundary

Adaptive stepsizes, meta-gradient actor--critic, asynchronous actor--critic,
and composite Lyapunov analysis all pre-exist.  The only defensible novelty is
their problem-specific intersection: owner-self-fresh but teammate/critic-
stale training packets, an actor-created moving centralized target, sealed
charged drift sensing, and one online joint action controlling progress and
repair.  A paper titled or framed merely as adaptive actor/critic learning
rates would not clear the novelty bar.

## Final CPU upper-bound gate

Before deriving a full Markov coefficient estimator, grant the controller an
ideal unbiased five-scalar sketch and charge its sensor share against update
variance.  The gate must be frozen before outcomes.  It must require:

1. at least 10% AUC gain over the per-scenario best fixed pair;
2. at least 3% AUC gain over the privileged exact online-diagonal rule and
   directionality in at least 60% of primary cells;
3. recovery of at least 60% of the exact coupled-versus-diagonal headroom;
4. target-sensitivity-zero coupling gain below 1%;
5. the coefficient-estimation regret term below 25% of oracle drift in at
   least 70% of favorable cells;
6. complete split charging, constant control complexity and byte-exact
   reproduction.

Failure of the equal-cost diagonal or headroom-capture gate stops the entire
coupled-timescale idea as the ICML mainline.  Passing only authorizes the
harder construction of observable Markov coefficient estimators; it is not
paper evidence and does not authorize GPU work.
