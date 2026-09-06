# Strategic-clock equalization for asynchronous MARL

Date: 2026-09-06

Status: **new problem-level candidate; no efficacy claim and no GPU
authorization.**  This document does not reopen any stopped participation,
paid-sensing, collaboration-graph, perishable-update, or actor--critic drift
controller.

## The one scientific question

In event-driven CTDE, different actor blocks become trainable at random and
time-varying rates.  A returned block is not an interchangeable worker update:
it changes one strategic component of the same joint policy.  Consequently,
the arrival process multiplies the game-gradient field by a time-varying
diagonal clock.  Even when every gradient is fresh and unbiased, this changes
the joint policy path and can change the basin or equilibrium reached.

The candidate asks:

> Can an online Lyapunov controller make asynchronous actor updates follow a
> common *strategic time*, preserving the learning path and policy quality of
> a synchronous reference under unknown Markov-modulated availability?

The primary target is policy quality and learning-curve area at matched
environment transitions and arrived actor batches.  Wall-clock time is a
secondary systems axis.  Training is centralized and event driven; execution
remains decentralized and adds no communication.

This is not an equilibrium-selection algorithm.  It does not introduce an
external welfare preference or opponent-aware meta-update.  Its target path is
the declared synchronous optimizer path; the problem is to prevent hardware
or software availability from silently selecting a different learning
dynamics.

## Why this is not an old route under a new name

- It does not choose participation count `q`, a collaboration graph, a probe,
  a correction mask, or an optimism call.
- It does not estimate hidden correlation or game geometry.
- It does not schedule independent compatible jobs for throughput.
- It does not use a separately paid sensor.  Actor identity, arrival time,
  update count, policy version and the ordinary actor gradient are already
  present in every asynchronous implementation.
- The action is update *mass* on an arriving strategic policy block.  The
  intended benefit is path fidelity and return, not merely faster service.

## Exact model

Let `theta=(theta_1,...,theta_n)` be a factorized joint policy and let
`F(theta)` be the joint policy-gradient field.  At event `k`, actor `I_k`
becomes available and supplies a Markovian stochastic block gradient

\[
 \widehat F_{I_k,k}=F_{I_k}(\theta_{k-d_k})+\xi_{I_k,k}.
\]

The executed action is a scalar update mass `u_k` and a common strategic-clock
increment `c_k`:

\[
 \theta_{I_k,k+1}=\theta_{I_k,k}+u_k\widehat F_{I_k,k},
 \qquad 0\le u_k\le \bar u_{I_k},\quad 0\le c_k\le\bar c.
\]

Define each actor's served strategic time and debt by

\[
 \tau_{i,k}=\sum_{r<k}{\bf1}\{I_r=i\}u_r,
 \qquad Q_{i,k}=s_k-\tau_{i,k},
 \qquad s_{k+1}=s_k+c_k.
\]

Thus

\[
 Q_{i,k+1}=Q_{i,k}+c_k-{\bf1}\{I_k=i\}u_k.
\]

The control Lyapunov function is

\[
 L_k={1\over2}\sum_i Q_{i,k}^2
      +V\,\mathcal E(\theta_k,\bar\theta(s_k)),
\]

where `bar theta(s)` is the synchronous reference interpolation and
`mathcal E` is a theorem-facing path or potential error.  A one-event
smoothness bound has the form

\[
 \Delta L_k
 \le B_k+\sum_i Q_{i,k}c_k-Q_{I_k,k}u_k
 +V\{a_k u_k^2-b_k u_k+r_k(d_k,\tau_{mix})\}.
\]

The online action minimizes this same upper drift.  With fixed `c_k`, it is
the clipped scalar root

\[
 u_k^*=\Pi_{[0,\bar u_{I_k}]}
       {Q_{I_k,k}+Vb_k\over 2Va_k},
\]

and allowing `c_k` produces a constant-size two-variable convex QP.  This is
the nondecorative use of Lyapunov theory: it chooses the executed update mass
and the speed of the reference clock.  The reference clock slows when the
arrival process makes equal strategic service infeasible, rather than allowing
unbounded debt or unstable rare-agent jumps.

The exact coefficients are not assumed observable in a neural implementation.
The first theorem must replace them by bounds made from the ordinary gradient,
declared trust-region curvature, delay and Markov-mixing envelopes.  If this
replacement needs a new rollout or a latent error state, the candidate fails.

## Minimal separation object

A one-state two-agent cooperative Markov game already exposes the issue.  With
shared reward matrix

\[
 R=\begin{bmatrix}r_L&r_M\\r_M&r_H\end{bmatrix},
 \qquad r_H>r_L>r_M,
\]

and Bernoulli policies `x_i=sigma(theta_i)`, the exact logit gradients are

\[
 F_1=x_1(1-x_1)
 \big[(r_M-r_L)(1-x_2)+(r_H-r_M)x_2\big],
\]

with the symmetric expression for `F_2`.  The interior saddle is the basin
boundary.  If the two initial policies lie on opposite sides, transient
over-service of one block can decide which coordination equilibrium is
reached even when both actors receive the same total number of arrivals.

The required separation theorem is pathwise, not a favorable numerical
example: construct two equal-count availability paths with opposite early
bursts such that no single fixed block scaling tracks the synchronous path on
both, whereas a feasible causal debt policy has uniformly bounded strategic
clock discrepancy.  A Markov-potential-game theorem must then connect this
clock discrepancy to potential/Nash gap under Markov gradient noise and
bounded delay.

## Theory obligations

1. **Clock-distortion lemma.**  Show that uncorrected random block arrivals
   induce a rate-distorted game field and identify when the distortion changes
   the path rather than only its parametrization.
2. **Queue-feasibility theorem.**  State the necessary service condition and
   prove a finite-horizon debt bound for the same `(u_k,c_k)` rule used in
   code.  Infeasible rare-agent rates must be reported, not hidden by an
   unbounded step.
3. **Path-shadowing theorem.**  Bound asynchronous-to-synchronous trajectory
   error under smooth block gradients, Markov noise and bounded policy delay.
4. **Learning theorem.**  For a declared Markov potential-game class, convert
   path error into a finite-time potential/Nash-gap result.  A general neural
   global-convergence claim is out of scope.
5. **Separation/lower bound.**  Prove that a clock-oblivious fixed rule has
   nonzero worst-case path or basin error over a declared pair of
   indistinguishable equal-count availability paths.

An SDDE is optional and secondary.  It is admissible only if the small-step
limit with random delay yields a quantitative phase boundary not already in
the discrete event theorem.  The discrete stochastic-approximation proof is
the main guarantee.

## Pre-algorithm kill gates

No sampled learning experiment or algorithm name is authorized until both
gates pass.

### Gate A: theorem interface

The five obligations above must be proved for an exact one-state game and a
smooth Markov-potential-game upper class without using latent state, future
arrivals, extra trajectories, or an unbounded update mass.

### Gate B: equal-resource oracle headroom

On the frozen analytic population in
`strategic_clock_oracle_gate_manifest_20260906.json`, an ideal causal
drift-minimizing clock controller must be compared with:

- raw asynchronous block PG;
- barrier/fresh sequential PG;
- true stationary-rate inverse scaling;
- EWMA rate correction;
- count-only debt equalization;
- the per-population best fixed block scaling; and
- the synchronous reference.

All methods receive the identical arrival paths and gradient observations.
The primary quantity is recovery of synchronous normalized learning AUC at
matched arrivals; final value and wall-clock are secondary.  The candidate
survives only if it recovers at least 50% of the raw-to-synchronous AUC loss,
beats the strongest causal non-Lyapunov baseline by at least 10% of that loss
in median, and is directionally better in at least 60% of headroom-active
cells.  Homogeneous and single-equilibrium controls must have at most 1%
unnecessary loss.

Failure of either gate closes this problem.  It may not be rescued by dropping
the best baseline, conditioning the population on observed wins, or making
wall-clock the only metric.

## Standard-benchmark contract if both gates pass

The eventual experiment would modify a maintained CTDE implementation, not
invent a bespoke benchmark.  Candidate families are SMACv2/Multi-Agent
MuJoCo plus one mixed-coordination environment.  Agent optimizer availability
would be replayed from preregistered stationary and Markov-bursty traces.  The
same rollout batches, total environment transitions and arrived actor batches
would be shared across methods.  Main plots would report return/AUC versus
environment transitions; accepted updates, gradient compute and wall-clock
would be separate resource axes.

GPU/HPC4 is not authorized by this scope document.

## Bounded novelty boundary

- Chasnov et al. show that nonuniform learning rates distort continuous-game
  vector fields and can alter convergence paths.  They do not give a
  Markov-modulated, path-faithful CTDE controller:
  https://proceedings.mlr.press/v115/chasnov20a.html
- A2PO studies sequential agent updates, monotonic improvement and update
  order, but not unknown event clocks or Lyapunov path tracking:
  https://arxiv.org/abs/2302.06205
- Multi-timescale cooperative MARL deliberately assigns different learning
  rates to mitigate nonstationarity; it does not cancel exogenous transient
  clock distortion or recover a synchronous reference path:
  https://proceedings.mlr.press/v232/nekoei23a.html
- AFedPG corrects delayed workers training one shared policy.  Its workers are
  not strategic policy blocks in a joint game:
  https://arxiv.org/abs/2404.08003
- Recent equilibrium-selection work targets preferred equilibria through
  learning-rule or opponent-aware corrections.  Strategic-clock equalization
  instead removes an exogenous optimizer-induced selection effect:
  https://arxiv.org/abs/2406.08844 and
  https://arxiv.org/abs/2605.18078

The novelty is therefore conditional, not declared: a paper exists only if
the same low-complexity causal controller closes the path-shadowing theorem and
the strong-baseline headroom gate.

## Devil's-advocate verdict

**One bounded attempt is justified; ICML readiness is not established.**  The
strongest reviewer objection is that this is inverse-frequency coordinate
descent with a queue.  The response cannot be rhetoric.  It must be a theorem
showing strategic path/basin distortion that does not exist in the relevant
single-objective convex reduction, plus positive matched-resource evidence
against true-rate, EWMA, count-debt and barrier baselines.  If count-only debt
already closes the gap, the Lyapunov performance term has no value and the
candidate stops.
