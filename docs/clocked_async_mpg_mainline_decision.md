# From Iterations to Wall-Clock Time: finite-time asynchronous learning in Markov potential games

Status: **theorem-first replacement candidate; no efficacy experiment or GPU
work is authorized.**

## Paper-level decision

The project must stop searching for another online scheduler that narrowly
beats a tuned static rule.  The frozen evidence now rejects perishable-update
backpressure, unconditional cross-agent Hessian transport, policy-inventory
control and compatible-set MaxWeight as positive mainlines.  Their correct
lemmas remain reusable, but none supplies the missing paper-level performance
claim.

The only defensible replacement inside the requested scope is a finite-time
theory of **truly event-driven policy learning by distinct agents in one Markov
game**.  The contribution would be a wall-clock convergence theorem and a
matching phase boundary, not a new participation count, queue scheduler or
heuristic clipping rule.

Provisional title:

> **From Iterations to Wall-Clock Time: Finite-Time Asynchronous Independent
> Policy Learning under Markovian Sampling**

This candidate is not yet an ICML claim.  It survives only if the delayed
Markov-game proof is genuinely stronger than composing existing asynchronous
stochastic-approximation and synchronous Markov-potential-game theorems.

## One coherent question

When heterogeneous agents in a common Markov game finish policy-gradient and
critic work at different random times, can they learn without a global barrier
and retain a finite-time convergence guarantee whose **elapsed-time** rate
reflects useful aggregate compute rather than the slowest agent?

The tension is intrinsic.  Waiting for every actor gives fresh joint-policy
data but pays the straggler clock.  Applying updates on arrival uses aggregate
compute but evaluates an agent's gradient under old teammate policies and
correlated Markov data.  An iteration count hides both effects.  The intended
result makes the tradeoff explicit in one bound.

This is centralized training with distinct actor blocks and a centralized
critic in the practical implementation; execution remains decentralized and
adds no communication.  The theorem may first use tabular policies and exact
or linear critics, but it must retain distinct agent policies and a joint
Markov trajectory.  Federated workers estimating one shared policy are not the
model.

## Mathematical interface

Let `theta=(theta_1,...,theta_n)` parameterize a factorized joint policy in a
discounted common-payoff Markov potential game with potential `Phi(theta)`.
At event `k`, actor `I_k` returns an estimate computed from a trajectory born at
joint-policy version `b_k <= k`.  The server applies one block update

\[
\theta_{I_k}^{k+1}
=\Pi_{I_k}\!\left(\theta_{I_k}^k
+\alpha_{I_k,N_{I_k}(k)}\widehat g_{I_k,k}\right),
\]

and leaves the other blocks unchanged.  `N_i(k)` is the local update count, not
a synchronized round number.

The multi-agent staleness object is not scalar age.  If `L_ij` controls how a
change in actor `j` changes actor `i`'s block gradient, define the realized
interaction-weighted drift

\[
\mathsf D_k^2
=\sum_j L_{I_kj}^2
  \|\theta_j^k-\theta_j^{b_k}\|^2.
\]

This quantity is zero for teammate blocks that do not affect the arriving
gradient and can be large for several recent high-coupling updates even when
timestamp age is small.  It is an analysis quantity until the source and
estimability of `L_ij` are stated; visual distance or a learned attention map
is not automatically a valid interaction certificate.

The proof candidate is a discrete Lyapunov--Krasovskii functional

\[
\mathcal V_k
=\Phi^\star-\Phi(\theta^k)
+a\,\|e_k^{\rm critic}\|^2
+\sum_{r<k}c_{k,r}\|\theta^{r+1}-\theta^r\|^2,
\]

where the final term stores precisely the still-relevant update history.  Its
one-event conditional drift must pay for stale teammate policies and Markov
critic bias.  Writing this functional is not the theorem: the coefficients
`a,c_(k,r)`, filtration, delayed-data conditioning and telescoping inequality
must all be explicit.

SDDE is optional.  A stochastic delay differential equation may be used only
after the discrete event-time theorem, for a diffusion/continuous-time limit
that produces an additional stability or phase-boundary result.  It cannot
replace the finite-time discrete proof.

## Required theorem chain

All five items must hold for the same executable update rule.

1. **Delayed Markov-gradient decomposition.**  Decompose the arriving block
   estimator into the current block gradient, an explicitly bounded teammate
   drift term, a Markov mixing bias and a martingale term without treating
   overlapping trajectories as independent.
2. **One-event Lyapunov drift.**  Prove a negative conditional drift outside a
   stated error floor.  The history functional must absorb cross-block delay
   terms rather than replace them by an unexplained maximum-delay constant.
3. **Finite-time game guarantee.**  Convert the potential/stationarity bound
   to an average Nash-gap or another standard Markov-potential-game criterion
   under explicit distribution-mismatch and policy-regularity assumptions.
4. **Wall-clock conversion.**  Under a declared renewal or Markov completion
   model, convert event count to elapsed time.  The bound must expose both the
   aggregate completion rate and the interaction-weighted delay penalty.
5. **Lower bound or separation.**  Give a family showing when asynchronous
   learning improves the slowest-agent barrier and when interaction delay
   destroys that speedup.  A comparison only against a deliberately bad raw
   asynchronous implementation is insufficient.

The desired theorem is finite-time and instance-sensitive.  Merely proving
almost-sure convergence under bounded delay, or substituting generic delayed
SA constants into an existing MPG proof, fails the novelty gate.

## Low-complexity algorithm boundary

The default algorithm is ordinary event-driven block policy learning with
local-clock step-size sequences.  It uses no Hessian, covariance inverse,
preconditioner, agent-count scan or global scheduling optimization.  Any
optional delay normalization must be a scalar, derived from the same drift
proof and charged in experiments.  A new heuristic trust-region rule is not a
contribution: AFedPG, delayed Markov SA, MA-Trace and recent staleness-adaptive
trust regions already occupy that space.

## Novelty boundary

The candidate must be distinguished from the following primary lines.

- AFedPG gives sample and wall-clock complexity for asynchronous federated
  workers updating one global policy; its agents are not distinct strategic
  policy blocks in one Markov game.
- Delayed stochastic approximation under Markovian sampling already gives
  finite-time bounds and an average-delay adaptive scheme for contractive SA.
- Asynchronous decentralized actor--critic in Markov potential/near-potential
  games already gives asymptotic convergence with asynchronous step sizes.
- Asynchronous gradient play already handles delayed feedback in zero-sum
  polymatrix games.
- MA-Trace already corrects multi-worker policy lag in practical distributed
  MARL.
- HAPPO/HAML, A2PO and B2MAPO already cover sequential ordering and batched
  policy updates with monotonic-improvement arguments.

Consequently the defensible novelty is the *finite-time wall-clock theorem for
distinct policy blocks under joint Markov data and stale teammate policies*,
including a nontrivial delay--coupling phase boundary.  If that proof reduces
to any item above after notation changes, stop the candidate.

## Experimental ladder

No efficacy run is authorized yet.  If the first four theorem obligations
close, use the following ladder.

1. **Exact CPU theory confirmation.**  Tabular Markov potential games with
   exact values and declared heterogeneous completion clocks.  Verify every
   term of the Lyapunov recursion and the predicted wall-clock phase boundary.
2. **Independent CPU stochastic confirmation.**  Fresh seeds, continuing
   Markov trajectories and measured estimator bias.  Compare synchronous
   independent policy learning, raw asynchronous learning, generic
   average-delay SA, and the theorem-facing event-driven rule.  Report Nash
   gap/potential gap versus both samples and wall-clock.
3. **Standard MARL benchmark.**  Only after stages 1--2 pass: separate actors
   under CTDE on at least two accepted benchmark families, with real or
   faithfully emulated heterogeneous rollout/optimization times.  Include
   HAPPO/HARL, MAPPO, MA-Trace where applicable, a barrier implementation and
   a strong asynchronous lag-correction baseline.  Report final return,
   return-versus-wall-clock, environment transitions, update count, policy lag,
   utilization and overhead.  This stage will likely require GPU/HPC4.

The main-text empirical claim must be positive across multiple tasks and delay
regimes.  A speedup obtained only by assigning the baseline fewer transitions
or weaker hardware is invalid.

## Immediate kill gates

Before any new scientific trajectory:

1. derive the delayed block-gradient decomposition with a valid filtration;
2. show that the Lyapunov history term closes without a hidden bounded-gradient
   or independence assumption incompatible with policy gradients;
3. establish a finite-time bound not obtainable as a direct corollary of
   delayed Markov SA plus synchronous MPG convergence;
4. exhibit an analytic family with a nonempty async-speedup region after full
   compute and sample accounting.

Failure of any item stops this theorem route.  Existing T-083A and all later
negative results remain frozen and cannot be used to tune the new population.

## Current assessment

- Coherence: high; one question, one event-driven learner, one Lyapunov
  functional, one wall-clock theorem and one matching experimental axis.
- Theoretical feasibility: plausible but unproved; the critic/mixing and
  delayed block coupling are the hard steps.
- Novelty: plausible only at the complete theorem level, not at the level of
  delay-adaptive learning rates.
- Experimental success probability: moderate for wall-clock speedup versus a
  barrier under real heterogeneity; unknown versus strong async correction.
- ICML readiness: **not ready**.  No acceptance probability is asserted, and
  no paper should be drafted as if the theorem already exists.

