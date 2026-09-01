# Two Clocks, One Game: the consolidated ICML mainline

Date: 2026-09-02.

Status: consolidated theory-and-evidence programme.  This document does not
turn previous development experiments into formal evidence and does not
authorize GPU work.

## The one question

In centralized training with decentralized execution, distinct agents in one
Markov game finish fixed-cost policy-gradient packets at heterogeneous wall
clocks.  A returned packet advances its owner's policy block, but it was
sampled before some teammate updates and is therefore strategically stale.

The paper asks:

> When does barrier-free independent policy learning provide a genuine
> wall-clock speedup, and when must interaction-induced staleness erase that
> speedup?

The answer is a **rate--coupling phase**, not another participation controller.
The favorable force is heterogeneous aggregate packet completion; the adverse
force is the interaction-weighted path travelled by teammate policies while a
packet is in flight.  The slowest strategically essential agent remains an
unavoidable bottleneck.  This makes the claim meaningful rather than a generic
"linear speedup from more workers" statement.

## Why there are two clocks

The actor clock records physical packet completion.  Agent `i` has service
rate or bounded service interval `(lambda_i, [a_i,b_i])`.  It determines which
policy block can be updated and how much useful computation has completed by
wall-clock time `T`.

The learner clock records joint-policy versions.  If agent `i`'s packet was
born at version `b_k` and returns at learner event `k`, its relevant delay is
not the timestamp `k-b_k` alone, but

\[
 \mathsf D_k^2=
 \sum_{j\ne i}L_{ij}^2
 \|\theta_j^k-\theta_j^{b_k}\|^2 .
\]

The off-diagonal constants `L_ij` express a genuinely multi-agent effect: how
agent `j`'s policy change rotates agent `i`'s gradient.  Single-flight
ownership makes the packet self-fresh in its own block.  Shared-policy
federated workers do not have this structure.

## One executable learning rule

Each agent owns one policy block and has at most one fixed-horizon packet in
flight.  A completed packet is applied immediately with a common certified
step `alpha`; the owner then launches its next packet from the new joint-policy
version.  The online update is ordinary `O(dim(theta_i))` block policy
gradient.  There is no Hessian, covariance inverse, preconditioner, agent-count
scan, online QP, paid sensor or offline learning-rate catalogue.

The largest theorem-facing common step is the minimum positive root of

\[
 L_{jj}\alpha+(1+\delta)D^2u_j\alpha^2\le 1,
 \qquad
 u_j=\max_i\Big(\sum_{\ell\ne i}L_{i\ell}\Big)L_{ij},
\]

over policy blocks `j`.  Computing it once costs `O(n^2)` for a dense declared
interaction envelope and less for a sparse envelope.  A practical neural
implementation may use a public conservative bound; it may not call a sampled
PPO ratio a uniform cross-sensitivity certificate.

## Lyapunov theory is the mechanism

Let `f=Phi_star-Phi` and `Delta^r=theta^(r+1)-theta^r`.  The proof uses the
Lyapunov--Krasovskii functional

\[
 \mathcal V_k=f(\theta^k)
 +\frac{D(1+\delta)}2
 \sum_{d=1}^{D}(D-d+1)
 \sum_j w_j\|\Delta_j^{k-d}\|^2,
 \qquad w_j=\alpha u_j .
\]

The potential gap measures learning error.  The triangular history energy is
exactly the policy motion that can still invalidate an in-flight packet.  Its
one-event shift removes one copy of every old step and adds `D` copies of the
new step.  Under the displayed step condition, the negative history shift
absorbs the cross-agent stale-gradient term.  The same drift inequality then
produces the finite-time stationarity bound; Lyapunov theory is not an
after-the-fact stability metaphor.

For fixed-horizon reset trajectories, the Markov dependence inside each
packet is handled by an exact likelihood-gradient identity, truncation bias
and a bounded packet second moment.  Completion must be conditionally
independent of the packet innovation.  Continuing-chain critics and
trajectory-dependent completion are extensions, not silently covered cases.

## The phase statement

The intended main theorem has three pieces using the same model.

1. **Upper bound.**  The event-time Lyapunov drift yields a finite-time
   full-policy stationarity bound under arbitrary bounded completion order.
   Deterministic service bounds convert it to wall-clock time and expose both
   aggregate completed work and the slow-essential-block coverage window.
2. **Favorable phase.**  Heterogeneous completion removes repeated global
   barriers when interaction-weighted staleness is below the Lyapunov margin.
   A fully utilized frozen-policy barrier cannot turn duplicate packets at one
   policy into additional adaptive query depth.
3. **Unfavorable phase.**  If a returned direction has signal `s` and the
   admissible current-gradient uncertainty is `B`, the best uniformly safe
   progress certificate along that direction is

   \[
   \sup_{\alpha\ge0}
   \inf_{\|g-\widehat g\|\le B}
   \left\{\alpha\langle g,\widehat g\rangle
   -\frac L2\alpha^2\|\widehat g\|^2\right\}
   =\frac{[s-B]_+^2}{2L}.
   \]

   Hence no stale-direction method can promise positive progress when
   interaction debt reaches the signal.  Separately, every full-policy or
   Nash guarantee must wait for each essential agent's packets, giving a
   wall-clock lower bound governed by its local completion rate.

Together these statements explain both sides of the same phenomenon.  The
paper must not claim universal no-harm or that aggregate throughput erases an
essential slow actor.

## Evidence already available

The exact multi-state CPU confirmation passed all ten frozen gates.  In its 12
heterogeneous primary cells, the geometric async/shadow-barrier target-time
ratio was `0.444557`, with all 12 cells faster.  Homogeneous controls include
cells where the barrier is better, as required by the phase claim.

The independent stochastic Markov-packet confirmation passed all twelve gates
and reproduced byte for byte.  Across 64 fresh seeds and 12 heterogeneous
cells, the certified asynchronous learner had wall-clock ratio `0.431353`,
charged-transition ratio `0.429687`, and lower median time in all 12 cells
relative to the fully utilized frozen-policy barrier.  Increasing service
heterogeneity helped; increasing coupling hurt, in every registered
direction.

These are positive, fully charged CPU results for the theorem-facing model.
They do not establish standard neural MARL performance.  The later sensor,
freshness, correction, optimism and geometry branches failed their own frozen
gates and are not components of this mainline.

## What is still required for an ICML submission

1. consolidate the upper bound, safe-progress obstruction, essential-agent
   lower bound and fresh-query barrier separation into one publication-grade
   theorem chain;
2. sharpen the lower result enough to state which dependence on service,
   coupling and delay is matched and which remains only qualitative;
3. define a conservative neural cross-sensitivity interface, or clearly split
   the theorem-facing bound from the practical measured proxy;
4. run standard nonlinear CTDE benchmarks against a fully utilized barrier,
   raw asynchronous learning, a strong delay-adaptive asynchronous baseline,
   HAPPO/HARL and MAPPO, reporting return versus wall-clock, transitions and
   utilization separately;
5. give positive main-text results on more than one task family while retaining
   the predicted high-coupling or homogeneous negative side of the phase;
6. complete a fresh primary-source novelty and citation-integrity audit before
   manuscript delivery.

Items 1--3 are CPU/theory work.  Item 4 will require GPU/HPC4 and a separate
frozen preregistration.  No GPU job is authorized by this document.

## Stop rules

Stop this mainline before GPU if either of the following occurs:

- the lower bound reduces entirely to generic delayed SGD after removing the
  distinct policy blocks and joint Markov trajectory;
- the neural implementation cannot expose a charged, reproducible asynchronous
  advantage over both a fully utilized barrier and a strong asynchronous
  reference without outcome-selected delay injection.

The target contribution is the complete rate--coupling wall-clock theory and
its empirical phase, not any individual step-size formula.
