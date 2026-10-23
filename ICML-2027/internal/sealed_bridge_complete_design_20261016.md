# Complete successor design: sealed-anchor bridge gradient transport

## Material Passport and decision

ARS research/experiment-design, 2026-10-16. Human authorization: change the
stale-data update direction/correction mechanism, preserve all previous results,
and evaluate complete learning against strong controls across MARL environments.
Status: complete native integration PASS (six original+six replay cells;
56.875 CPU seconds) with a substantive algorithm design; NOT positive
MARL evidence, a qualified ICML contribution, or a promise of success.
All old sources, outcomes, gates and TSP manuscripts remain unchanged.

## Research question and complete mechanism

Can a delayed, independent gradient cohort act as a useful control variate for a
CURRENT joint-policy gradient, after paying for every source episode, current
bridge episode, dual evaluation/backward, copy and communication payload?
The question is not whether a noisy maximum over old PPO parameter rays wins.

Maintain distinct decentralized policies pi_theta^i and a central server.
A source worker receives an immutable complete joint snapshot mu (all local
actor versions), an immutable state-value baseline, an episode count N>=2,
and a unique source ordinal. It collects N independent complete episodes and
forms per-episode Monte Carlo policy gradients h_mu. It imports NO optimizer
state. The cohort gradient, returns, states and diagnostic norms stay sealed
until its one assigned receipt; service delay is drawn from a fixed stream
independent of selection, reward and update count. Source parameters, baseline,
shape/charge metadata and ordinal are visible. No actor/critic update may use
these cohort episodes before that receipt. Independent source cohorts can share
mu, but a gradient cohort is never adaptively reused.

At receipt, before the current source gradient or fresh data is opened, freeze
theta, a current state-value baseline, and an option (m,beta), m>=2, beta in [0,1].
Draw m independent bridge EPISODES: one Bernoulli coin per episode selects the
complete current joint policy pi_theta or complete source joint policy mu, each
with probability 1/2. All agents retain that selected policy for the episode.
This is NOT independent per-agent or per-step policy mixing.

Let p_theta(tau), p_mu(tau) be trajectory laws, and
q(tau)=(p_theta(tau)+p_mu(tau))/2. Environment/initial-state factors cancel in
the ratio. Compute joint trajectory log likelihoods over ALL agents and times
with log-sum-exp. Define w_theta=p_theta/q and w_mu=p_mu/q, both <=2.
Compute paired, genuine episode policy-gradient vectors
f=w_theta*h_theta and c=w_mu*h_mu on each current bridge trajectory.
Use Monte Carlo reward-to-go and a data-independent frozen state baseline;
neither stale normalized GAE nor clipped PPO surrogate gradients define h.
Importance weights are detached coefficients, not extra autodiff paths.

Open the assigned source cohort once, A=mean(h_mu_source), and commit

    d = mean(f) + beta*(A-mean(c)); theta_next = theta + alpha*d.

This changes the actual update vector, not merely its length. The server owns
the transaction. The first proved version uses SGD with a predictable scalar
alpha; ordinary Adam/momentum, gradient clipping, PPO epochs and repeated-data
updates are not silently covered. A source cohort is consumed even if beta=0.
Discarding its cost or using its norm to select beta would invalidate fairness
or the conditional mean interface. Train the baseline only AFTER all gradient
roles of that cohort/bridge finish; subsequent critic fits cannot affect the
already committed gradient.

## Why this addresses the observed failure mechanism

- No same-bank return maximization and no winner's-curse action selection.
- The old gradient is not applied as a supposed current gradient: the paired
  current bridge subtracts its source-distribution expectation.
- Joint likelihoods include teammate drift and temporal occupancy. No local
  importance ratio is substituted for a joint trajectory law.
- Defensive weights are bounded by construction even for continuous policies;
  arbitrary delayed off-policy packets are NOT retroactively declared bridge data.
- The current target is frozen during the bridge. Changing it mid-episode would
  destroy the claimed weights. Critic-fitting/evaluation leakage is prohibited.
- One-use sealed cohorts explicitly avoid the adaptive cached-anchor bias. This
  does mean old-source sampling is paid per cohort; there is no free amortization.

## Conditional gradient and finite-time proof

Fix the receipt filtration F containing theta, mu, both baselines, service/shape
metadata, beta and m, but not the assigned source cohort or bridge episodes.
Assume (a) independent episodes with the same initial-state/transition law;
(b) source and bridge cohorts are conditionally independent given F; (c)
differentiation under the trajectory integral is justified and h is integrable;
(d) J(theta) is L-smooth, bounded above by Jmax, and the needed second/fourth
gradient moments are finite. Complete histories replace states if policies
are recurrent. The current implementation is feedforward.

E[f|F]=grad J(theta), E[c|F]=grad J(mu), E[A|F]=grad J(mu). Hence
E[d|F]=grad J(theta). There is no clipping bias or anchor-error floor in THIS
one-use interface. Adaptive reuse, selection based on the opened anchor,
partial joint logs or evaluation-driven choices do not satisfy it.

With sigma_f^2=tr Cov(f), sigma_c^2=tr Cov(c), kappa=tr Cov(f,c), and
sigma_A^2=tr Cov(h_mu_source), the exact conditional variance is

    r(m,beta) = (sigma_f^2-2 beta kappa+beta^2 sigma_c^2)/m
                + beta^2 sigma_A^2/N.

For fixed m, its minimizer in [0,1] is
beta*=clip(kappa/(sigma_c^2+m*sigma_A^2/N),0,1), with beta=0 when the denominator
is zero. Unknown population coefficients are NOT replaced by the current
sample and then called predictable. Past-only online optimization is below.
For alpha in (0,1/L], smoothness and the conditional mean give

    E[J(theta_next)-J(theta)|F]
      >= alpha*(1-L*alpha/2)*||grad J(theta)||^2 - L*alpha^2*r/2.

For K receipts at fixed alpha, telescoping proves

    (1/K) sum E||grad J(theta_k)||^2
      <= (Jmax-J(theta_0))/(K*alpha*(1-L*alpha/2))
         + L*alpha*sum E[r_k]/(2*K*(1-L*alpha/2)).

This is own-objective expected stationarity, NOT superiority to another HAPPO
trajectory, monotonic improvement of each realized update, global optimality,
or a high-probability no-harm certificate. At an adaptively budget-stopped
horizon, retain sums/indicators and their expectations; do not divide inside
expectations by an arbitrary random K.

## Lyapunov, learnable risk and prospective resource actions

Catalogue all (m,beta), not return-ranked policy candidates. From paired bridge
episode vectors and independent source episode vectors, the unbiased sample
trace variances provide feedback for EVERY catalogue option:

    rhat_j = sampleVar(f-beta_j*c)/m_j
             + beta_j^2*sampleVar(h_mu_source)/N.

Observed m may differ from counterfactual m_j: only the same-distribution
variance coefficient is extrapolated. No counterfactual policy performance is
inferred. Feedback is used only for future receipt choices. Each option's
resource cost is known before data: source N is a paid common cost, bridge m_j
changes actual native sampling, forward/backward work and transport payloads.
Dual forward/backward feedback for all beta choices is paid even when beta=0.

For normalized costs c_j and targets b, use Q_next=[Q+c_actual-b]_+ and
L=V*(Jmax-J)+||Q||^2/2. The conditional drift is bounded by

    Bq - Q.b - V*a*||grad J||^2
       + E[V*L*alpha^2*r_selected/2 + Q.c_selected | F],
    a=alpha*(1-L*alpha/2).

Projected online gradient descent on the probability simplex chooses an option
BEFORE opening data and uses the all-option losses
ellhat_j=V*L*alpha^2*rhat_j/2 + Q.c_j AFTER the transaction. It has expected
static-comparator regret <= D^2/(2*eta)+(eta/2)sum E||ellhat||^2; finite fourth
moments are explicitly required, no silent loss clipping or bounded-loss claim.
Comparator losses are on the algorithm's visited policies/source laws, not
another algorithm's training trajectory. Under an explicit catalogue Slater
option c_j<=b-zeta and bounded its risk, summing drift plus this regret yields
the corresponding finite-time gradient-norm/queue-sum inequality. A practical
queue stability/rate statement additionally needs constants/scalings satisfying
that bound; the one-step drift alone is not relabelled a stability theorem.

Hard native-budget reservation is separate: reserve source work at launch,
bridge work before sampling and charge every in-flight/drained/discarded cohort.
Restrict the catalogue to affordable options before an episode starts. Virtual
queues do not excuse overspending a finite budget.

## Novelty boundary and cost-feasibility decision

Defensive sampling, variance-reduced PG and trajectory control variates are
established technology, including very recent LVR-PG. They are bases/competitors,
not first-of-kind claims. The research candidate is sealed asynchronous source
transport plus predictable risk/resource control with a complete causal and
cost interface. Whether that combination is distinct enough for ICML remains
OPEN, and only real strong-baseline learning can establish practical value.

The source tax N, dual neural work and gradient-vector communication are real.
At theta=mu and ideal identical baselines, the optimal combination merely
recovers ordinary pooling efficiency at equal native sample count; it does not
create an intrinsic strict gain over fresh sampling. A positive learning claim
therefore needs measured gradient-quality/resource gains outside that trivial
limit, and complete return curves against strong fresh AND adaptive controls.
This identity is a design sanity check, not a reason to weaken a gate.

The quantitative necessary screening comparison is the actual per-budget MSE,
not raw variance reduction relative to one tiny current batch. With the same
per-episode baseline convention, fresh PG using N+m current episodes has
variance sigma_theta^2/(N+m). A useful bridge needs its fully priced optimized
r(m,beta) below that benchmark, and must separately satisfy neural/payload cost
constraints. Positive covariance alone is not sufficient. This condition is
not a theorem of return dominance over optimized HAPPO/PPO.

Two alternatives were rejected analytically before any scientific run:
(1) random completion of clipped IS residuals increases variance relative to
the exact estimator and does not save the already required joint forward path;
(2) checkpoint/maximal-coupling trajectory repair requires simulator state access
and cannot beat fresh sampling in native-transition cost by distributional
repair alone when original source transitions are fully charged. Neither is
launched as another numbered performance experiment.

## Prospective validation sequence (not yet an execution registration)

1. Implement the COMPLETE native loop, ledger, source/service clock, one-use
   transaction and all-method analyzer, not only estimator tests.
2. Lock one bounded outcome-free integration qualification and replay before
   native data. Include actual actor/critic networks and continuous-policy path.
   Source/bridge/evaluation seeds must be disjoint, complete episode roles sealed.
3. Separately lock learning development only after integration and static full-cost
   feasibility pass. Freeze task-specific learning horizons BEFORE outcomes;
   do not repeat the prior weak-baseline-learning setup or use tiny MuJoCo runs
   as cross-family performance evidence.
4. Target official MPE Spread/Reference plus official MaMuJoCo HalfCheetah 2x3
   and Walker2d 2x3: four tasks/two families. The local legacy HARL MuJoCo-v2
   path needs mujoco-py, which is absent. Do NOT silently substitute a handmade
   fully observable Gym partition. A separate Gymnasium1.2.0/Robotics1.4.0 runtime
   has now been installed without modifying old environments; official
   HalfCheetah2x3 is included in the registered integration qualification.
   Walker's native early-termination/accounting interface still needs integration
   before scientific task registration; no Walker data or substitution is allowed.
5. Baselines: fresh HAPPO LR grid, MAPPO, matched delayed HAPPO, joint MA-Trace,
   adaptive KL, defensive variance-reduced PG (including the nearest recent
   method when reproducibly implementable). Ablations: no transport, fixed
   variance/batch schedule, no resource prices. Same native budget, all source
   and calibration costs, native/message/byte/wall-clock return axes.
6. Each task must improve >=1% native AUC over the strongest registered control,
   >=2/3 positive paired development seeds, final >=-2%, and baseline learning
   >=2% against same-world frozen initial policies; preserve exact replay and
   <=1.5x full-cost screens. Fresh confirmation and wider evidence follow a
   pass, not a posthoc task/threshold removal. No claim of population robustness
   from a three-seed development screen.

## Integrity and AI assistance

This is AI-assisted internal research design, not manuscript prose. Prior
failures and all frozen evidence remain preserved. New science results do not
exist yet. Source verification and the nearest-method matrix are maintained
separately; no paper bibliography is modified here.

The fresh machine-readable verification is
sealed_bridge_source_verification_20261016.json (eight authoritative arXiv
matches; listed publisher cross-checks; Semantic Scholar unavailable, no
rate-limit workaround). The closest current record is [LVR-PG, October2026]
(https://arxiv.org/abs/2610.13692), which already integrates defensive sampling
with recursive first-order gradients. [SVRPG](https://proceedings.mlr.press/v80/papini18a.html),
[DR-PG](https://proceedings.mlr.press/v119/huang20b.html),
[active IS](https://rlj.org/papers/papini2024policy),
[MA-Trace](https://arxiv.org/abs/2111.11229),
[IMPACT](https://arxiv.org/abs/1912.00167),
[VCPO](https://arxiv.org/abs/2602.17616) and
[GAC](https://arxiv.org/abs/2603.01501) prevent first-variance-reduction,
first-off-policy-correction, first-asynchronous-policy-update and
first-stale-gradient-projection novelty claims. The implementation is a
candidate to test the explicitly stated new causal/resource combination,
not evidence that this novelty test already passed.
