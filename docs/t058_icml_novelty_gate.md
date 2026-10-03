# T-058 ICML novelty and evidence gate

## Decision

The core research line passes a conditional ICML novelty gate. It is not yet
submission-ready. T-057A supplies strong formal evidence for the registered
fixed-policy Markov-TD scope, and T-056 closes its finite-budget theorem loop.
The remaining claim-critical empirical gap is external validity under a
standard nonlinear value learner; the remaining presentation gap is a
contribution-led theorem architecture and fresh full bibliography audit.

The recommended title is:

> Beyond Linear Speedup: Correlation-Adaptive Participation for Delayed
> Multi-Agent Markov Learning

“Mixing-adaptive” is removed from the title because T-017 proved that uniform
unknown-mixing adaptation is impossible without separation or an independent
certificate. “MARL” should not appear alone in the title because the proved
and formal scope is cooperative multi-agent policy evaluation / stochastic
approximation, not a general Markov game or actor--critic result.

## Defensible contribution stack

1. **Participation phase beyond independent-agent speedup.** Under a charged
   per-round overhead, cross-agent trajectory correlation changes the
   stationary PR coefficient to
   `(h+q)[rho+(1-rho)/q]`, creating an interior optimal participation
   `sqrt(h(1-rho)/rho)` rather than universal “use all agents” speedup.
2. **Observable low-complexity dependence probe.** Equality of short public
   state-path fingerprints has expectation `c+(1-c)rho`. With two probe
   actors the match count is exactly Binomial, so the action law is finite and
   computable without covariance estimation, Hessians, or preconditioning.
3. **End-to-end finite-budget certificate.** Conditioning T-048's exact
   delayed PR risk on the independent fingerprint count charges both budgets,
   probe opportunity cost, integer candidate horizons, Markov lag covariance,
   transient bias, and delay. The stationary phase is its limit rather than a
   substituted empirical surrogate.
4. **Formal standard-kernel evidence.** T-057A uses 256 isolated seed clusters,
   exact Gymnasium transition/reward outcomes, strong no-probe fixed-q
   comparators, 50,000 paired cluster-bootstrap replicates, and byte-level
   reproduction. Aggregate risk falls 12.76%; every task and delay passes
   simultaneous upper-bound gates.

## Closest-work confrontation

| Work | What it already covers | Boundary relative to this project |
|---|---|---|
| Khodadadian et al., ICML 2022 | Federated on/off-policy TD and Q-learning with Markovian sampling and linear agent speedup | Does not formulate or identify a cross-agent correlation-dependent optimal q under charged participation |
| Salgia and Chi, NeurIPS 2024 | Order-optimal sample--communication trade-off for federated Q-learning | Optimizes intermittent communication complexity, not instance-adaptive participation from observed cross-agent dependence |
| Adibi et al., AISTATS 2024 | Tight finite-time effects of Markov mixing and delays; delay-adaptive SA | Adapts to delay, not cross-agent correlation or participation count |
| Dal Fabbro et al., CDC 2024 (DASA) | Multi-agent SA with average-delay dependence and N-fold speedup | Explicitly assumes independent agent Markov chains; the current phase begins where that speedup degrades under dependence |
| Mou et al., COLT 2022 | Instance-optimal Markov linear SA and PR averaging | Single-trajectory statistical efficiency; no participation/probe resource decision |
| Sun et al., ICLR 2025 | Correlated client *availability* over rounds and debiasing | Correlation concerns participation timing/bias, not observation-trajectory redundancy and optimal simultaneous q |
| Cummins et al., L4DC 2025 | Feedback control of FL client participation | ADMM/federated optimization controller based on optimization dynamics, not fingerprinted Markov dependence |
| Huynh et al., NeurIPS 2025 | Streaming FL with Markovian client data | Temporal Markov dependence and client speedup, not an observed cross-client correlation phase |
| Lan et al., NeurIPS 2023 | Communication-efficient federated natural policy gradient | Reduces second-order message dimension with ADMM; no adaptive number of correlated agents |

No checked source jointly contains the phase law, observable fingerprint,
fully charged classify-and-commit rule, delay, and exact finite-risk
composition. This is a scoped novelty conclusion, not a universal priority
claim. A fresh citation search remains mandatory immediately before the ICML
submission.

## Why this can be ICML-level

The result is more than a controller beating a baseline. It identifies a
failure mode of the canonical linear-speedup intuition, derives the correct
resource-dependent phase, shows how to observe the latent dependence at low
cost, and proves exactly when the information cost pays for itself. The
formal experiment then tests that causal chain rather than only end reward.

The current formal evidence is unusually clean but experimentally narrow:
FrozenLake, CliffWalking, and Taxi are standard yet small fixed-policy
kernels, and the registered dependence is trajectory-switch coupling. An
ICML reviewer can reasonably ask whether the phase survives nonlinear
features, stochastic optimization, and a richer observation space. Therefore
one nonlinear benchmark family is a submission gate, not optional decoration.

## Authorized external-validity program

T-059 may design a CPU-first benchmark using the official MinAtar interface,
a fixed public behavior policy, and a small nonlinear value network. Candidate
games are Asterix, Breakout, and Seaquest. Agent marginals must remain
unchanged under the same common/private trajectory-switch coupling. The scan
must first run fixed q in `{1,4,16}` only and demonstrate, after all resource
costs, an oracle ceiling of at least 5% aggregate gain with at least 60% of
cells directionally improved. Controller data may not be generated unless
that outcome-free/fixed-q gate is separately preregistered and passes.

If the fixed-q value gate passes, a new controller pilot may be run. A GPU is
likely useful for the eventual multi-seed neural benchmark, but is not needed
for environment installation, interface tests, fixed-policy state statistics,
or a small CPU feasibility scan. If the gate fails, no benchmark hopping is
allowed; the paper remains a theory-plus-linear-policy-evaluation submission
or changes venue rather than cherry-picking tasks.

