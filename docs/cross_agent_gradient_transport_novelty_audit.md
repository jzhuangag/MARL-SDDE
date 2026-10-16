# Cross-agent gradient transport novelty and feasibility audit

Date: 2026-09-01.
Status: search-bounded internal audit, not a publication novelty claim.

## Decision

The generic ingredients “delay compensation,” “Hessian-vector correction,” “Lyapunov drift,” and “asynchronous RL” are inherited and cannot be claimed as new.
The only defensible candidate contribution is their problem-specific synthesis for distinct policies in one cooperative Markov game:

> transport an arriving agent's block policy gradient through updates made by other policy blocks, certify the Markov/HVP remainder, and use a composite optimization-plus-in-flight-error Lyapunov drift to choose one continuous application step under heterogeneous completion times.

This boundary is potentially ICML-sized only if the same executable method obtains a stronger delay-dependent stationarity/wall-clock result and positive standard-MARL evidence.
Without those two pieces it is an incremental application of delay-compensated asynchronous stochastic gradient descent.

## Primary-source confrontation

| Neighbor | What it already establishes | Remaining distinction that must be proved |
|---|---|---|
| [DC-ASGD, ICML 2017](https://proceedings.mlr.press/v70/zheng17b.html) | Taylor/Hessian approximation for delayed gradients in generic shared-parameter asynchronous SGD | Distinct policy blocks, cross-agent Markov-game externality, uniform policy-gradient/HVP certificate, and Lyapunov value of in-flight rollouts |
| [AFedPG](https://arxiv.org/abs/2404.08003) | Shared global-policy federated PG, delay-adaptive lookahead sampling, sample/time convergence | No single shared policy: teammate updates alter another agent's block gradient; transport is performed after cross-policy changes and coupled to pending-rollout error |
| [Asynchronous Actor-Critic for MARL, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1c153788756d35559c22d105d1182c30-Abstract-Conference.html) | Asynchronous action execution and macro-action actor-critic under decentralized/centralized/CTDE paradigms | The proposed problem is asynchronous *training update completion* with ordinary decentralized execution |
| [Cooperative MARL with asynchronous communication, ICML 2023](https://proceedings.mlr.press/v202/min23a.html) | Communication-triggered cooperative learning with linear function approximation | Does not transport a distinct actor's policy gradient through teammate policy updates |
| [GAC](https://arxiv.org/abs/2603.01501) | Gradient-alignment stabilization for asynchronous RL/LLM training | Shared-policy first-order alignment rather than block-Hessian cross-agent transport and pending-rollout Lyapunov accounting |
| [SAT](https://arxiv.org/abs/2607.18722) | Staleness-adaptive trust-region clipping for asynchronous RL | Sampled policy-mismatch clipping rather than correction of cross-agent gradient externalities |
| [Adaptive asynchronous mini-batching, ICML 2025](https://proceedings.mlr.press/v267/attia25a.html) | Delay-quantile adaptive stochastic optimization | Generic first-order optimizer, without Markov policy-gradient transport or distinct-policy coupling |

The two 2026 arXiv works are contemporaneous unreviewed boundary signals and must not be represented as settled peer-reviewed results.

## Feasibility

### Mathematical feasibility: conditional pass

Taylor transport and the scalar Lyapunov minimizer are elementary but correct under explicit smoothness and confidence assumptions.
The difficult, publishable part is the complete bridge from Markov trajectory estimators and stochastic Hessian-vector products to wall-clock joint-policy stationarity.
The main proof appears feasible for episodic identical-interest Markov games with bounded score functions, bounded importance weights, Lipschitz block Hessian and bounded in-flight population.
Continuing-chain sampling, arbitrary critics and unbounded delays should not be claimed in the first theorem.

### Algorithmic feasibility: conditional pass

One Hessian-vector product costs approximately one additional reverse-mode pass and does not form a Hessian.
The scalar step solve is negligible.
The central practical risk is not arithmetic but estimator quality: a noisy or biased policy Hessian can erase the second-order advantage.
The CPU headroom study must therefore include exact HVP, stochastic HVP and diagonal/low-rank approximations before GPU work.

### Experimental success probability: materially better, not guaranteed

The stopped first-order PUB controller penalized predictable cross changes as if they were irreducible error.
Exact transport removes that penalty on quadratic games and should recover fresh-coordinate progress while retaining parallel completions.
This creates genuine oracle headroom against barrier and discard rules.
Whether approximate neural transport beats strong delay-adaptive and trust-region baselines remains unknown and is the GPU-stage risk.

## Mandatory kill conditions before a new pilot

Stop this mainline before efficacy experiments if any condition fails:

1. the Markov gradient/HVP confidence radius cannot be written from observable, fully charged quantities;
2. the Lyapunov drift cannot telescope to the same joint-gradient stationarity target optimized in code;
3. the scalar event problem loses convexity under the chosen executable radius envelope;
4. an outcome-free strong-comparator oracle calculation shows less than 10% high-delay wall-clock headroom or a material low-delay penalty;
5. HVP cost removes the wall-clock advantage under the registered latency model;
6. a deeper primary-source search finds an existing distinct-policy cross-agent transport theorem with the same scope.

Only after these checks may a new experiment receive an identifier and preregistration.
