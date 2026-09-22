## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-free formulation feasibility audit
- Origin Date: 2026-09-07
- Verification Status: FROZEN BEFORE INDEPENDENT AUDIT SEEDS
- Version Label: pursuit_local_factor_interface_v1

# Local-factor interface qualification for asynchronous MARL

## Why this is a formulation change

The stopped Pistonball line used a monolithic centralized critic and then hoped
that a geometric causal cone would imply low policy-dependency degree. The
frozen exact-score audit disproved that premise: some owners had 16 eligible
donors. The prospective formulation instead starts with a locally factored
Markov game and a critic whose action factors have bounded arity. Sparsity is
therefore an explicit model/architecture contract, not an empirical property
claimed after training.

PettingZoo Pursuit is used only for a structural interface qualification. It
has eight controlled pursuers, random evaders, square 7-by-7 ego-centric
observations, and local capture mechanics. At centralized training time, a
directed factor graph connects each owner to at most its four nearest pursuers
inside the observable Chebyshev radius three. The tie-break is deterministic.
No reward, return, gradient, critic output, or prior learning result enters the
graph.

This bounded graph is the support of a prospective local critic factor. It is
not asserted to equal the environment's unrestricted optimal value function;
the theorem must carry an explicit factor-approximation error. Decentralized
execution remains unchanged and communication-free.

## Independent CPU audit

The exploratory mechanics scan used environment seeds 90000--90031 and action
seed 8841. It is design information only. The independent audit freezes fresh
environment seeds 91000--91031 and action seed 8842, 100 cycles per seed,
eight pursuers, 30 evaders, observation range seven, `n_catch=2`, shared reward,
and maximum factor degree four.

The audit reads positions and graph identities but explicitly discards every
reward returned by the environment. It must satisfy all of:

1. at most six reverse evaluations per owner action (current, null, and four
   local-edge counterfactuals);
2. raw local degree exceeds four in at most 2% of owner-state pairs;
3. consecutive graph turnover in at least 30% of state transitions;
4. the initial static graph misses at least 30% of later dynamic edge
   occurrences;
5. a nonempty graph in at least 95% of states.

Passing establishes only that the benchmark exposes a dynamic, mostly natural
low-degree interface at bounded exact-score cost. It does not establish
learning-value headroom or authorize a GPU learner. Failure stops this
factorization before any efficacy experiment.

## Unified algorithmic object if the interface passes

At an asynchronous launch/receipt event, the server will eventually choose a
local cache-refresh vector and a scalar owner update weight by minimizing one
bound on the drift of

\[
  \mathcal L_t
  = V\,\Phi(\theta_t)
  + \frac{\beta}{2}\sum_{(i,j)\in E_t}
      w_{ij,t}\|\theta_{j,t}-\widehat\theta_{j\mid i,t}\|^2
  + \frac{Q_t^2}{2\nu}.
\]

Here `Phi` is the cooperative learning potential, the middle term is
strategic cache mismatch on the state-dependent factor graph, and `Q` is the
communication debt. Thus Lyapunov is the design objective for both update
weight and graph refresh, not only a post-hoc stability proof. The exact
discrete event-time theorem remains primary. An SDDE is admissible only after
a generator/coupling theorem connects its delayed jump process to this
recursion.

The two remaining kill conditions after this structural audit are a proved
performance bound containing factor-approximation and Markov-bias errors, and
a positive equal-resource oracle headroom result against strong dynamic and
static schedulers. Until both exist, this is a candidate formulation rather
than an ICML contribution claim.
