# Paper blueprint

## Central thesis

Correlation, temporal persistence, communication overhead, and delay jointly determine the finite-time value of an additional agent trajectory.
A single certificate-driven controller can expose this structure through three coupled actions, namely participation `q`, spacing `b`, and step size `eta`, while a learning-aware information gate prevents reliable but economically unproductive probing.

## Argument chain

1. Cross-agent correlation limits the variance reduction obtained from participation, while temporal spacing reduces Markov bias at an environment cost.
2. A delayed affine temporal-difference recursion admits a computable Lyapunov finite-time bound whose curvature, forcing, mixing, and delay terms depend explicitly on `(q,b,eta)`.
3. Minimizing this bound over a finite stability-screened catalogue gives a low-complexity resource controller rather than a heuristic participation rule.
4. When the dependence regime is unknown, any adaptive controller must acquire directional information and therefore incurs an occupation-measure opportunity cost.
5. A learning-aware explore-then-commit rule pays this cost only beyond a sufficient learning-value threshold and otherwise uses the certified all-agent fallback.
6. The necessary and sufficient thresholds are comparable on a compact separated class, which connects information limits to optimization value.
7. Three independent preregistered CPU studies validate the predicted correlation saturation, joint action response, finite-time coverage, safety, and learning-value separation.

## Section architecture

| Section | Purpose | Target share |
|---|---|---:|
| I. Introduction | Establish the finite-budget dependence-control problem and contributions | 12% |
| II. Related Work | Position against distributed TD, Markovian stochastic approximation, and active testing | 10% |
| III. Model and Objective | Define data, costs, delay, catalogue, and finite-time risk | 15% |
| IV. Lyapunov-Certified Resource Control | Derive the affine bound and joint `(q,b,eta)` selection | 20% |
| V. Learning-Aware Dependence Adaptation | Derive adaptive information lower bound, safe upper rule, and threshold theorem | 20% |
| VI. Experiments | Test mechanism, certificate, and learning-value predictions | 17% |
| VII. Conclusion | State the solved problem and significance | 6% |

## Evidence map

| Claim | Theoretical evidence | Experimental evidence |
|---|---|---|
| Correlation saturates multi-agent speedup | Exact Gaussian minimax identity | EXP-007A effective participation |
| `(q,b,eta)` can be selected by a finite-time certificate | Affine delayed Markov-TD Lyapunov theorem | EXP-010B calibration and action response |
| Reliable identification need not be worth its cost | Adaptive KL lower bound and learning-value decomposition | EXP-016B Layer A |
| The mechanism transfers to affine delayed TD | Finite-risk catalogue and fallback theorem | EXP-016B Layer B |
| The controller is reproducible and resource-valid | Exact accounting and deterministic analysis | Independent byte-exact reproductions |
