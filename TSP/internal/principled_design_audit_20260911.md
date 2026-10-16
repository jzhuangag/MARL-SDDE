# Principled-design audit for the TSP controller

## Decision

The theorem-facing method is a principled finite-budget controller rather than a variance-tuning heuristic.
Its decision variable is the mixed action `a=(q,b,eta)`, its objective is the finite-horizon Lyapunov risk certificate, and its dependence probe is admitted by the performance and safety inequalities in the main theorem.

## Component audit

| Component | Mathematical status | Consequence |
|---|---|---|
| Discrete participation `q` and physical spacing `b` | Physical integer controls in a public finite action domain | Exact enumeration is the exact solution of the discrete part, not a relaxation heuristic |
| Step size `eta` | Continuous scalar control on the certified stability domain | Bounded one-dimensional minimization solves the continuous part to declared numerical tolerance |
| Terminal-risk score | Finite composition of the proved one-block Lyapunov drift over the action-dependent usable horizon | The optimization target prices contraction, residual error, delay, messages, and environment interaction together |
| Likelihood probe | Fixed, disjoint, and fully charged controlled Markov experiment | Its error is bounded by an exact Bhattacharyya exponent |
| Probe admission | Theorem-derived learning-gain and low-instance risk inequalities | Reliable identification alone cannot trigger probing |
| Confidence inputs | Predictable simultaneous upper certificates | The guarantee holds on their declared joint coverage event; plug-in point estimates are not treated as theorem certificates |
| MAPPO transfer | Frozen empirical correlation proxy coupled to the same observable probe-then-commit interface | It is policy-learning evidence, while theorem-level nonlinear coverage is not claimed |

## Closest-work boundary

The closest SDDE scheduling work is Yu, Chen, and Poor, IEEE Transactions on Signal Processing, 2025, DOI `10.1109/TSP.2025.3546574`.
That work uses an SDDE and a Poisson arrival approximation to analyze asynchronous stochastic gradient descent and optimize worker activation or staleness policies for runtime convergence.
Recent Lyapunov federated-learning scheduling also couples device and transmit-power decisions to a convergence bound and average communication time (Perazzone et al., IEEE Transactions on Networking, 2025, DOI `10.1109/TON.2025.3539857`).
The present manuscript instead studies correlated Markov observations and minimizes an exact-discrete finite-budget learning-risk certificate jointly over participation, physical spacing, and step size; it additionally decides whether dependence information is worth its downstream learning cost.

## Remaining interpretation boundary

The affine delayed Markov controller and its finite-horizon guarantee are theorem-derived.
The nonlinear MAPPO experiment demonstrates transfer through a frozen empirical correlation proxy and must remain evidence for transfer rather than be described as a theorem-certified nonlinear actor--critic controller.
