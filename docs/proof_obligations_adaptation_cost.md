# Proof-obligation ledger: adaptive participation cost

| ID | Statement | Status | Blocking consequence |
|---|---|---|---|
| AC-1 | Spatial rotation reduces the fixed \(q\)-agent Gaussian experiment to covariance \(I+q\theta R_n(\lambda^b)\) | **proved** | none |
| AC-2 | Exact directional KL and Bhattacharyya formulas | **proved**, dense numerical identity tested | none |
| AC-3 | Fixed-design necessary sample threshold from binary data processing | **proved** | none |
| AC-4 | Fixed-design likelihood test with Bhattacharyya sufficient threshold | **proved** | none |
| AC-5 | Dual-budget and delay threshold for a fixed probe block | **proved** by accounting identity | none |
| AC-6 | All-agent individual feedback has positive regime information, so strict no-harm is not universally incompatible with adaptation | **proved** for the registered Gaussian model | rules out an overbroad impossibility claim |
| AC-7 | Adaptive chain-rule lower bound for history-dependent \((q_t,b_t)\) with dimension-changing observations | **proved in T-016** for the stationary Gaussian model with known \(\lambda\) | exact controlled-belief information functional is available |
| AC-8 | Uniform composite test over unknown \(\lambda\) reaching arbitrarily close to one | **closed negatively in T-017** for positive distinct \(\theta_0,\theta_1\); compact separated positive route open | unrestricted unknown-mixing wording is impossible; require known mixing or \(\lambda\leq1-\gamma\) certificate |
| AC-9a | Universal constant/log safe-adaptation matching without separation | **closed negatively in T-017** | oracle-gap degeneration makes \(B_S/B_N\to\infty\) |
| AC-9b | Finite-budget necessary/sufficient adaptation-threshold sandwich | **proved in T-017** on the declared compact separated class | permits the narrow threshold theorem, not global adaptive optimality |
| AC-9c | Match an adaptive controller to the entire controlled-belief occupation optimum | **open** | no Track-and-Stop-style global matching claim |
| AC-10 | Intersect information-optimal probes with the full matrix delayed-stability region without scalarization loss | **open** | delay enters exactly through budgets here, but general matrix stability coupling is incomplete |
| AC-11 | Extend the Gaussian common-factor result to latent collisions or bounded kernels with unknown baseline | **conjectured route via Theorems 7--8** | nonlinear/general-observation scope remains open |
| AC-12 | SDDE-to-discrete approximation error | **open, inherited** | SDDE remains interpretation, not a discrete approximation theorem |

No open item is silently used in the EXP-015A gate. The pilot tests the
fixed-design mechanism and the horizon transition only.

T-017 supersedes the T-016 AC-8/AC-9 status. Its decision is **A under D**:
retain the separated-class threshold theory and permanently narrow
unknown-mixing scope. T-017 itself does not authorize an EXP-016A
preregistration or experiment. See
`adaptive_change_of_measure.md`, `adaptive_pareto_lower_bound.md`,
`unknown_mixing_impossibility.md`, `adaptation_threshold_sandwich.md`, and the
machine-readable `theorem_dependencies_t017.json`.

T-018 adds a safety-metric alignment obligation and a preregistered
learning-value separation scan. The theorem-facing safety metric is `S_mean`;
`S_path` is only a stronger descriptive tail metric unless a separate
Gaussian quadratic-loss tail bound is proved. The conditional separation
claim is formulated through `B_id < B_value` and the zone
`Z={B:B_id<=B<B_value}`. The frozen scan grid hash is
`c5d2dd5ddac7540888d708ab59d4e3954994da018951797a79d05200ef0ee2db`.
At the preregistration point, no scan result or experiment is authorized; see
`theorem_dependencies_t018.json`.
