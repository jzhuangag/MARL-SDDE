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
| AC-8 | Composite test with unknown \(\lambda\), rather than public \(\lambda\) or a separately certified bound | **open; boundary obstruction proved** | claim narrowed to known mixing or a separately certified upper bound bounded away from one |
| AC-9 | Near-matching regret/safety Pareto upper bound under adaptive probes | **partially proved, matching open** | theorem-derived fallback/error/regret/safety bounds hold, but no uniform constant/log comparison to the adaptive occupation optimum |
| AC-10 | Intersect information-optimal probes with the full matrix delayed-stability region without scalarization loss | **open** | delay enters exactly through budgets here, but general matrix stability coupling is incomplete |
| AC-11 | Extend the Gaussian common-factor result to latent collisions or bounded kernels with unknown baseline | **conjectured route via Theorems 7--8** | nonlinear/general-observation scope remains open |
| AC-12 | SDDE-to-discrete approximation error | **open, inherited** | SDDE remains interpretation, not a discrete approximation theorem |

No open item is silently used in the EXP-015A gate. The pilot tests the
fixed-design mechanism and the horizon transition only.

T-016 therefore triggers stop gates **B** and **D**. It does not authorize an
EXP-016A preregistration or experiment. See
`adaptive_change_of_measure.md`, `adaptive_pareto_lower_bound.md`,
`theorem_derived_fallback.md`, `unknown_mixing_identifiability.md`, and the
machine-readable `theorem_dependencies_t016.json`.
