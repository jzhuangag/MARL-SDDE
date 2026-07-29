# Validation report: EXP-005A budget-matched participation surface

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-29
- Verification Status: VERIFIED
- Version Label: validation_v1

## Validation Report

- **Source**: EXP-005A-budget-participation-surface
- **Overall Confidence**: CAUTION

The numerical mechanism and deterministic reproduction are solid. Confidence
in external or algorithmic generalization remains `CAUTION` because the result
is an oracle surface in a scalar one-factor model and several selected actions
lie on the search boundary.

### Statistical findings

| Metric | Method | Value | Interpretation | Confidence |
|---|---|---:|---|---|
| Independent-noise optimal participation | Exact finite-horizon risk | \(q^\star=32\) for \(D=4,16\) | Full participation is retained when averaging removes independent noise | SOLID within model |
| High-correlation optimal participation | Exact finite-horizon risk | \(q^\star=1\) for \(D=4,16\) | Common noise makes additional messages redundant under the registered budget | CAUTION: boundary optimum |
| High-correlation resource gain | Exact risk ratio | 0.2598 and 0.2581 | Approximately 74% lower MSE than the best all-agent action at matched message budget | CAUTION: simulator/resource scope |
| Wall-clock sensitivity | Exact finite-horizon risk | 4/4 hard cells passed | Participation remains useful under the registered time proxy | CAUTION: proxy time, not measured time |
| Uniform-rank sensitivity | Exact finite-horizon risk | 2/2 high-correlation cells passed | Effect is not solely fastest-agent selection | SOLID within model |
| Exact/Monte Carlo agreement | 10,000 replications per action | 8/8 passed; maximum relative difference 2.10% | Exact implementation is empirically cross-checked | SOLID |

No p-value is reported because the primary surface is an exact calculation
over a registered finite grid, not a sample from a population of tasks. The
Monte Carlo runs validate numerical agreement rather than test an empirical
scientific null.

### Warnings

| Type | Detail | Affected claim |
|---|---|---|
| Boundary optimum | Correlated primary cells select both \(q=1\) and the minimum step size 0.0025 | Shape and magnitude of the participation gain |
| Oracle information | Correlation, delay model, and resource coefficients are available to the grid search | Implementability |
| Resource model | Message overhead and wall-clock coefficients are registered simulator choices, not hardware measurements | Systems generalization |
| Correlation model | Agents share one exchangeable common factor plus independent residuals | Heterogeneous/clustered agent populations |
| Task scope | Scalar linear fixed-point learning is evaluated | TD, nonlinear RL, and multi-agent games |

### Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Direction was checked across delay, alignment, selection rule, and overhead; no reversal invalidates the registered primary result. |
| Ecological fallacy | NOTE | Conclusions are restricted to aggregate simulator risk; no individual-agent behavior is inferred. |
| Berkson's paradox | NOTE | Fastest-agent selection is a filtered sample, but the registered uniform-rank sensitivity preserves the direction. |
| Collider bias | NOTE | No post-treatment covariate or common-effect adjustment is used. |
| Base-rate neglect | NOTE | Diagnostic conditional probabilities are not used. |
| Regression to the mean | NOTE | There is no extreme-group pre/post selection. |
| Survivorship bias | NOTE | Unstable grid actions remain in the raw surface and are excluded only by the registered admissibility rule; every selected action is stable. |
| Look-elsewhere effect | NOTE | The full registered grid and all five gates are retained; results were not selected by significance. |
| Garden of forking paths | CAUTION | The experiment was pre-registered, but external conclusions remain sensitive to the chosen budget and time proxy coefficients. |
| Correlation versus causation | NOTE | Comparisons are interventions in a mathematical simulator; no real-system causal claim is made. |
| Reverse causality | NOTE | Not applicable to the controlled parameter sweep. |

### Reproducibility

- **Method**: deterministic same-code, same-seed rerun to an independent output
  directory;
- **Verdict**: REPRODUCIBLE.

The retained `surface.csv`, `optimal_actions.csv`,
`monte_carlo_validation.csv`, `summary.json`, and both figures were
byte-identical by SHA-256.

### Validated conclusion

At a fixed message or proxy-time budget in the evaluated one-factor delayed
Markov model, the best number of participating agents changes with
cross-agent dependence, and using all available agents can be strongly
resource-inefficient. This validates proceeding to a genuinely online,
probe-charging controller experiment; it does not yet validate such a
controller or a general reinforcement-learning claim.

