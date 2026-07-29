# Validation report: EXP-006A oracle participation phase diagram

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: validation_v1

## Experiment result

- **ID**: EXP-006A-oracle-participation-phase
- **Type**: deterministic simulation
- **Status**: completed
- **Command**:
  `python run_oracle_phase.py --output-dir results/oracle_phase`
- **Working directory**: `experiments/dependence_delay_linear`
- **Primary duration**: 87.6 seconds
- **Exit code**: 0
- **Anomalies**: none

The registered grid contains exactly 9,720 finite oracle cells. No cell was
excluded, interpolated, or added after execution.

## Preregistered findings

| Gate | Observed result | Verdict |
|---|---:|---|
| Non-degenerate surface | all six \(q\) values exceed 1%; largest share 43.12% | PASS |
| Correlation responsiveness | 37.22% responsive tracks | PASS |
| Decision margin | 79.60% of responsive tracks well-separated | PASS |
| Delay relevance | 5.31% delay-sensitive groups | FAIL |
| Actionable region | 10 contiguous rectangles | PASS |
| Numerical validity | 9,720/9,720 finite cells | PASS |
| Overall | all six gates required | **FAIL** |

Responsive-track fractions were 35.56% for global dependence, 43.06% for
clustered dependence, and 33.06% for balanced dependence. Oracle action
occupancy was 27.11% for \(q=1\), 9.40% for \(q=2\), 11.80% for \(q=4\),
5.23% for \(q=8\), 3.34% for \(q=16\), and 43.12% for \(q=32\).

Because the surface is an exhaustive deterministic grid rather than a random
sample, these percentages are descriptive coverage measures; confidence
intervals and null-hypothesis tests are not applicable.

## Mechanism interpretation

The formal overall result is a preregistered failure: the current model does
not support a combined claim that oracle participation is strongly adaptive to
both dependence and delay.

The failure is localized. Correlation produces a non-degenerate, frequently
well-separated participation transition. Responsive-track coverage increases
from 7.78% at budget 250 to 63.33% at budget 8000, and is concentrated at
current error amplitudes 0.1--1.0. Ten adjacent budget-error rectangles satisfy
the registered endpoint and margin requirements simultaneously for delays 4
and 16. These are valid candidate domains for a future correlation-adaptive
controller.

Delay rarely changes the pointwise optimal \(q\): only 5.31% of fixed
path-strength-budget-error-overhead groups select different \(q\) across the
four delays. A post-registration diagnostic found that delay changed the
optimal scalar step size in 7.57% of the same groups. Delay nevertheless shifts
where correlation-responsive tracks occur: responsive coverage rises from
29.26% at \(D=0\) to 47.04% at \(D=32\). This supports treating delay as a
stability/rate parameter rather than assuming it must directly control agent
count.

The phase diagram is also state dependent. At high dependence and small
current error, \(q=32\) can remain optimal, whereas at error 0.1--1.0 the
global and balanced paths often select \(q=1\), and the clustered path often
selects \(q=4\). A controller based on correlation alone would therefore be
misspecified; the finite-budget state or an observable error proxy is required.

## Reproducibility

- **Method**: deterministic full rerun with the same code and grid; output
  isolated under `results/reproduction/oracle_phase`
- **Rerun duration**: 86.2 seconds
- **Verdict**: REPRODUCIBLE

All seven expected artifacts matched byte-for-byte by SHA-256:
`oracle_surface.csv`, `track_summary.csv`, `delay_summary.csv`,
`actionable_rectangles.csv`, `summary.json`, `fig1_oracle_phase.png`, and
`fig2_q_frequency.png`.

## Fallacy scan

- **Coverage**: 11/11 statistical fallacy types checked.

| Fallacy | Status | Finding |
|---|---|---|
| Simpson's paradox | CAUTION | Aggregate occupancy masks strong budget/error dependence; conclusions are therefore stated at track and rectangle levels. |
| Ecological fallacy | NOTE | Grid-cell coverage is not interpreted as a probability over real deployments. |
| Berkson's paradox | NOTE | The full registered grid was retained; no outcome-conditioned selection occurred. |
| Collider bias | NOTE | No variables were conditioned on after observing oracle outcomes. |
| Base-rate neglect | NOTE | No diagnostic classification probabilities are used. |
| Regression to the mean | NOTE | The experiment has no pre-post sampling or extreme-cell selection. |
| Survivorship bias | NOTE | All 9,720 registered cells completed and were analyzed. |
| Look-elsewhere effect | NOTE | All gates and the rectangle search were registered before the main scan. |
| Garden of forking paths | NOTE | The failed delay threshold was retained; no grid or cutoff was repaired post hoc. |
| Correlation versus causation | NOTE | Dependence is directly set in a deterministic model; no observational causal claim is made. |
| Reverse causality | NOTE | Not applicable to the controlled oracle surface. |

## Validated conclusion

EXP-006A does not pass its combined delay-and-correlation participation
go/no-go rule. It does establish a reproducible oracle target for
state- and correlation-adaptive participation over ten contiguous regions.
The next experiment may test that narrower target, while SDDE delay enters the
stability and convergence analysis rather than being forced to produce a large
pointwise change in \(q^\star\).
