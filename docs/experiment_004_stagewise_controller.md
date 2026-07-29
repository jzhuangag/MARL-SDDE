# Experiment 004: Predictable stagewise step–participation controller

## Material Passport

- Artifact type: pre-registered code experiment
- Experiment ID: `EXP-004-stagewise-controller`
- Parent experiments: `EXP-001` and `EXP-003`
- Status: completed; primary gate failed; reproduction pending
- Registration time: before inspecting EXP-004 outputs
- Compute target: local CPU
- Machine-readable outputs:
  `experiments/dependence_delay_linear/results/stagewise/`

## Research question

Can a low-complexity controller use only statistics from the preceding stage
to adapt the scalar step size and accepted-agent count after changes in
cross-agent dependence and staleness?

## Model and information pattern

The experiment uses the delayed scalar Markov model from EXP-001 with 32
agents. Candidate accepted-agent counts are

\[
\mathcal Q=\{1,2,4,8,16,32\}.
\]

At stage \(m\), the server uses a fixed pair \((\eta_m,q_m)\). Statistics
collected during stage \(m\) may affect only \((\eta_{m+1},q_{m+1})\); the
controller is therefore predictable with respect to the stage filtration.

All 32 scalar messages are observed and prefix-aggregated to estimate candidate
statistics, while only the selected prefix enters the parameter update. Thus
EXP-004 tests low-complexity aggregation selection, not communication-saving
worker activation. The extra statistic is \(O(|\mathcal Q|)\) scalars per
stage; no \(32\times32\) covariance matrix is estimated.

## Controller statistic

For each \(q\in\mathcal Q\), a scalar regression removes the delayed drift from
the aggregate update. Batch means applied to the residual series produce a
long-run variance estimate \(\widehat{\nu}_{m,q}\). The batch size is fixed at
20 before execution.

For each candidate \((q,\eta)\), the controller constructs the deterministic
delay companion matrix. Its stage-end bias and unit-white-noise Lyapunov gain
define the proxy

\[
\widehat R_m(q,\eta)
=
\widehat{\operatorname{Bias}}_m^2(q,\eta)
+
\widehat{\nu}_{m,q}\,
\mathcal H_2(q,\eta).
\]

The next-stage action minimizes this proxy over \(\mathcal Q\) and 13
log-spaced step sizes from 0.0025 to 0.04. Unstable candidates are excluded.

## Regime schedule

The experiment contains 12 stages of 200 updates each:

1. stages 0–3: independent noise, \(\rho=0\), maximum delay 4;
2. stages 4–7: persistent common noise, \(\rho=0.9\), maximum delay 16;
3. stages 8–11: partial dependence, \(\rho=0.3\), maximum delay 8.

The primary setting uses a shared server-time common factor, corresponding to
agents observing a contemporaneous global environment with stale parameter
copies. The same schedule under sample-time alignment is a pre-specified
sensitivity check.

The baseline run uses 64 paired random seeds. Every policy receives the same
common and idiosyncratic Markov noise path within a seed.

## Policies

- `adaptive_joint`: estimates candidate long-run variances and adapts both
  \(q_m\) and \(\eta_m\);
- `delay_only`: uses the same predictable architecture but imposes the
  independence model
  \(\widehat\nu_{m,q}=\widehat\nu_{m,1}/q\);
- `all_agents_adaptive_eta`: estimates dependence but fixes \(q_m=32\);
- `all_agents_fixed`: fixes \(q_m=32\) and \(\eta_m=0.02\);
- `proxy_oracle`: knows the current regime parameters and delay profile and
  minimizes the same proxy. It is a diagnostic reference, not a dynamic-
  programming oracle.

## Primary metrics

- mean MSE during adapted high-correlation stages 5–7;
- final-phase and full-run mean MSE;
- chosen \(q_m\) and \(\eta_m\) by stage;
- paired bootstrap confidence intervals for high-correlation MSE ratios;
- gap to the proxy oracle.

The first high-correlation stage is excluded from the adapted-phase comparison
because a predictable controller cannot react before observing the shift.

## Pre-registered go/no-go criteria

The controller passes the primary feasibility gate only if all conditions hold
in the server-time setting:

1. `adaptive_joint` high-correlation MSE is at least 10% below `delay_only`;
2. it is at least 5% below `all_agents_adaptive_eta`;
3. its mean high-correlation MSE is no more than 1.25 times the proxy oracle;
4. its median selected count is at least 16 in stages 1–3 and at most 8 in
   stages 5–7;
5. no policy has a non-finite trajectory and the same-seed reproduction
   changes every reported aggregate by less than 5%.

The sample-time sensitivity passes if criteria 1 and 4 retain the same
direction; it is not allowed to rescue a failed primary gate.

## Interpretation boundary

Passing would justify further work on a stagewise dependence-aware controller
for linear temporal-difference learning. It would not establish convergence
for nonlinear MARL, communication savings, or minimax optimality. Failure
would indicate that step-size adaptation alone is sufficient in this model or
that the scalar long-run variance estimator is too noisy to support reliable
participation control.

## Execution

The registered 64-seed experiment completed on 2026-07-29 with an empty stderr
log. Every policy used paired common and idiosyncratic Markov paths within each
seed. The command was

```powershell
python run_stagewise_controller.py `
  --output-dir results/stagewise `
  --num-seeds 64 `
  --bootstrap-replications 2000
```

Eight deterministic tests covering the base model and controller primitives
passed before execution.

## Primary results

During adapted high-correlation stages 5–7 in the server-time setting:

- `adaptive_joint / delay_only` MSE ratio:
  \(0.3616\), bootstrap 95% interval \([0.3083,0.4208]\);
- `adaptive_joint / all_agents_adaptive_eta` MSE ratio:
  \(1.0014\), interval \([0.9770,1.0289]\);
- `adaptive_joint / proxy_oracle` MSE ratio:
  \(1.2345\), interval \([1.1214,1.3616]\).

The adaptive controller's median participation was 32 during independent
stages 1–3 and 4 during adapted high-correlation stages 5–7. Thus it detected
the dependence shift and changed participation in the expected direction.
However, this action change did not improve MSE over fixing all 32 agents and
adapting only the scalar step size.

The corresponding phase-averaged MSE values were:

| Policy | High-correlation MSE | Full-run MSE | Final-window MSE |
|---|---:|---:|---:|
| `adaptive_joint` | 0.04712 | 0.03047 | 0.02031 |
| `all_agents_adaptive_eta` | 0.04705 | 0.03033 | 0.01947 |
| `delay_only` | 0.13030 | 0.06363 | 0.05068 |
| `all_agents_fixed` | 0.28445 | 0.13639 | 0.08382 |
| `proxy_oracle` | 0.03817 | 0.02415 | 0.01516 |

No simulated trajectory was non-finite.

## Gate outcome

The pre-registered primary gate **failed** because criterion 2 required at
least a 5% improvement over `all_agents_adaptive_eta`, whereas the observed
point ratio was 1.0014. The server-time criteria for delay-only improvement,
oracle point gap, participation response, and finite trajectories passed.

The sample-time sensitivity also failed its directional participation gate:
the high-correlation median remained 32, although the distribution was broad
and the MSE improvement over delay-only remained large.

## Interpretation

EXP-004 supports the following narrower result:

> Estimating aggregate dependence is essential for selecting a stable,
> effective scalar step size after a correlation shift; once that step size is
> adapted, participation control provides no detectable MSE improvement in
> the present linear model.

The controller often selected the minimum available step size, which reduced
the effective physical delay \(\eta\tau\) and largely removed the accuracy cost
of accepting stale agents. This explains why the participation response was
mechanistically sensible but statistically unnecessary for MSE.

The observed reduction in selected \(q\) cannot yet be claimed as a
communication saving because EXP-004 observes all 32 scalar messages to
estimate candidate statistics.

## Reproduction status

A same-seed 64-run reproduction was attempted with a 180-second hard timeout.
The process was terminated before output collection and produced no
machine-readable result. In accordance with the execution protocol it was not
silently retried. Therefore the EXP-004 status remains
`COMPLETED_PENDING_REPRODUCTION`, not `VERIFIED`.

## Research decision

The current evidence does not support making dynamic participation the main
accuracy contribution. The next experiment should either:

1. focus the theory and algorithm on correlation-aware scalar step-size
   adaptation; or
2. redesign participation as a resource-saving mechanism using occasional
   probing, and evaluate accuracy at matched communication or wall-clock cost.

Option 2 requires a new pre-registration because it changes the information
pattern and target metric.
