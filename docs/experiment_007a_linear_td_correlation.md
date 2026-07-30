# Experiment 007A: correlation-limited speedup in delayed linear TD

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: UNVERIFIED
- Version Label: preregistration_v1

## Objective

Test the retained mechanism in an actual Markov reward process with linear
TD(0), rather than another scalar stochastic-gradient proxy:

1. independent agent trajectories should exhibit near-linear long-run-variance
   reduction;
2. shared Markov transitions should make effective participation saturate;
3. under a fixed message-equivalent budget, this saturation should move the
   optimal agent count from large to small;
4. heterogeneous delay should primarily restrict the stable/best step size,
   without requiring a large direct effect on the optimal agent count.

This is a confirmatory mechanism experiment, not a claim about deep MARL.

## Markov reward process

- seven states on a ring;
- transition probabilities from state \(s\):
  - stay at \(s\): 0.65;
  - move to \(s+1\): 0.20;
  - move to \(s-1\): 0.10;
  - uniform restart: 0.05;
- discount factor \(\gamma=0.9\);
- deterministic bounded sinusoidal reward;
- four fixed features, orthonormalized under the exact stationary
  distribution;
- exact projected TD fixed point computed from
  \(A=\mathbb E[\phi(s)(\phi(s)-\gamma\phi(s'))^\top]\) and
  \(b=\mathbb E[\phi(s)r(s)]\).

The transition matrix, reward, features, stationary distribution, \(A\),
\(b\), and fixed point are saved as artifacts.

## Cross-agent Markov dependence

For every seed, generate one common stationary Markov chain and 32 independent
stationary chains with the same transition matrix. For agent \(i\) and update
\(t\), use the common transition pair with probability \(\sqrt{\rho}\), and
the agent's independent transition pair otherwise. The switching masks are
independent across agents and time. Hence two agents share the common pair
with probability \(\rho\), while every marginal transition pair has the exact
stationary MRP law.

Registered dependence levels are

\[
\rho\in\{0,0.25,0.5,0.75,0.9,1\}.
\]

The augmented common/idiosyncratic chain and masks form the Markov data source.
No policy sees \(\rho\) during a run; EXP-007A maps the oracle surface.

## Delayed TD update and resource budget

At server update \(t\), the first \(q\) agents compute TD directions using
the delayed server parameter \(\theta_{t-\tau_i}\), and the server averages
them:

\[
\theta_{t+1}=\theta_t+
\eta\frac1q\sum_{i=1}^{q}
\phi(s_{i,t})
\left[
r(s_{i,t})+
\gamma\phi(s'_{i,t})^\top\theta_{t-\tau_i}
-\phi(s_{i,t})^\top\theta_{t-\tau_i}
\right].
\]

- \(q\in\{1,2,4,8,16,32\}\);
- heterogeneous maximum delay \(D\in\{0,8,32\}\), using the existing
  deterministic monotone delay profile;
- message-equivalent update cost \(4+q\);
- budgets \(B\in\{2000,8000,32000\}\);
- 13 step sizes geometrically spaced from 0.001 to 0.08;
- 32 paired seeds beginning at 20260930.

Every update, message, and budget is charged. All \(q,\eta,\rho,D\) policies
for one seed use the same underlying paths.

## Metrics

### Finite-budget TD error

At each registered budget, report

\[
\|\theta-\theta^\star\|_2^2,
\]

which equals the mean squared projected-value error because the features are
stationary-distribution orthonormal. For every \((\rho,D,B,q)\), select the
best registered step size by mean error across the 32 paired seeds. Then select
the oracle \(q^\star\).

### Effective participation

At \(\theta^\star\), record the stationary TD-noise sequence and estimate the
trace long-run variance by non-overlapping batch means. Define

\[
N_{\rm eff}(q,\rho)=
\frac{\Omega(1,\rho)}{\Omega(q,\rho)}.
\]

This is measured independently of the optimization-error comparison.

## Preregistered go/no-go gates

All mechanism gates must pass:

1. **independent speedup**:
   at \(\rho=0\), median \(N_{\rm eff}(32,0)\ge16\);
2. **correlation saturation**:
   at \(\rho=0.9\), median \(N_{\rm eff}(32,0.9)\le4\);
3. **participation transition**:
   for \(B=32000\) and \(D=8\), \(q^\star(0)\ge16\) and
   \(q^\star(0.9)\le4\);
4. **material resource value**:
   at the same cells, each oracle-\(q\) choice improves mean error by at least
   10% over using the opposite endpoint's oracle \(q\), after separately
   optimizing the registered step size;
5. **delay/step-size consistency**:
   the best step size at \(D=32\) is no larger than at \(D=0\) in at least
   80% of matched \((\rho,B,q)\) cells;
6. **accounting and numerical validity**:
   all runs are finite, every update is charged, all registered seeds/cells
   complete, and no trajectory exceeds its budget.

The overall mechanism passes only if all six gates pass. A failed gate will
not be repaired by changing the MRP, correlation construction, feature set,
grid, seeds, cell weights, or threshold.

## Statistical reporting

- paired seed-level bootstrap with 2,000 replications for the two
  endpoint-\(q\) material-value comparisons;
- mean, median, 95% interval, and per-seed distributions for
  \(N_{\rm eff}\);
- complete oracle phase table over \((\rho,D,B)\);
- no seed or unstable-looking finite run is discarded.

## Execution

- local Windows CPU with Numba;
- smoke uses two seeds and is excluded from scientific evidence;
- primary command:
  `python run_linear_td_correlation.py --output-dir results/linear_td_correlation --num-seeds 32 --bootstrap-replications 2000 --workers 4`;
- exact reproduction uses the same command and an isolated output directory;
- hard timeout: 30 minutes per run.

### Execution-only amendment after smoke

The two-seed smoke required 116.5 seconds, projecting beyond the registered
30-minute primary timeout. Before the primary run, the 13 step sizes were
batched inside one Numba call and seed evaluation was assigned to four CPU
threads with ordered result collection. A direct test requires the batched and
original scalar kernels to match exactly for errors, updates, and charged
budgets. This amendment changes no MRP, path, seed, update, grid, metric, or
gate; `--workers 4` is now explicit in both primary and reproduction commands.

## Decision after EXP-007A

- If all gates pass, proceed to a fresh-seed nonlinear or larger linear-TD
  confirmation and prove the effective-participation/finite-budget theorems.
- If effective-participation gates pass but the finite-budget transition fails,
  retain only the correlation-limited speedup theorem and redesign the resource
  model before further adaptive-control work.
- If either effective-participation gate fails, reject the current shared-chain
  construction as evidence for the proposed mechanism.
