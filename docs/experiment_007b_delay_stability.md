# Experiment 007B: active delay-stability boundary in linear TD

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: UNVERIFIED
- Version Label: preregistration_v1

## Motivation

EXP-007A validates correlation-limited effective participation, but its
registered delay gate passes only by equality: the finite-budget optimal
step size is far inside the delayed stability region. EXP-007B separately
activates the delay mechanism. It does not change or retroactively rescue any
EXP-007A gate.

## Mean delayed TD system

Use the same registered seven-state MRP, four features, and projected TD matrix
\(A\). Around the fixed point, the mean error recursion for the first \(q\)
agents is

\[
e_{t+1}=e_t-\frac{\eta}{q}\sum_{i=1}^{q}A e_{t-\tau_i}.
\]

For each \(q,D,\eta\), construct the exact block companion matrix and compute
its spectral radius. Define \(\eta_{\rm crit}(q,D)\) by deterministic bisection
as the largest positive step size with spectral radius below one.

## Registered cells

- \(q\in\{8,16,32\}\);
- maximum delay \(D\in\{0,8,32\}\), with the existing monotone heterogeneous
  delay profile;
- relative step sizes
  \(m\in\{0.50,0.80,0.95,1.05,1.20\}\), using
  \(\eta=m\eta_{\rm crit}(q,D)\);
- Markov sharing probability \(\rho\in\{0,0.9\}\);
- 16 paired seeds beginning at 20261030;
- 4,000 server updates;
- divergence threshold \(\|\theta-\theta^\star\|^2>10^{12}\);
- initial parameter error is the fixed all-ones direction with squared norm
  one.

For the controller comparison:

- `delay_adaptive`: \(\eta=0.8\eta_{\rm crit}(q,D)\);
- `delay_blind`: \(\eta=0.8\eta_{\rm crit}(q,0)\), applied unchanged at
  \(D>0\).

The simulator stops a divergent trajectory at the registered threshold and
records the first crossing; it does not allow floating-point overflow to define
the outcome.

## Go/no-go gates

All gates must pass:

1. **active boundary**:
   \(\eta_{\rm crit}(q,32)/\eta_{\rm crit}(q,0)\le0.35\) for
   \(q=16,32\);
2. **exact spectral separation**:
   every \(m\le0.95\) companion matrix has radius below one and every
   \(m\ge1.05\) matrix has radius above one;
3. **Monte Carlo boundary agreement**:
   at \(m=0.8\), no run crosses the divergence threshold and median final
   error is below initial error; at \(m=1.2\), at least 90% of delayed
   \(D\in\{8,32\}\) runs cross the threshold;
4. **delay-adaptive value**:
   for \(D=32,q\in\{16,32\}\), every delay-adaptive run remains below the
   threshold, while at least 90% of delay-blind runs cross it;
5. **correlation/stability separation**:
   the exact critical step size is independent of \(\rho\), while at
   \(m=0.8\) the median final error at \(\rho=0.9\) is no smaller than at
   \(\rho=0\) in at least 80% of matched \(q,D\) cells;
6. **accounting, determinism, and numerical validity**:
   all registered cells complete, stable-run metrics are finite, threshold
   crossings are explicitly recorded, and the exact boundary is reproducible.

Failure of a gate rejects this exact delay-stability demonstration. Threshold,
horizon, multipliers, cells, and classification rules will not be changed
after smoke execution.

## Interpretation limit

This experiment can validate the SDDE/Lyapunov stability mechanism and a
low-complexity delay-aware scalar step size. It cannot establish that delay
directly changes the resource-optimal agent count, and no such claim will be
made.

## Execution

- local Windows CPU;
- two-seed smoke excluded from evidence;
- primary:
  `python run_td_delay_stability.py --output-dir results/td_delay_stability --num-seeds 16`;
- exact reproduction uses an isolated output directory;
- hard timeout: 20 minutes.

## Execution outcome

The 16-seed primary run completed all 1,632 stochastic trajectories and passed
four of six gates. Exact mean-system delay boundaries were active and all
spectral classifications were correct, but the nominally stable 0.8-boundary
rule still produced stochastic TD divergence. The delay-blind rule crossed the
threshold in every target run, yet the mean-boundary delay-adaptive rule also
crossed under strong correlation. The overall verdict is **FAIL**.

The full rerun reproduced all seven artifacts byte-for-byte. The failure
identifies a mean-versus-mean-square stability gap and a correlation-dependent
random-Jacobian effect. See `validation_exp007b.md`.
