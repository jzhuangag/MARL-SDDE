# Fixed end-block moment integration audit

## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theory/code validation
- Origin Date: 2026-08-31
- Verification Status: ANALYZED
- Version Label: audit_v1

## Implementation and contract

`fixed_end_block_moments_audit.py` is a separate audit evaluator, not a replacement for any registered runner or optimizer. Its deterministic state contains current parameters, delay history ordered oldest-to-newest, and the persistent local shadow. Every local step updates current and shadow parameters using the same observation. At a decision block end, delay zero mixes current local parameters; positive delay uses the oldest history entry for off-diagonal donors while each recipient's diagonal uses its current local parameter. History then shifts and stores the mixed parameters at every block, including nondecision blocks.

Composing this deterministic end-block map with the last local-step map preserves both its noise injection and target offset. The independent augmented AR(1) covariance engine retains the boundary noise state and all parameter/noise cross terms. Target offsets propagate means. Squared bias plus covariance diagonal gives each block's expected risk, averaged over recipients.

Contract: deterministic fixed graph, deterministic decision blocks, piecewise-constant deterministic targets, common gain in (0,1), stationary vector AR(1), deterministic initial parameters/history/shadow. Adaptive graph choices are not supported. The supplied covariance must already include any intended sampler jitter; no hidden jitter is added. This audit does not optimize weights or establish a global static optimum.

## Independent integration check

The test invokes the **unchanged frozen T-081 simulator with fixed weights** on a small deterministic target sequence. For the full observation-noise covariance K=R(lambda) tensor Sigma, let L be its Cholesky factor and f the simulator's vector of block risks. Because fixed-policy predictions are affine in observations, f is quadratic and its exact expectation is

    E f(mu+noise) = f(mu) + sum_j [(f(mu+L_j)+f(mu-L_j))/2-f(mu)].

This symmetric basis identity uses no Monte Carlo draws and no formal seeds. It independently checks the complete risk output, not merely the same covariance recursion twice. All block risks agree at atol=1e-12, rtol=1e-11 across 12 combinations: delays {0,1,3}, lambda {0,.8}, and full versus intermittent decision schedules. The fixture has correlated two-agent noise, changing targets, nonzero initialization and two local steps per block. Broader dimensions/horizons still require validation before production use.

## Execution

Starting checkout `080604e`; tree clean and no Python processes observed. Command: `.venv/Scripts/python.exe -m pytest experiments/dependence_delay_linear/test_fixed_end_block_moments_audit.py -q`. Result: **12 passed in 1.17s**. Deterministic test evaluations are not new efficacy experiments. No scientific result directory or formal artifact was generated or changed, no optimization rerun, no GPU/HPC4 operation.

## Next

The fixed-policy law interface is now verified on the declared fixture. Next perform outcome-free confidence-width/nontriviality analysis from public model parameters, including the probability that the shadow lies outside the target interval. Compare strict per-block shielding with a mathematically specified cumulative-risk requirement, without changing historical gates or tuning on formal results. A valid covariance evaluator alone does not authorize a new efficacy run or establish ICML-level contribution.
