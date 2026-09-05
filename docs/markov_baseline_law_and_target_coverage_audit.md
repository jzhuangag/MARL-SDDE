# Continuous Markov law and target-coverage audit

## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theory/code validation
- Origin Date: 2026-08-31
- Verification Status: ANALYZED
- Version Label: audit_v1

Internal algebra and source audit; no new citations or manuscript delivery. No efficacy trajectories, optimization reruns, or formal data tuning.

## Confirmed baseline optimization law mismatch

T-083A imports its fixed graph weights through T-081 from T-080. T-079's continuous-graph moment propagator uses `_pre_block` in `t070_nonstationary_graph.py`. That function computes block innovation covariance with the correct within-block AR(1) multiplier, then propagates `A P A^T + B Sigma_block B^T`. Its state contains parameter history and local shadow but not the boundary noise state or parameter/noise cross covariance. Consequently this recursion treats successive block innovations as independent.

In contrast, `sample_markov_observations` maintains one continuous AR(1) noise state across all blocks. This is a genuine law mismatch for nonzero temporal correlation, not just sampling noise.

A two-block witness suffices. Set local_steps=1, gain=eta=.2, identity collaboration, zero initial error/target and unit marginal noise variance. Write a=1-eta=.8. Continuous noise gives

    e_2 = eta*(a*xi_1+xi_2),
    Var(e_2) = eta^2*(a^2+1+2*a*lambda).

The legacy propagator returns eta^2*(a^2+1)=.0656 independently of lambda. At lambda=.8 the correct value is .1168, missing .0512. At lambda=0 they match. The sampler's 1e-14 covariance jitter is a separate negligible numerical difference; it does not explain the missing cross term.

### Scope of impact

The sampled formal comparison uses identical observations and the same end-block simulator for adaptive, fixed and local policies. Those measured comparisons remain what was executed. But at nonzero temporal correlation, the fixed weights were optimized against a mismatched moment law and cannot be described as optimized for the continuous-chain law. The direction and size of any resulting baseline weakness are unmeasured; do not infer a corrected performance ratio.

The temporal-zero primary is unaffected by this cross-block covariance omission. This is not a full endorsement of every other baseline implementation detail. Historical T-068/T-070/T-079 claims of exactness need scope review. Preserve all historical files, weights and outcomes.

### Correct prospective moment interface

For a fixed graph and fixed schedule, augment the parameter/history/shadow state with the last AR noise vector. Propagate that augmented state through individual local steps, including their shared noise, then apply the end-block mixing map. The covariance recursion is exact for deterministic affine maps because the newly injected innovations are independent of the augmented past. Retain stationary initial noise covariance and the parameter/noise cross covariance. The resulting deterministic fixed-graph evaluator can be checked against direct full-trajectory covariance, without Monte Carlo or outcome-based tuning. It does not make adaptive, data-dependent graph moment propagation automatically exact.

## A sufficient target confidence set under explicit assumptions

Consider n agents and a deterministic partition into blocks k=1,...,T of lengths m_k. Observations are Y_(k,s,i)=theta_(k,i)+xi_(k,s,i), with target constant inside each block. Noise is a stationary mean-zero Gaussian AR(1) chain with 0<=lambda<=lambda_bar<1 and marginal variance Var(xi_i)<=vbar_i. The variance and mixing upper bounds are known public valid bounds, not estimates from the flawed NoiseCertificate. Cross-agent correlation is unrestricted subject to the Gaussian covariance law. Blocks can be dependent.

For each fixed block define hat_theta_(k,i) as its sample mean. Direct covariance summation gives

    Var(hat_theta_(k,i)-theta_(k,i))
      <= vbar_i * H(m_k,lambda_bar),
    H(m,l) = [m + 2*sum_{h=1}^{m-1}(m-h)*l^h]/m^2.

Monotonicity follows termwise for 0<=lambda<=lambda_bar. Also H<=min(1,(1+lambda_bar)/(m*(1-lambda_bar))). A scalar Gaussian exponential-moment bound gives Pr(|error|>r)<=2 exp(-r^2/(2 vbar_i H)). Hence

    r_(k,i) = sqrt(2*vbar_i*H(m_k,lambda_bar)*log(2*n*T/delta))

provides simultaneous coverage of all n*T target intervals with probability at least 1-delta, by a union bound. No block independence is required. For a countable deterministic partition, replace delta/(n*T) with delta/(n*k*(k+1)); summability gives all-block coverage. Decisions selected adaptively among those covered blocks are allowed. Arbitrarily data-selected block boundaries or window lengths require a separate union allocation over all allowable windows; they are not covered by the stated n*T formula.

This constructs a valid target interval without estimating spatial correlation. It can feed the previous audit's uniform-in-candidate shield inequality. It is not a learned unknown-mixing certificate, and may be too wide to allow useful collaboration. A separately certified bound can be used only with its own failure probability included.

## What this does and does not prove

On the joint coverage event, enforcing U_i<=epsilon at each evaluated block gives pointwise excess squared error relative to the chosen shadow reference <=epsilon, and therefore the corresponding time-average bound on that event. An explicit shadow action ensures feasibility. This does not prove unconditional expected no-harm: squared Gaussian losses are unbounded on the failure event. An expectation theorem needs bounded predictions/targets or a proved tail/moment contribution. Nor does this prove next-target tracking under arbitrary shifts, positive adaptation gain, or Lyapunov queue stability.

Next authorized CPU work: implement a separate augmented-law audit evaluator and deterministic covariance checks; derive confidence-width/nontriviality feasibility analytically before any new efficacy registration. No old comparator or formal artifact is to be overwritten.

## Execution record

Starting checkout: `4cc4f65`; working tree clean and no Python experiment processes observed. Command: `.venv/Scripts/python.exe -m pytest experiments/dependence_delay_linear/test_markov_baseline_law_audit.py experiments/dependence_delay_linear/test_end_block_debt_audit.py -q`. Result: **11 passed in 0.76s** (six new law/variance cases plus five previous debt cases). The new tests deliberately witness the frozen model discrepancy rather than correcting it. They do not validate coverage empirically or constitute new scientific efficacy evidence. Frozen runs/gates remain unchanged; no GPU or remote operations.
