# Efficient critic and a stronger baseline requirement

## Why this extension is required

CPU-001 checks a data-driven action score for a constant-step TD class. That
qualification cannot establish superiority to an efficient averaging critic.
Before any ICML algorithm claim we therefore enlarge the learner class and
compare the proposed allocator with equally efficient fixed-q learners.
This extension is frozen independently of CPU-001 confirmation outcomes.

## Lyapunov-optimal update gain

For retained persistence r=lambda^b and observations Y_t=theta+Z_t with
stationary Gaussian AR(1) noise variance v_q, the variables

Y_1, (Y_2-r Y_1)/(1-r), ..., (Y_N-r Y_(N-1))/(1-r)

are independent unbiased observations of theta. Their noise variances are v_q
for the first and v_q(1+r)/(1-r) for the rest.

If the current unbiased estimate has variance V and the next independent
observation has variance W, a gain kappa gives Lyapunov risk

V_next = (1-kappa)^2 V + kappa^2 W.

Minimizing this quadratic gives kappa=V/(V+W). Starting from Y_1 gives the
online gain kappa_t = [(1-r)/(1+r)] / I_t for t>=2, where

I_t = [t(1-r)+2r]/(1+r).

The gain is causal, has no target-value input, and requires constant memory.
The resulting estimator has exact risk v_q/I_N. Direct inversion of the AR(1)
covariance gives Fisher information I_N/v_q. The estimator attains its
Gaussian Cramer--Rao bound and is the best linear unbiased estimator.
The Kalman/GLS identity is inherited machinery, not claimed as novel.

## Role of parameter staleness in this exact subclass

A stale scalar TD packet contains g_old = w_old-Y_t and its parameter version.
The server reconstructs Y_t=w_old-g_old exactly. Whitening uses the original
observation order and clock gaps. This cancels parameter staleness for this
known constant Jacobian; it does not cancel unknown data delays, nonlinear
Jacobians, missing versions, or policy-induced target movement. CPU-001's D
is parameter staleness. D cells are retained, but this exact cancellation is
reported rather than presented as a general asynchronous-MARL theorem.

## Fully charged allocation and true-risk intervals

Every (q,b) has its exact post-calibration usable N. Minimize fitted v_q/I_N
over the complete catalogue. The same exact calibration intervals from
observable_risk_interface.md give L_q/I_N <= R <= U_q/I_N simultaneously.
The selected true-risk regret is at most selected upper minus minimum lower.
Fixed-q comparators receive the same efficient estimator, free true-context
optimal spacing, full budgets, and no calibration tax.

The new evidence therefore tests allocation value against strong learning
rules. It does not infer a nonlinear PPO learning-response law from Gaussian
observations. A separate real-optimizer response qualification remains required.
