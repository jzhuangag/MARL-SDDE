## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-free theorem-interface derivation
- Origin Date: 2026-09-07
- Verification Status: FINITE-HORIZON KERNEL PERTURBATION PROVED; NEURAL RADIUS OPEN
- Version Label: pursuit_controlled_kernel_bridge_v1

# Controlled-kernel bridge for locally factored asynchronous MARL

## Purpose

The tabular forecast-reversal model uses a common exogenous state kernel.  A
Pursuit cache refresh is harder: replacing one cached teammate policy changes
the joint behavior policy and therefore the finite-horizon trajectory law that
generates the next owner packet.  A standard-task theorem may not reuse the
common-kernel simplification.  This note states the exact perturbation charge
that must enter the plug-in Lyapunov oracle term before a Pursuit efficacy run
is allowed.

The setup is CTDE.  At launch event `p`, the owner is `i`; its own current
policy is used, while teammate `j` is represented at the worker by cached
version `chi_(j|i)`.  Candidate `a=j->i` replaces only that cache before the
next trajectory.  Execution of the final actors remains decentralized.

## Lemma 1: a one-edge refresh changes the controlled kernel locally

Let `pi_p^a(.|s)` be the product joint policy after candidate refresh `a` and
let `P_p^a` be its induced state kernel.  Relative to the null refresh define

\[
 d_p(a)=\sup_s \operatorname{TV}(\pi_p^a(\cdot|s),
                                  \pi_p^0(\cdot|s)).
\]

If `a` changes only teammate `j`, product-policy coupling gives

\[
 d_p(a)\leq \sup_{o_j}\operatorname{TV}
 \bigl(\pi_{\theta_{j,p}}(\cdot|o_j),
       \pi_{\chi_{j|i,p}}(\cdot|o_j)\bigr).
\tag{1}
\]

For every state `s`, data processing through the environment transition gives

\[
 \operatorname{TV}\bigl(P_p^a(s,\cdot),P_p^0(s,\cdot)\bigr)
 \leq d_p(a).
\tag{2}
\]

Both inequalities are exact consequences of coupling: couple every unchanged
actor identically, maximally couple actor `j`, and then use the same
environment randomness whenever the joint actions agree.  No stationarity or
mixing assumption is used.

## Lemma 2: finite-horizon trajectory-law charge

Start the two candidate trajectories from the same observed launch state.  If
their one-step controlled kernels differ by at most `d_p(a)` in total
variation, a stepwise maximal coupling implies

\[
 \operatorname{TV}(\mathcal L_a(Z_h),\mathcal L_0(Z_h))
 \leq 1-(1-d_p(a))^h\leq \min\{1,h d_p(a)\}.
\tag{3}
\]

Consequently, for predictable bounded state scores `f_(p,a,h)` with
oscillation `R_(p,a,h)`, the difference between candidate and null expected
finite-horizon scores satisfies

\[
 \left|\mathbb E_a\sum_{h=0}^{H-1}f_{p,a,h}(Z_h)
       -\mathbb E_0\sum_{h=0}^{H-1}f_{p,a,h}(Z_h)\right|
 \leq
 \sum_{h=0}^{H-1}R_{p,a,h}\min\{1,h d_p(a)\}.
\tag{4}
\]

If the score also evaluates the refreshed actor's sampled action at step `h`,
replace `h` by `h+1`.  Equation (4), divided by the score normalization, is the
candidate-specific controlled-kernel term `epsilon_p^ker(a)`.  It is preferable
to a stationary perturbation bound when the launch state carries the switching
signal; it does not introduce a factor `1/(1-rho)`.

## Locally factored critic error

Let the centralized training critic expose owner-local factors

\[
 Q_i(s,u)=\sum_{f:i\in N_f}Q_f(s,u_{N_f})+r_i(s,u),
 \qquad |\{f:i\in N_f\}|\leq \Delta_i,
\tag{5}
\]

where the residual is not assumed to vanish.  Suppose the owner score-function
norm is at most `B_i`, each fitted local factor has uniform error
`epsilon_(Q,f,p)`, and the residual contribution to the owner policy gradient
has norm at most `epsilon_(fac,i,p)`.  For a normalized `H`-step policy-gradient
score, replacing true factors by fitted factors changes the expected packet by
at most

\[
 \epsilon_{p}^{critic}
 \leq B_i\sum_{f:i\in N_f}\epsilon_{Q,f,p}
       +\epsilon_{fac,i,p}.
\tag{6}
\]

For an unnormalised trajectory sum the right-hand side is multiplied by `H`.
If the launch gradient used in the alignment is bounded by `G_F`, the induced
alignment error is at most `G_F epsilon_p^critic`.  This follows directly by
linearity of the score-function estimator, the triangle inequality, and
Cauchy--Schwarz.  Equation (6) is an approximation interface, not a claim that
an unrestricted neural critic has a known uniform radius.

## Predictable plug-in radius and performance consequence

All estimates used at launch must be functions of completed data.  A valid
simultaneous radius for the finite candidate set has the decomposition

\[
 \epsilon_p = \max_a\{
   \epsilon_p^{stat}(a)
  +\epsilon_p^{Markov}(a)
  +\epsilon_p^{ker}(a)
  +G_F\epsilon_p^{critic}(a)
  +\epsilon_p^{receipt}(a)\}.
\tag{7}
\]

The last term contains only error not already charged by the predictable
launch-to-receipt motion `M_p` in the joint Lyapunov index.  The same quantity
must never be counted in both places.

On the simultaneous event

\[
 \max_a|\widehat A_p(a)-A_p(a)|\leq\epsilon_p,
\]

the exact plug-in comparison already proved in
`plugin_lyapunov_oracle_bridge_20260907.md` adds at most
`2 V alpha_max epsilon_p` to the selected one-step Lyapunov index.  Therefore
the discrete finite-time stationarity bound remains valid with cumulative
penalty `2 alpha_max sum_p epsilon_p`.  The useful asymptotic condition is

\[
 N^{-1}\sum_{p<N}\mathbb E\epsilon_p\longrightarrow 0,
\tag{8}
\]

not a per-packet positive lower-confidence requirement.  A hard certificate
may still be reported as an optional conservative mode.

## What is observable without paid sensing

At centralized training time the policy-version ledger contains every current
and cached actor parameter.  For finite categorical actors, the right-hand
side of (1) is computable on the launch observation; for neural actors it can
be upper-bounded on completed replay observations with an explicitly declared
generalization radius.  The local factor critic can evaluate the null and at
most `Delta_i` one-edge counterfactuals on the same completed replay batch.
This consumes computation but no environment transition and no policy-cache
message.  Only the selected refresh is charged as communication.

After the local scores are formed, joint graph--weight selection remains the
closed-form scalar minimization plus an `O(Delta_i)` scan.  The critic work is
also `O(Delta_i)` factor evaluations under (5).  Both actual runtime and memory
must be measured; asymptotic locality alone is not an overhead result.

## Remaining proof gate before Pursuit

Lemmas 1--2 close the controlled-kernel discrepancy that the common-kernel
tabular model does not exercise.  They do not close the neural statistical
radius in (7).  Before a Pursuit efficacy preregistration, an outcome-free
interface must provide either:

1. a finite-dimensional compatible local critic with a stated confidence
   radius under completed Markov data; or
2. an explicit assumption-bound theorem plus empirical calibration coverage,
   with neural approximation retained in the main theorem and measured in the
   benchmark.

The interface must also demonstrate nonzero signed counterfactual scores,
bounded degree, no reward/outcome leakage into graph support, and acceptable
measured overhead.  Structural sparsity and a positive tabular confirmation
alone do not authorize GPU execution.
