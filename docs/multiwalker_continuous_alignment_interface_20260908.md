## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theorem--implementation interface audit
- Origin Date: 2026-09-08
- Verification Status: ALGEBRA AND COMPLEXITY VERIFIED; LEARNING HEADROOM NOT ESTABLISHED
- Version Label: multiwalker_continuous_alignment_interface_v1

# Continuous-action factor alignment for Multiwalker

## Purpose and boundary

This note closes one structural gap between the topology-robust Lyapunov rule
and a continuous-action MARL task.  It proves how to score the null action and
every one-edge cache refresh without one actor backward pass per candidate.
It does not establish that the score is statistically calibrated, that a
refresh improves Multiwalker return, or that a learned controller beats a
strong scheduler.

## Fixed launch-time objects

At an owner launch, let the deterministic actor be

\[
 u_i=\mu_{\theta_i}(o_i)\in[-1,1]^{d_a}
\]

and let `v_i` be a launch-measurable parameter-space reference direction.  A
typical later choice is a normalized descent direction formed only from
completed packets.  Define its induced action direction once:

\[
 s_i=J_{\theta_i}\mu_{\theta_i}(o_i)v_i\in\mathbb R^{d_a}.
\]

The centralized local critic is declared to factor over the fixed physical
neighbor universe,

\[
 Q_i(o,u_i;\chi_i)
 =q_i(o_i,u_i)+
 \sum_{j\in\mathcal N_i}
 \psi_{ij}(o_i,o_j,u_i,z_{j\to i}),
\]

where `z_(j->i)` is a policy profile computed from the version of teammate
`j` cached at recipient `i`.  For candidate `a=j`, only that profile is
replaced by its current version; `a=empty` retains every cached profile.

The declared signed score is

\[
 \widehat d_i(a)=
 s_i^\top\nabla_{u_i}Q_i(o,u_i;\chi_i^a).
\]

For this declared factor critic, the null and every one-edge substitution are
exact:

\[
\begin{aligned}
 g_i^0
 &=\nabla_{u_i}q_i+
   \sum_j\nabla_{u_i}\psi_{ij}(z_{j\to i}^{\rm cache}),\\
 g_i^{(j)}
 &=g_i^0-
   \nabla_{u_i}\psi_{ij}(z_{j\to i}^{\rm cache})+
   \nabla_{u_i}\psi_{ij}(z_j^{\rm current}),\\
 \widehat d_i(a)&=s_i^\top g_i^a.
\end{aligned}
\]

No topology-cache energy is used in this identity.  The physical neighbor
universe is fixed for the primary Multiwalker theorem; the selected directed
cache edge may change on every launch.  If an extension changes the factor
universe or its weights, its topology-motion remainder must be added
explicitly.

## Lyapunov insertion

The score is an observable input, not a second optimizer.  With a simultaneous
error allowance `epsilon_i(a)`, the conservative learning coefficient is
formed with the sign convention of the core theorem, for example

\[
 \underline A_i(a)=\widehat A_i(a)-\epsilon_i(a),
 \qquad
 m_i(a)=V[\underline A_i(a)-L_iG_i(a)M_i].
\]

The same topology-robust drift index then jointly selects the cache edge and
the receipt-time packet weight:

\[
 \widehat\alpha_i(a)=
 \Pi_{[0,\bar\alpha]}
 \frac{[m_i(a)]_+}{C_i(a)},
 \qquad
 a_i\in\arg\min_{a\in\{\emptyset\}\cup\mathcal N_i}
 \left[-m_i(a)\widehat\alpha_i(a)
 +\frac{C_i(a)}2\widehat\alpha_i(a)^2+Q_ic_i(a)\right].
\]

Thus Lyapunov drift remains the design principle: it decides both which
teammate version enters the next trajectory and how strongly the returned
owner packet is applied.  The present audit verifies the factor score that
feeds this decision; the statistical error allowance and performance theorem
remain separate obligations.

## Complexity

For a deterministic actor, `s_i` needs exactly `d_a` reverse-mode scalar
gradients, independent of the number of neighbors.  The current transparent
implementation uses one self-factor action gradient and cached/current action
gradients for each neighbor, namely `1+2 Delta_i` critic calls.  Candidate
assembly is vector arithmetic.  Multiwalker has `d_a=4` and chain degree at
most two, so the candidate work is constant at fixed task size and is
`O(Delta_i)` as the number of walkers grows.

A Gaussian-policy extension can use a fixed set of `K` reparameterization
samples, giving `O(K Delta_i)` candidate work.  This extension is not part of
the verified interface until its estimator bias and common-random-number
coupling are specified.

## Verification

`continuous_factor_alignment.py` is compared against a centered
parameter-space finite difference for the null action and every donor.  The
double-precision discrepancy is below `5e-8`.  Separate degree tests verify
four actor reverse calls for degrees one, two, and four, while critic gradient
calls equal `1+2 Delta`.

Together with the outcome-free cache contract, this result authorizes only a
separately frozen equal-resource oracle-headroom experiment.  It does not
authorize controller fitting, a standard-task efficacy claim, formal seeds,
GPU, or HPC4.
