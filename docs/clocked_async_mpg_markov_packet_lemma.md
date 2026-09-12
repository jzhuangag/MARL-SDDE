# Fixed-horizon Markov packet interface

Status: theorem-facing episodic sampling lemma.  It closes the first bounded
Markov-gradient interface, not the full asynchronous Markov-potential-game
theorem.

## Packet model

Consider a finite discounted Markov potential game with factorized stationary
joint policy

\[
\pi_\theta(a\mid s)=\prod_{i=1}^n\pi_{i,\theta_i}(a_i\mid s),
\qquad 0<\gamma<1.
\]

A packet for agent `i` is born from a versioned joint policy `theta^b`.  It
uses a fresh environment replica, draws the initial state from the declared
distribution `rho`, and runs exactly `H` joint transitions.  The packet is
fully charged for all `H` transitions.  Its completion metadata may depend on
agent and declared workload, but is conditionally independent of the realized
trajectory and gradient innovation.  Different in-flight replicas are
conditionally independent at birth; trajectories within a packet are Markov,
not iid.

For bounded agent reward `|r_i| <= R_i`, define the absolute-discount
REINFORCE block packet

\[
\widehat g_{i,H}(\Xi_b)
=\sum_{t=0}^{H-1}
\nabla_{\theta_i}\log\pi_{i,\theta_i^b}(a_{i,t}\mid s_t)
\left(\sum_{\ell=t}^{H-1}\gamma^\ell r_i(s_\ell,a_\ell)\right).
\tag{1}
\]

The use of absolute discount in the inner return is important: it makes (1)
the score gradient of the objective whose reward at global trajectory time
`ell` is weighted by `gamma^ell`.

## Lemma 1: exact conditional mean

Let

\[
J_{i,H}(\theta)=\mathbb E_\rho^\theta
\left[\sum_{t=0}^{H-1}\gamma^t r_i(s_t,a_t)\right].
\]

Conditionally on the birth policy and pre-trajectory history,

\[
\mathbb E[\widehat g_{i,H}(\Xi_b)\mid\mathcal F_b]
=\nabla_i J_{i,H}(\theta^b). \tag{2}
\]

This follows by differentiating the finite trajectory likelihood.  The Markov
transition kernel contains no policy parameter; every action-score occurrence
at time `t` multiplies exactly the downstream discounted rewards in (1).
No within-trajectory independence is used.

If the game has an exact policy-level Markov potential `Phi`, then unilateral
policy differences imply
`nabla_i J_i(theta)=nabla_i Phi(theta)`.  The identical-interest case checked
by the implementation is a strict subclass.

## Lemma 2: truncation bias and packet norm

Assume the policy score satisfies

\[
\|\nabla_{\theta_i}\log\pi_{i,\theta_i}(a_i\mid s)\|\le G_i.
\]

The bias relative to the infinite discounted gradient is bounded by

\[
\begin{aligned}
\beta_{i,H}
&:=\|\nabla_iJ_i(\theta)-\nabla_iJ_{i,H}(\theta)\|\\
&\le G_iR_i\gamma^H
\left(\frac{H}{1-\gamma}+\frac{1}{(1-\gamma)^2}\right). \tag{3}
\end{aligned}
\]

The first term covers tail rewards attached to the first `H` scores; the
second covers all score occurrences after time `H`.  A pathwise packet bound is

\[
\|\widehat g_{i,H}\|
\le C_{i,H}:=
\frac{G_iR_i}{1-\gamma}
\left(\frac{1-\gamma^H}{1-\gamma}-H\gamma^H\right)
\le\frac{G_iR_i}{(1-\gamma)^2}. \tag{4}
\]

Thus, at the birth filtration,

\[
\widehat g_{i,H}
=\nabla_iJ_i(\theta^b)+B_{i,H}(\theta^b)+\xi_{i,b},
\quad
\|B_{i,H}\|\le\beta_{i,H},
\quad
\mathbb E[\xi_{i,b}\mid\mathcal F_b]=0,
\quad
\mathbb E\|\xi_{i,b}\|^2\le C_{i,H}^2. \tag{5}
\]

Under the stated non-informative completion assumption and independent
environment replicas, the centered property persists after conditioning on
the pre-application sigma-field and predictable packet metadata.  It does not
persist under outcome-dependent first-arrival selection.

## Bias-aware Lyapunov insertion

Write `f=-Phi`.  The scalar-block derivation in
`clocked_async_mpg_exact_gradient_lemma.md` extends verbatim to Euclidean vector
blocks when the scalar absolute values are replaced by block norms and each
`L_ij` is a valid operator cross-smoothness bound.

For any `delta>0`, inflate the history coefficient to

\[
\mathcal V_k=f(\theta^k)
+\frac{\alpha D(1+\delta)}2 H_k.
\]

If

\[
L_{ii}\alpha+(1+\delta)D^2w_i\alpha^2\le1
\quad\text{for every }i, \tag{6}
\]

Young's inequality applied to the truncation bias gives the conditional drift

\[
\begin{aligned}
\mathbb E[\mathcal V_{k+1}\mid\mathcal G_k^-]
\le{}&\mathcal V_k
-\frac\alpha2\sum_i p_i\|\nabla_i f(\theta^k)\|^2\\
&+\frac\alpha2(1+\delta^{-1})\sum_i p_i\beta_{i,H}^2\\
&+\frac12\sum_i p_i
\left(L_{ii}\alpha^2+(1+\delta)D^2w_i\alpha^3\right)
C_{i,H}^2. \tag{7}
\end{aligned}
\]

Consequently,

\[
\begin{aligned}
\frac1K\sum_{k=0}^{K-1}\mathbb E\|\nabla f(\theta^k)\|^2
\le{}&\frac{2(\mathcal V_0-f_{\inf})}{\alpha p_{\min}K}\\
&+\frac{1+\delta^{-1}}{p_{\min}}\sum_i p_i\beta_{i,H}^2\\
&+\frac1{p_{\min}}\sum_i p_i
\left(L_{ii}\alpha+(1+\delta)D^2w_i\alpha^2\right)C_{i,H}^2.
\tag{8}
\end{aligned}
\]

Equation (8) is finite and nonvacuous: choosing `alpha` on the order of
`K^(-1/2)` and increasing `H` logarithmically makes both the optimization/noise
term and squared truncation bias vanish.  It is still an event-count
stationarity result.  It does not yet establish a Nash gap or elapsed-time
advantage.

## Exact computational validation

The validation code evaluates a two-agent factorized softmax policy in a
finite Markov game by dynamic programming and by complete trajectory
enumeration.  Six tests establish:

- finite- and infinite-horizon policy gradients match central finite
  differences;
- the enumerated expectation of (1) matches the finite-horizon gradient;
- (3) covers the exact finite/infinite gradient gap in random finite games;
- (4) is finite and below its horizon-independent envelope;
- malformed transition kernels and horizons are rejected.

These are algebraic tests, not experimental evidence.

## Remaining gate

The initial episodic Markov packet interface is now closed under explicit,
restrictive assumptions.  The paper candidate still stops unless all of the
following are proved for the same update rule:

1. vector-block drift with heterogeneous local steps or a justified common
   step;
2. a potential-stationarity to standard Nash-gap conversion;
3. a renewal/clock theorem consistent with packet birth, service and bounded
   event staleness;
4. a nonempty fully charged speedup region against a strong synchronous and a
   strong asynchronous comparator;
5. novelty beyond composing generic delayed coordinate descent with a known
   synchronous MPG gradient-domination lemma.

Continuing-chain sampling, learned critics and trajectory-dependent completion
remain outside this first theorem.  They may be extensions only after the
bounded theorem closes; they cannot be claimed now.

## Conditional softmax stationarity-to-Nash conversion

For completeness, the infinite-horizon softmax gradient can be converted to a
unilateral Nash gap without treating a stationary point as automatically Nash.
Let `d_rho^pi` be the normalized discounted state occupancy and let
`pi_i^BR` be an exact best response to `pi_-i`.  Define

\[
M_i(\pi)=\left\|
\frac{d_\rho^{(\pi_i^{\rm BR},\pi_{-i})}}
     {d_\rho^\pi}
\right\|_\infty,
\qquad
\underline\pi_i=\min_{s,a_i}\pi_i(a_i\mid s).
\]

The performance-difference identity and the softmax policy-gradient formula
give

\[
\operatorname{Gap}_i(\pi)
\le
\frac{M_i(\pi)\sqrt{|\mathcal S|}}{\underline\pi_i}
\|\nabla_iJ_i(\theta)\|_2. \tag{9}
\]

Indeed, the logit-gradient component is

\[
\nabla_{\theta_i(s,a_i)}J_i
=\frac{d_\rho^\pi(s)}{1-\gamma}
\pi_i(a_i\mid s)\,\overline A_i^\pi(s,a_i).
\]

Substitution into the performance-difference identity cancels both
`d_rho^pi(s)` and `1-gamma`; summing the largest absolute logit component over
states and applying Cauchy--Schwarz proves (9).

If, along the analyzed trajectory, `M_i(pi)<=M_bar` and
`pi_i(a_i|s)>=pi_bar>0`, then for a uniformly sampled event iterate `R`,

\[
\mathbb E\max_i\operatorname{Gap}_i(\pi_{\theta^R})
\le
\frac{\overline M\sqrt{|\mathcal S|}}{\underline\pi}
\sqrt{\frac1K\sum_{k=0}^{K-1}
\mathbb E\|\nabla\Phi(\theta^k)\|_2^2}. \tag{10}
\]

This is a valid conditional Nash conversion, not a uniform global guarantee.
Plain softmax updates do not themselves ensure a horizon-independent
`pi_bar`; the constant can become vacuous under probability collapse.  The
project will not silently add a projection or barrier regularizer because that
would change the executable update and reopen the Lyapunov drift.  A final
paper may either state the interiority/mismatch assumptions explicitly or use
first-order potential stationarity as its primary criterion.  Claiming an
unconditional Nash rate is not authorized.

The exact validator solves each agent's best-response MDP, computes its
discounted occupancy mismatch and confirms (9) on 20 random two-agent finite
Markov games.  This checks the algebra; it does not establish uniform coverage
for learned neural policies.
