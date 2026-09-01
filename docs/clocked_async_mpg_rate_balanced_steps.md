# Rate-balanced local steps and the slow-block boundary

Status: closed event-time allocation lemma; wall-clock process and matching
game lower bound remain open.

## Why a common step is the wrong clock

With agent `i` activated at an event with probability `p_i`, the common-step
drift controls

\[
\sum_i p_i\|\nabla_i f\|^2
\ge p_{\min}\|\nabla f\|^2.
\]

The resulting `p_min` is not just a loose proof artifact.  A rare essential
policy block cannot reach a full-gradient or Nash criterion without receiving
updates.  Nevertheless, forcing every block to use the same step needlessly
adds another imbalance.  The local steps below equalize the expected descent
weight without scanning a catalogue or solving a QP.

## Heterogeneous-step drift

Let block `i` use step `alpha_i`.  Define

\[
\ell_i=\sum_jL_{ij},
\qquad
\widetilde w_j=\sum_i p_i\alpha_i\ell_iL_{ij},
\]

and let `H_tilde_k` be the Krasovskii history with block weights
`w_tilde_j`.  For a bias-splitting constant `delta>0`, use

\[
\mathcal V_k=f(\theta^k)+\frac{D(1+\delta)}2\widetilde H_k.
\]

The same one-event proof closes if

\[
L_{ii}\alpha_i+(1+\delta)D^2\widetilde w_i\alpha_i\le1
\quad\text{for every }i. \tag{1}
\]

For exact stale gradients this gives

\[
\mathbb E[\mathcal V_{k+1}\mid\mathcal G_k^-]
\le\mathcal V_k-rac12\sum_i p_i\alpha_i
\|\nabla_i f(\theta^k)\|^2. \tag{2}
\]

Conditional packet bias and noise add the explicit terms

\[
\begin{aligned}
&\frac{1+\delta^{-1}}2\sum_i p_i\alpha_i\beta_i^2\\
&\quad+\frac12\sum_i p_i
\left(L_{ii}\alpha_i^2
 +(1+\delta)D^2\widetilde w_i\alpha_i^2\right)C_i^2. \tag{3}
\end{aligned}
\]

No independence between coordinates is used; `L_ij` carries the interaction.

## Closed-form rate balancing

Impose

\[
p_i\alpha_i=c. \tag{4}
\]

Let

\[
u_i=\sum_\ell\ell_\ell L_{\ell i}.
\]

Then `w_tilde_i=c u_i`, and (1) becomes a scalar quadratic:

\[
\frac{L_{ii}}{p_i}c
+\frac{(1+\delta)D^2u_i}{p_i}c^2\le1. \tag{5}
\]

Define `a_i=L_ii/p_i`, `b_i=(1+delta)D^2u_i/p_i` and

\[
c_i=
\begin{cases}
\dfrac{-a_i+\sqrt{a_i^2+4b_i}}{2b_i},&b_i>0,\\
1/a_i,&b_i=0,\ a_i>0,\\
+\infty,&a_i=b_i=0.
\end{cases}
\]

The maximal rate-balanced scale and local steps are

\[
c^\star=\min_i c_i,
\qquad
\alpha_i=\frac{c^\star}{p_i}. \tag{6}
\]

They require `O(n^2)` arithmetic to form `u`, followed by `O(n)` scalar roots.
There is no Hessian, covariance inverse, participation scan, finite step
catalogue or numerical optimization.  With (6), (2) becomes

\[
\mathbb E[\mathcal V_{k+1}\mid\mathcal G_k^-]
\le\mathcal V_k-\frac{c^\star}{2}\|\nabla f(\theta^k)\|^2. \tag{7}
\]

This is the strongest common full-gradient coefficient within the restricted
rate-balanced family.  It is not claimed to solve the unrestricted step
allocation problem.

## Wall-clock interpretation and unavoidable limit

If an ideal exogenous marked event process has total completion rate `Lambda`
and mark probabilities `p_i=lambda_i/Lambda`, the event-time coefficient maps
formally to `Lambda*c_star` per expected unit time.  At zero event delay,

\[
c^\star=\min_i\frac{p_i}{L_{ii}},
\qquad
\Lambda c^\star=\min_i\frac{\lambda_i}{L_{ii}}. \tag{8}
\]

Thus a full-gradient/Nash theorem cannot generally scale only with aggregate
compute.  On a separable game, an essential block cannot improve before its
agent returns a packet, giving an immediate first-completion lower bound of
order `1/lambda_i`; repeated noisy estimation strengthens this to the number
of required local packets divided by `lambda_i`.

The defensible paper-level question is therefore not “does asynchrony erase
the slowest agent?”  It is:

> When does barrier removal beat its interaction-weighted delay cost, after
> respecting the unavoidable rate of every essential policy block?

The positive region can still be nonempty.  Synchronous rounds repeatedly pay
the maximum of heterogeneous service times, whereas asynchronous learning
need only satisfy the essential-block rate and the coupling-delay stability
condition.  Dense interaction or large event delay can eliminate the gain.
That rate--coupling transition is the prospective phase theorem.

## What is and is not closed

The implementation validates (5)--(7) on random coupled positive-semidefinite
quadratics with nonuniform activation probabilities and delays.  It verifies
that `p_i*alpha_i` is constant, at least one stability constraint is tight and
the exact expected Lyapunov drift lies below (7).

Still open:

- a physically consistent renewal process connecting packet birth, completion
  rate and event staleness;
- a strong synchronous comparator that charges useful work performed while
  waiting, rather than assuming all fast agents idle;
- a matching lower/separation family with stochastic Markov packets;
- online estimation of completion rates without invalidating predictability;
- Nash-gap conversion and a standard MARL confirmation.

Therefore (6) is a theorem-facing candidate, not an authorized efficacy
algorithm.
