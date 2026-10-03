# Finite-time guarantee for online Lyapunov participation

## Executable object

At decision block (t\in\{0,\ldots,T-1\}), the learner chooses one
participation level (q_t\in\mathcal Q), performs one MAPPO update, and
evaluates the policy immediately before and after that update on the same
registered validation seed.  The validation stream is disjoint from both the
training stream and the final reporting stream.  If ̅(R_t^-) and
̅(R_t^+) denote the two mean validation returns, define

\[
  \Phi(x)=\frac12-\frac1\pi\arctan(x/s),\qquad
  Y_t=\operatorname{clip}\!\left(
  \frac{\Phi(\bar R_t^-)-\Phi(\bar R_t^+)}{g_0},-1,1\right).
\]

Thus (Y_t>0) means that the selected update reduced bounded validation
risk.  Only (Y_t(q_t)) is observed.  A separate final evaluation seed is
never exposed to the controller.

The fully charged action costs are

\[
c_m(q)=K(h_s+qH_{\rm train})
       +2(h_s+q_vH_v),\qquad
c_e(q)=KH_{\rm train}+2H_v.
\]

The two validation packets, their trajectories, and all (K) selected
training packets are included.  The registered block length (K) amortizes
validation without changing actions inside a block.  Reporting evaluations are measurement operations
shared by every method and do not update the policy or controller.

## Controller

Let (b_j=B_j/T), (j\in\{m,e\}), be the registered per-block allowances.
The implementation stores raw queues

\[
 Q_{j,t+1}=[Q_{j,t}+c_j(q_t)-b_j]_+,
\]

and uses the scaled quadratic Lyapunov function

\[
 L_t=\frac{Q_{m,t}^2}{2b_m}+\frac{Q_{e,t}^2}{2b_e}.
\]

For every action, the controller retains its most recent (W) observations.
With (n_t(q)\) retained observations, registered confidence radius

\[
 \beta_t(q)=\sigma_v
 \sqrt{\frac{2\log(2|\mathcal Q|T^2/\delta)}{n_t(q)}}+\zeta_H,
\]

and empirical mean ̅(Y_t(q)), it selects, except at registered forced
re-probe blocks,

\[
 q_t\in\arg\max_{q\in\mathcal F_t}
 \left\{
   \bar Y_t(q)+\beta_t(q)
   -\frac1V\sum_{j\in\{m,e\}}
       \frac{Q_{j,t}c_j(q)}{b_j}
 \right\}.
\]

​(​\mathcal F_t​) is the hard reserve mask.  It admits an action only
when executing it still leaves enough of both budgets to execute the single
registered reserve action for every remaining block.  The code rejects a
catalogue without an action that jointly minimizes both costs.

## Assumptions

**A1 (predictability and bounded progress).**  The selected action is
measurable with respect to the pre-block history and (Y_t(q)\in[-1,1]).
Within each of (S) latent phases, the conditional mean
​(​\mu_r(q)=\mathbb E[Y_t(q)\mid\mathcal H_t]​) is constant.

**A2 (simultaneous validation certificate).**  With probability at least
(1-\delta), every window that contains observations from only one phase
satisfies

\[
 |\bar Y_t(q)-\mu_r(q)|\leq\beta_t(q).
\]

The theorem below is conditional on this event.  The Markov proposition below
gives sufficient, separately auditable conditions; raw training reward is not
a certificate.

**A3 (registered reserve).**  One action (q_0) jointly minimizes (c_m)
and (c_e), and (Tc_j(q_0)\leq B_j) for both resources.

**A4 (comparator).**  The phase-wise comparator (q_r^*) is available under
the reserve mask on the blocks where it is invoked and obeys
(c_j(q_r^*)\leq b_j).  If this condition is not met, every excluded block is
counted explicitly in (N_{\rm mask}); it is not silently removed.

## Theorem 1: safety and finite-time learning-value regret

Let (N_p) be the number of forced re-probe blocks, and let
​(​\mathcal C​) contain the at most (W(S-1)) blocks whose retained window
crosses a phase boundary.  Define

\[
B_L=\frac12\max_{q\in\mathcal Q}
\left\{
\frac{(c_m(q)-b_m)^2}{b_m}+
\frac{(c_e(q)-b_e)^2}{b_e}
\right\}.
\]

On the event in A2, the executable controller has:

1. **Pathwise physical safety**
   \[
     \sum_{t<T}c_j(q_t)\leq B_j,​\qquad j\in\{m,e\}.
   \]
2. **Finite-time phase-comparator regret**
   \[
   \sum_{t<T}\bigl[\mu_{r(t)}(q_{r(t)}^*)-
                         \mu_{r(t)}(q_t)\bigr]
   \leq
   \frac{TB_L+L_0}{V}
   +\sum_{t\notin\mathcal C}
       [\beta_t(q_t)+\beta_t(q_{r(t)}^*)]
   +2W(S-1)+2N_p+2N_{\rm mask}.
   \]
3. **Risk bridge.**  Let
   (F_H(\theta)=\mathbb E_Z[\Phi(\bar R_H(\theta;Z))]), where every
   registered validation seed (Z_t) is independent of the pre-block
   history and is reused for the before/after pair.  When clipping is
   inactive,
   \[
     \mathbb E[F_H(\theta_T)-F_H(\theta_0)]
       =-g_0\sum_{t<T}\mathbb E[Y_t].
   \]
   With clipping, the equality has an explicit residual equal to the sum of
   clipped tails.  Thus the optimized quantity is the expected finite-horizon
   validation risk used by the controller, not a detached resource proxy.

### Proof

The reserve statement follows by backward induction.  Before the last block,
every admitted action leaves enough budget for (T-t-1) copies of (q_0);
after charging the admitted action, the same invariant holds for the next
block.  At (t=T-1), nonnegative remaining budgets follow directly.

Using ([x]_+^2\leq x^2), the scaled one-step drift obeys

\[
L_{t+1}-L_t\leq B_L+
\sum_j\frac{Q_{j,t}}{b_j}[c_j(q_t)-b_j].
\]

On a clean-window, non-probe, non-mask block, optimism and the controller's
argmax imply

\[
\mu_r(q_r^*)-\mu_r(q_t)
\leq \beta_t(q_t)+\beta_t(q_r^*)+\frac1V
\sum_j\frac{Q_{j,t}}{b_j}
[c_j(q_t)-c_j(q_r^*)].
\]

By A4, the comparator's queue term relative to the allowance is nonpositive.
Substitution into the drift inequality, summation, and telescoping of (L_t)
give the first two terms of the regret bound.  Progress lies in ([-1,1]), so
each contaminated, forced-probe, or comparator-masked block contributes at
most two; their registered counts give the remaining terms.  Conditional
independence of each fresh validation seed makes the paired observation
unbiased for (F_H(\theta_t)-F_H(\theta_{t+1})); these population
differences telescope in expectation.  □

This is a finite-time theorem.  It does not claim a vanishing average regret
for a fixed (W).  Choosing (W) exposes the standard nonstationary tradeoff:
confidence cost scales as (T/\sqrt W), whereas switch contamination scales
as (SW).

## Proposition 2: a sufficient Markov validation certificate

Suppose that during each validation rollout the policy is frozen, per-step
team reward is in ([-R,R]), and every registered policy induces a Markov
chain satisfying the uniform total-variation bound

\[
 \sup_x\|P_\pi^k(x,\cdot)-d_\pi\|_{\rm TV}\leq C\rho^k,
 \qquad 0\leq\rho<1.
\]

Then the finite-horizon initialization bias of one mean return is at most

\[
 \epsilon_H\leq\frac{2RC}{H(1-\rho)}.
\]

Because (Phi) is (1/(\pi s))-Lipschitz and one progress observation is a
paired before/after difference, a valid registered bias term is

\[
 \zeta_H=\frac{4RC}{\pi s g_0H(1-\rho)}.
\]

A standard concentration inequality for uniformly geometrically mixing
bounded additive functionals can sharpen ​(sigma_v) using the corresponding
effective sample size.  For the finite-horizon risk ​(F_H) itself, however,
the clipped paired observation is already bounded in ​([-1,1]); fresh
validation seeds therefore give a distribution-free martingale confidence
sequence across blocks.  The mixing envelope is needed only when interpreting
​(F_H) as an approximation to stationary-return risk.  A union bound over all
registered adaptive observation endpoints then gives A2.

This proposition is deliberately conditional on a uniform mixing envelope.
The experiments must register (sigma_v) and (zeta_H); estimating them from
final evaluation returns would violate the outcome-isolation design.

## Complexity

The controller stores at most (W|\mathcal Q|) scalars and scans the catalogue
once per block, hence it uses (O(W|\mathcal Q|)) memory and
(O(|\mathcal Q|)) arithmetic.  It computes no Hessian, covariance matrix,
counterfactual gradient, or offline QP.  MAPPO remains the underlying CTDE
learner; only the number of contemporaneous rollout workers is selected
online.
