# Finite-Budget Lyapunov Learning-Progress Controller

## 1. Model and decision variable

Let `Q` be a finite public catalogue of participation levels.  At decision
time, action `q` consumes `c_m(q)` messages and `c_e(q)` environment ticks per
learning update.  After charging the calibration phase and delay, the exact
feasible horizon is

\[
N_B(q)=\min\!\left\{
\left\lfloor B_m^{\rm rem}/c_m(q)\right\rfloor,
\left\lfloor B_e^{\rm rem}/c_e(q)\right\rfloor
\right\}.
\]

For a nonnegative learning Lyapunov function `V`, assume that a fixed action
obeys the conditional affine drift inequality

\[
\mathbb E[V_{k+1}\mid\mathcal F_k,q]
\le a_q V_k+c_q,
\qquad 0\le a_q<1,\quad c_q\ge0.
\tag{1}
\]

Both coefficients are task dependent.  Cross-worker correlation affects their
confidence radii, but it does not determine them.  This separates statistical
redundancy from the learning efficiency of a participation level.

Iterating (1) for the feasible horizon gives the terminal-risk certificate

\[
R_B(q;a_q,c_q,V_0)
=a_q^{N_B(q)}V_0
+c_q\frac{1-a_q^{N_B(q)}}{1-a_q}.
\tag{2}
\]

The controller selects `q`; the learning algorithm, policy architecture, and
physical budgets remain fixed.

## 2. Fully charged calibration

Each candidate is initialized from the same registered checkpoint and receives
`m` short micro-training blocks.  A disjoint validation block produces paired
observations `(X_{q,s},Y_{q,s})`, representing the Lyapunov values before and
after the candidate update.  These branches are part of the algorithm: all
rollouts and messages are removed from the remaining budgets, and only the
selected branch is retained.

For every candidate, calibration returns simultaneous intervals

\[
a_q\le \bar a_q,\qquad c_q\le\bar c_q
\tag{3}
\]

with joint probability at least `1-delta`.  The radii use the registered
Markov-mixing bound and the effective sample size induced by the certified
cross-worker dependence.  Evaluation return is not required; a valid
Lyapunov observation can be a held-out squared temporal-difference residual or
a performance-difference lower-bound deficit.

Define

\[
\bar R_B(q)=R_B(q;\bar a_q,\bar c_q,V_0),
\qquad
\hat q\in\arg\min_{q\in\mathcal Q}\bar R_B(q).
\tag{4}
\]

Equation (4), rather than a hand-tuned mixture of correlation and reward, is
the TSP-V2 control rule.

## 3. Deterministic comparison lemma

**Lemma 1 (monotonicity).**  For an integer `N>=1`, the function
`R_N(a,c,V)=a^N V+c(1-a^N)/(1-a)` is nondecreasing in `a`, `c`, and `V` on
`0<=a<1`, `c>=0`, and `V>=0`.

**Proof.**  Write

\[
R_N(a,c,V)=a^N V+c\sum_{j=0}^{N-1}a^j.
\]

Every coefficient is nonnegative, and every monomial is nondecreasing on the
declared domain.  This proves the claim.  `square`

## 4. Main finite-budget guarantee

**Theorem 1 (simultaneous-certificate terminal safety).**  Suppose (1) holds
for every candidate throughout its post-calibration horizon, the calibration
cost is charged before `N_B(q)` is computed, and (3) holds simultaneously.
Then, on the joint confidence event,

\[
\mathbb E[V_{N_B(\hat q)}\mid\mathcal F_0]
\le \bar R_B(\hat q)
=\min_{q\in\mathcal Q}\bar R_B(q).
\tag{5}
\]

Moreover, if the certificate excess satisfies

\[
0\le \bar R_B(q)-R_B(q;a_q,c_q,V_0)\le\varepsilon_q,
\tag{6}
\]

then

\[
R_B(\hat q;a_{\hat q},c_{\hat q},V_0)
\le \min_{q\in\mathcal Q}R_B(q;a_q,c_q,V_0)
+\varepsilon_{q^\star},
\tag{7}
\]

where `q^star` is a true catalogue minimizer.

**Proof.**  Iterating (1) yields (2).  Lemma 1 and (3) upper-bound the true
terminal risk of every candidate by its certified risk.  The first equality in
(5) follows from the definition of `hat q`.  For (7),

\[
R_B(\hat q)\le\bar R_B(\hat q)
\le\bar R_B(q^\star)
\le R_B(q^\star)+\varepsilon_{q^\star}.
\]

`square`

The result is pathwise with respect to the selected action on the simultaneous
confidence event.  It does not infer terminal performance from correlation
alone: the action-specific drift coefficients are indispensable.

## 5. Separation condition and identification

**Corollary 1 (correct interior-action recovery).**  If a unique oracle action
`q^star` satisfies

\[
R_B(q;a_q,c_q,V_0)-R_B(q^\star;a_{q^\star},c_{q^\star},V_0)
>\varepsilon_q+\varepsilon_{q^\star}
\]

for every `q != q^star`, then (4) selects `q^star` on the confidence event.

This condition admits an interior optimum.  For example, `q=2` can dominate
`q=1` through a smaller contraction factor while dominating `q=8` through a
longer feasible horizon.  A dependence-only score cannot express this case.

## 6. Outstanding proof interface before a manuscript claim

Theorem 1 is complete for valid simultaneous coefficient certificates.  The
remaining domain-specific obligation is to prove that the chosen observable
calibration statistic supplies (3) under the exact MARL sampling and update
mechanism.  TSP-V2 therefore does not yet claim that arbitrary PPO training
rewards or in-sample critic losses are such certificates.  The GPU interface
test must use a disjoint validation statistic and must precede any return-based
development experiment.

