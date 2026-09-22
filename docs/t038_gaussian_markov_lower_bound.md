# T-038 predictable Gaussian Markov lower bound

## Result

T-038 extends the old iid, single-cost Gaussian location lower bound to
temporally correlated common noise, irregular participation/stride actions,
predictable randomized selection, delay, and separate message/environment
budgets.  It gives an exact finite-catalogue dynamic-program value, not an
iid action-count surrogate.

## Gaussian Markov subclass

Let the unknown scalar target be \(\theta\).  At observation time \(S_t\),

\[
X_{i,t}=\theta+C_{S_t}+\epsilon_{i,t},\qquad i=1,\ldots,q_t,
\]

where \(C\) is a stationary Gaussian AR(1) process with variance
\(\sigma_c^2\) and coefficient \(\lambda\), while
\(\epsilon_{i,t}\sim N(0,\sigma_e^2)\) are private and fresh.  The action
\(a_t=(q_t,b_t)\), selected from prior observations, sets
\(S_t-S_{t-1}=b_t\).  Spatial rotation leaves the sufficient scalar

\[
Y_t=\sqrt{q_t}(\theta+C_{S_t})+E_t,qquad
E_t\sim N(0,\sigma_e^2).
\tag{1}
\]

## Deterministic covariance state

Place the proper prior \(\theta\sim N(0,\tau^2)\) and define the hidden state
\(u_t=(\theta,C_{S_t})\).  For a realized action sequence, its predicted
covariance follows

\[
P_t^-=F_tP_{t-1}^+F_t^\top+Q_t,
\quad
F_t=\operatorname{diag}(1,\lambda^{b_t}),
\quad
Q_t=\operatorname{diag}(0,(1-\lambda^{2b_t})\sigma_c^2),
\]

and the scalar Kalman update uses
\(H_t=\sqrt{q_t}(1,1)\) and noise variance \(\sigma_e^2\).  The first
observation starts from \(P_0^-=\operatorname{diag}(\tau^2,\sigma_c^2)\).
Crucially, \(P_t^+\) depends on the actions but not the observed values.

## Theorem: exact predictable dual-budget minimax value

Let \(\Pi_B\) contain every randomized predictable policy stopped under the
pathwise constraints

\[
\sum_{t\le N}(h+q_t)\le B_m,
\qquad
\sum_{t\le N}b_t+D\mathbf1\{N>0\}\le B_e.
\tag{2}
\]

Let \({\cal S}_B\) be all deterministic finite action sequences satisfying
(2), and write \(V_\tau(a_{1:n})=[P_n^+(a_{1:n})]_{11}\).  Then

\[
\inf_{\pi\in\Pi_B,\widehat\theta}
\sup_{\theta\in\mathbb R}
\mathbb E_\theta^\pi(\widehat\theta-\theta)^2
 =
\min_{a_{1:n}\in{\cal S}_B} I(a_{1:n})^{-1}
 =\lim_{\tau\to\infty}
\min_{a_{1:n}\in{\cal S}_B}V_\tau(a_{1:n}),
\tag{3}
\]

The right side retains \(q,b,\lambda,\sigma_c^2,\sigma_e^2,h,B_m,B_e,D\)
and is computable by deterministic covariance-state dynamic programming.

### Proof

Factor the stopped experiment into predictable action kernels and Gaussian
observation kernels.  The action factors are identical for every \(\theta\)
and cancel from the posterior likelihood.  Conditional on any realized
action/data history, linear-Gaussian conjugacy gives posterior variance
\(V_\tau(a_{1:N})\); data-dependent selection changes which covariance path
is realized but does not change the Riccati update on that path.  The Bayes
risk of every policy is therefore

\[
\mathbb E[V_\tau(A_{1:N})]
\ge\min_{a_{1:n}\in{\cal S}_B}V_\tau(a_{1:n}).
\]

Every realized sequence is in \({\cal S}_B\) by pathwise feasibility.  Bayes
risk lower-bounds minimax risk.  Monotonicity in \(\tau^2\) and the diffuse
prior limit prove the lower inequality in (3).  Conversely, choose a
deterministic sequence maximizing \(I(a_{1:n})\) and apply generalized least
squares.  Its risk is \(I(a_{1:n})^{-1}\) for every \(\theta\), proving the
matching upper inequality.  Data-dependent participation cannot improve the
minimax value in this subclass, although a nonconstant deterministic
participation/stride schedule may be optimal.

## Fixed-sequence closed form

For a fixed sequence, define observation times from the strides,
\(r_t=\sqrt{q_t}\), and

\[
\Sigma_{st}=\sigma_c^2r_sr_t\lambda^{|S_s-S_t|}
             +\sigma_e^2\mathbf1\{s=t\}.
\]

The Fisher information is \(I=r^\top\Sigma^{-1}r\), so the diffuse-prior
risk is exactly \(I^{-1}\).  This independently verifies the Kalman recursion
and shows explicitly how larger \(q\) saturates under common noise while
larger stride can decorrelate observations at environment cost.

## What is and is not matched

T-038 closes P3 exactly for a Gaussian Markov location subclass with
predictable actions and the same dual-resource/delay convention as T-037. It proves
that correlation, mixing, participation, stride, delay, and both budgets
cannot all be removed from a broader theorem.

The equality is for the optimal estimator/design, not automatically for the
constant-step T-037 stochastic-approximation iterate.  A comparison between
that iterate and generalized least squares remains a separate algorithmic
efficiency question; it is not needed for the exact subclass minimax value.
