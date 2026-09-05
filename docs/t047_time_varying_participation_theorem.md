# T-047 exact risk law for time-varying participation

## Status

T-047 starts a new theorem-aligned ICML research line. It does not amend or
reinterpret EXP-017A, T-020, T-043A, or T-045A. Their configurations, results,
and stop decisions remain unchanged. The new object is a sample-split
participation policy whose learning schedule may vary with the certified
correlation regime and the finite-resource learning phase.

## Model and assumptions

Let the delayed additive recursion be

\[
e_{t+1}=e_t-\eta A e_{t-D}+\eta\bar\xi_t.
\]

The following assumptions define the exact theorem class.

1. **Stable deterministic drift.** The matrix \(A\in\mathbb R^{d\times d}\),
   step size \(\eta>0\), and delay \(D\ge0\) yield a stable lifted companion
   matrix over every evaluated horizon.
2. **Common/private Markov innovations.** There are mutually independent,
   centered, second-order stationary processes \(u_t,v_{1,t},\ldots,v_{M,t}\)
   with common lag covariance
   \(K_k=\mathbb E[u_{t+k}u_t^\top]
        =\mathbb E[v_{i,t+k}v_{i,t}^\top]\).
3. **Prefix participation.** At update \(t\), the learner averages agents
   \(S_t=\{1,\ldots,q_t\}\) and observes
   \[
   \bar\xi_t=\sqrt\rho\,u_t+
   \frac{\sqrt{1-\rho}}{q_t}\sum_{i\in S_t}v_{i,t},
   \qquad \rho\in[0,1].
   \]
4. **Sample-split schedule selection.** The finite schedule
   \(q_{0:T-1}\) is deterministic conditional on probe data that are
   independent of the learning innovations. This prevents a data-dependent
   selection event from being silently treated as fixed.

These assumptions cover the exact additive theorem. Affine temporal-difference
learning requires the separate multiplicative remainder certificate in T-046.

## Exact covariance identity

For any two update times \(s,r\), independence of the private components gives

\[
\operatorname{Cov}(\bar\xi_s,\bar\xi_r)
=\left[\rho+(1-\rho)
\frac{|S_s\cap S_r|}{q_sq_r}\right]K_{s-r}.
\]

For prefix sets, \(|S_s\cap S_r|=\min(q_s,q_r)\), so the coefficient becomes

\[
c_{sr}(\rho)=\rho+\frac{1-\rho}{\max(q_s,q_r)}.
\tag{1}
\]

Let \(G_D\) be the delay companion matrix, \(J\) select the current iterate,
and \(B=J^\top\) inject innovations. With
\(H_{T,s}=JG_D^{T-1-s}B\), direct iteration yields

\[
\mathbb E\|e_T\|_Q^2
=\|JG_D^Tx_0\|_Q^2
+\eta^2\sum_{s,r=0}^{T-1}c_{sr}(\rho)
\operatorname{tr}\!\left(QH_{T,s}K_{s-r}H_{T,r}^\top\right).
\tag{2}
\]

Equation (2) is exact for every deterministic schedule and finite horizon. It
reduces to T-037 for constant \(q_t=q\). At \(\rho=1\), participation does not
change covariance at a fixed horizon; at \(\rho=0\), changing participation
alters both instantaneous variance and cross-time private-stream overlap.

## Affine correlation structure

For a fixed feasible schedule, every \(c_{sr}(\rho)\) is affine in \(\rho\).
Consequently the complete finite-horizon risk has the representation

\[
R_\pi(\rho)=a_\pi+b_\pi\rho,
\tag{3}
\]

where \((a_\pi,b_\pi)\) can be computed offline from two exact evaluations.
This property turns online selection over a frozen schedule library into an
\(O(|\Pi|)\) table scan. Gradient aggregation remains \(O(qd)\); neither step
uses a Hessian inverse, covariance-matrix inverse, or preconditioner.

## Probe-separated selection guarantee

Let \(C_m(q)=c_0+c_1q\) and charge a schedule by
\(\sum_t C_m(q_t)\) message units and \(Tb+D\) environment units. A probe phase
produces a confidence interval \(I=[\rho_L,\rho_U]\). After subtracting the
probe resources, let \(\Pi_I\) contain the remaining-budget schedules and let
\(\pi_0\) be the post-probe fallback. The robust selector chooses a candidate
only if

\[
\max_{\rho\in I}
\{R_\pi(\rho)-R_{\pi_0}(\rho)\}\le-\epsilon.
\tag{4}
\]

Because the difference of two risks is affine, it suffices to check the two
endpoints of \(I\). On the event \(\rho\in I\), (4) guarantees post-probe
improvement of at least \(\epsilon\). The guarantee deliberately does not erase
the opportunity cost already paid by probing. The total controller theorem
must add this cost explicitly, consistent with the T-016--T-017 adaptation
lower bound.

## ICML proof obligations

The following obligations must close before a standard nonlinear benchmark is
authorized.

1. Extend (2) to a tail Polyak--Ruppert readout for a schedule library
   containing every fixed \(q\) and prospectively specified two-stage schedules.
2. Prove a mixing-corrected confidence sequence for the scalar gradient
   covariance probe; state the certified-mixing separation condition.
3. Combine the selection error and probe opportunity cost into a finite-budget
   oracle inequality and match its correlation, mixing, delay, and dual-budget
   dependence to the existing lower bound.
4. Prove or numerically certify that the T-046 multiplicative remainder does
   not reverse the registered schedule ordering on the exact tabular class.
5. Freeze a standard-task benchmark suite and all resource rays before sampled
   outcomes. Benchmark inclusion may use public task structure and theorem
   constants, but not pilot performance.

Until these obligations close, T-047 is a theorem core and implementation
verification, not empirical evidence for a nonlinear controller.
