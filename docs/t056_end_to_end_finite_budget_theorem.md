# T-056 end-to-end finite-budget fingerprint theorem

## Purpose

T-050 gives the stationary participation phase and T-052 gives the exact
two-agent fingerprint decision law. Neither alone is a finite-horizon
learning guarantee. T-056 composes T-048's exact delayed PR risk with T-052
and closes that gap for the registered classify-and-commit controller.

## Model and resource horizon

Let a stationary finite-state Markov trajectory produce centered vector
innovations with lag covariance `K_l`. Prefix trajectory-switch coupling at
correlation `rho` preserves every actor's Markov marginal and gives the fixed-q
cross-time multiplier

\[
 g(q,\rho)=\rho+(1-\rho)/q.
\]

For message/environment budgets `(B_m,B_e)`, probe costs `(C_m,C_e)`, overhead
`h`, and fixed delay `D`, `B_e` and `C_e` count individual actor transitions.
A synchronized round at participation `q` therefore costs `h+q` message units
and `q` environment transitions. Define the number of affordable synchronized
rounds after probing by

\[
 H_q=\min\left\{
 \left\lfloor\frac{B_m-C_m}{h+q}\right\rfloor,
 \left\lfloor\frac{B_e-C_e}{q}\right\rfloor
 \right\}_+,
 \qquad
 N_q=[H_q-D]_+.                                        \tag{1}
\]

Thus the two budgets, integer packing, probe, and `q` actor transitions for
each of the `D` in-flight delay rounds are all charged before learning.  For a
feasible positive-horizon action this is equivalent to

\[
 C_m+(N_q+D)(h+q)\le B_m,
 \qquad
 C_e+(N_q+D)q\le B_e.
\]

## Theorem 1: exact finite risk of each commit action

For the stable delayed linear recursion, T-048 gives the exact mean impulse
and covariance identity for any deterministic schedule. Applying it to the
constant schedule of length `N_q` yields

\[
 R_q^{\rm fin}(\rho)
 =\|\mu_{q,N_q}\|_Q^2+
 \sum_{s,r=0}^{N_q-1}g(q,\rho)
 \operatorname{tr}\!\left(QH_sK_{s-r}H_r^\top\right). \tag{2}
\]

For fixed resources, (2) is affine in `rho`. It contains transient bias,
all Markov lags, the delayed transfer, the tail-averaging window, and the
candidate-specific integer horizon. It is not the T-050 leading coefficient.

## Theorem 2: exact end-to-end classify-and-commit risk

Use `n` independent two-agent fingerprint blocks, path collision probability
`c`, and the public plug-in map `a(k)` from a match count to an action. T-052
gives

\[
 K\sim\operatorname{Binomial}
 \bigl(n,c+(1-c)\rho\bigr).                            \tag{3}
\]

If the probe and learning streams are independent, conditioning on `K`
gives the exact finite-budget identity

\[
 \mathbb E R_{\rm ctrl}^{\rm fin}(\rho)
 =\sum_{k=0}^{n}\binom nk m(\rho)^k[1-m(\rho)]^{n-k}
 R_{a(k)}^{\rm fin}(\rho),                            \tag{4}
\]

where `m(rho)=c+(1-c)rho`. No Gaussian, iid-in-time, asymptotic-horizon, or
perfect-classification replacement occurs in (4).

### Proof

The probe sigma-field is independent of every common/private learning path.
Conditional on `K=k`, the selected action and its charged horizon are
deterministic, so (2) is its conditional expected risk. Multiplying by the
exact masses in (3) and applying total expectation proves (4).

## Corollary 1: finite full-cost comparison

Let the strong fixed baseline `q_0` receive the complete no-probe budgets and
therefore horizon

\[
 H_{q_0}^{0}=\min\left\{
 \left\lfloor\frac{B_m}{h+q_0}\right\rfloor,
 \left\lfloor\frac{B_e}{q_0}\right\rfloor
 \right\},
 \qquad
 N_{q_0}^{0}=[H_{q_0}^{0}-D]_+.
\]

The exact finite ratio is

\[
 \mathcal R_{\rm fin}(\rho)=
 \frac{\mathbb E R_{\rm ctrl}^{\rm fin}(\rho)}
 {R_{q_0}^{\rm fin,0}(\rho)}.                         \tag{5}
\]

Hence `R_fin<=1+epsilon` is a valid finite-budget no-harm certificate, and
`R_fin<=1-delta` is a valid finite-budget gain certificate. Unlike the
leading break-even law, (5) automatically prices probe opportunity cost and
transient bias.

## Corollary 2: relation to the stationary phase

Under the summability and stability conditions of T-050, substituting the
large-horizon expansion of (2) into (4) recovers T-052's expected leading
coefficient and its full-cost ratio. Thus the earlier phase diagram is the
limit of the finite identity, not a separate surrogate model.

## Complexity and claim boundary

Online cost is unchanged: two fingerprints per probe block, one match count,
and an `O(|Q|)` action scan. Equation (4) costs `O(n+|Q|)` once the finite-risk
table is available. A direct general scheduled-risk table can be expensive;
fixed-q Markov recurrences or offline exact covariance computation are
theorem/audit costs, not online controller costs.

The theorem covers delayed linear Markov stochastic approximation and the
registered fixed-policy TD bridge. It does not prove nonlinear actor--critic
or arbitrary cross-agent dependence. Formal experiments must estimate
expected risks with independent master-seed clusters and may not treat cell
rows as independent replicates.

## Accounting correction (2026-08-26)

The original version of (1) and Corollary 1 charged one environment unit per
synchronized server round. The registered nonlinear runner has always
charged `q` individual actor transitions per round. Equations (1) and the
full-budget baseline horizon above now use the same actor-transition unit as
the implementation. The correction changes the theorem-facing horizons and
the standalone T-056 helper; it does not change any previously generated
trajectory or experimental endpoint.
