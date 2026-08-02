# T-034 ICML 2027 main line

## Publication thesis

### Working title

> **Beyond Linear Speedup: Sharp Limits of Parallel Markov Learning under
> Correlation and Delay**

### Central question

Under fixed resources, when do parallel Markov streams yield linear speedup,
when does cross-agent dependence cause saturation, when do delay and reduced
update horizons reverse the benefit, and can a learner identify the correct
regime before its budget is exhausted?

This question directly strengthens independent-agent linear-speedup theory in
[Khodadadian et al.](https://proceedings.mlr.press/v162/khodadadian22a.html)
and complements delayed Markov stochastic approximation in
[Adibi et al.](https://proceedings.mlr.press/v238/adibi24a.html).  The new
object is their interaction: long-run cross-agent covariance changes the
statistical value of parallelism, while delay and resource accounting change
the number and age of usable updates.

### Candidate abstract

Parallel reinforcement learning analyses commonly equate the number of agents
with the number of independent samples.  We characterize how this conclusion
changes when agents observe temporally mixing Markov streams with shared
randomness and delayed updates.  For affine stochastic approximation and
linear temporal-difference learning, we derive a finite-horizon risk
decomposition whose stochastic term is governed by the long-run covariance of
the aggregated innovations.  The resulting effective-agent quantity yields
three resource-dependent regimes: linear speedup, correlation-induced
saturation, and delay-induced reversal.  A matching Gaussian Markov lower
bound shows that the dependence on correlation, mixing, delay, and dual
budgets is unavoidable.  We further characterize the cost of learning the
regime itself: below an explicit identification threshold, conservative
fallback is minimax optimal, whereas above it a predictable policy approaches
the oracle on a separated mixing class.  Exact calculations, formal affine
experiments, and fixed-parameter nonlinear temporal-difference gradients
verify the predicted phase boundaries, including regimes where adding agents
has negligible value.  These results replace agent count by
dependence-adjusted effective parallelism and delineate when adaptive
participation is statistically worthwhile.

**Keywords:** Markov learning; multi-agent reinforcement learning; stochastic
approximation; delayed updates; minimax adaptation.

## Main theorem package

### Main Theorem: sharp finite-resource phase law

Let a predictable policy aggregate a subset \(S_t\) of agent innovations at
update \(t\), and define their long-run covariance

\[
\Gamma_{S_t}=\sum_{k\in\mathbb Z}
\operatorname{Cov}(\bar\xi_{S_t,0},\bar\xi_{S_t,k}).
\]

For strongly monotone affine Markov stochastic approximation with bounded
predictable delay, the main result must give a finite-horizon risk expression
or two-sided bound

\[
R_B(\pi)=R_{\rm transient}(\pi)
       +R_{\rm Markov}(\Gamma_{S_{0:T-1}})
       +R_{\rm delay}(\pi),
\]

with explicit message, environment, and wall-clock budgets.  Its corollaries
must locate three computable regimes:

1. **speedup:** added agents reduce risk at the resource-matched horizon;
2. **saturation:** dependence caps the effective number of samples;
3. **reversal:** extra cost or staleness raises finite-horizon risk.

For equicorrelated simultaneous innovations, the result must recover

\[
n_{\rm eff}(q,\rho)=\frac{q}{1+(q-1)\rho}
\]

as a special case, rather than assuming it as a proxy.

### Theorem 2: matching predictable-policy lower bound

On a Gaussian joint-Markov subclass, prove a minimax lower bound for every
predictable participation and aggregation policy.  The upper and lower bounds
must match, up to universal constants or logarithmic terms, in \(q\),
long-run dependence, temporal mixing, delay, and both message and environment
budgets.  None of these quantities may be absorbed into an unspecified
constant.

### Theorem 3: cost of adaptation

On a certified-mixing separated class, define an explicit identification
threshold \(B_{\rm id}\) using the downstream learning-risk gap, observation
information, delay, and remaining dual budgets.  The target result is

\[
B<cB_{\rm id}\Longrightarrow
\text{fallback is minimax optimal},
\qquad
B>CB_{\rm id}\Longrightarrow
R_{\rm adapt}\le C'R_{\rm oracle}+\widetilde O(B^{-1}).
\]

The unrestricted unknown-mixing result remains negative.  The positive result
must not claim uniformity as the mixing coefficient approaches one or the
oracle gap approaches zero.

## SDDE result

The stochastic delay differential equation is retained as a theorem target,
not as the paper title.  For step size \(\varepsilon\), physical delays
\(\varepsilon\tau_i^\varepsilon\to\delta_i\), and centered scaled errors, the
desired bridge is weak convergence to

\[
dX(t)=-A\sum_iw_i(t)X(t-\delta_i(t))\,dt
      +\Gamma(w_t)^{1/2}\,dW(t).
\]

A Lyapunov--Krasovskii functional must recover the same stability/phase
boundary as the discrete theorem, with an explicit finite-\(\varepsilon\)
remainder or approximation rate.  Without this quantitative bridge, the SDDE
appears only as an appendix interpretation and no SDDE-derived algorithm is
claimed.

## Evidence architecture

The manuscript should present evidence by theorem prediction, not by research
chronology:

1. exact Gaussian/affine calculations identify speedup, saturation, and
   reversal regions;
2. independent formal affine experiments test the finite-budget adaptation
   threshold;
3. fixed-parameter nonlinear temporal-difference gradients test the
   dependence-adjusted covariance law;
4. fresh-seed standard tabular tasks test resource-matched phase directions;
5. a nonlinear transfer tests only qualitative regime ordering after the CPU
   gates pass.

The existing formal EXP-018B result supports the nonlinear covariance
mechanism, and EXP-016B supports the separated finite-budget threshold in its
registered synthetic and affine classes.  T-032 supplies exact evidence for a
predicted no-value region.  Earlier sampled controller outcomes remain design
information unless rerun prospectively under T-034.

## Claim boundary

The contribution is a sharp characterization and a regime-aware threshold
policy, not a universally superior multi-agent controller.  Strategic Markov
games, unrestricted unknown mixing, dense covariance inversion, general
actor--critic convergence, and generic client selection are outside the
claimed theorem class.

## Final hard stop

This is the last ICML pivot.  The ICML 2027 attempt stops if any one of the
following occurs:

1. the upper and lower bounds do not match in correlation, mixing, delay, and
   budgets;
2. the downstream-risk adaptation upper bound cannot close AC-9c on a
   separated class;
3. the bound is numerically unable to distinguish the three regimes on a
   frozen exact grid;
4. prospective exact phase classification is below 95% or tabular direction
   classification is below 80%;
5. no standard learning task contains both a predicted positive regime with
   at least 5% gain and a negative regime with at most 1% oracle value;
6. every nontrivial effect requires an injected common-factor-only task;
7. any principal metric, resource ray, or scenario family must be changed
   after observing its outcome.

If a stop condition occurs, the proved correlation/delay characterization,
unknown-mixing impossibility, and adaptation opportunity-cost results should
be submitted to a theory-appropriate or signal-processing venue without
another algorithm pivot.
