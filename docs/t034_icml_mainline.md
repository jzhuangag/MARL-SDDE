# T-034 ICML 2027 main line

## Final gate status after T-045A

The current ICML experimental line is stopped. T-045A was the final
standard-task feasibility attempt: its strongest task has only 1.404% oracle
adaptation value and the aggregate value is 0.704%, below the frozen 5%
threshold. This triggers hard-stop item 5 below. No further controller pivot,
standard-task redesign, nonlinear transfer, or GPU experiment is authorized
under T-034. The proved theory and already completed evidence remain active
for a theory-appropriate or signal-processing submission.

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
changes when agents observe temporally mixing streams with shared randomness
and delayed updates.  For delayed vector linear stochastic approximation with
additive Markov innovations, we derive an exact finite-horizon risk identity
governed by the full cross-agent lag-covariance sequence; for finite-state
affine temporal-difference systems, we give an exact mode-conditioned moment
recursion that retains sample--iterate dependence.  The resulting phase law
has three resource-dependent regimes: linear speedup, correlation-induced
saturation, and delay-induced reversal.  An exact Gaussian Markov minimax
theorem shows that predictable data-dependent participation cannot remove the
dependence on correlation, mixing, delay, and dual budgets.  We further
characterize the cost of learning the regime itself: below an explicit
identification threshold, conservative fallback is justified, whereas above
it an explore-then-commit policy reaches oracle-normalized minimax ratio
\(1+O(\log B/B)\) on a separated mixing class.  Exact calculations, formal
affine experiments, and fixed-parameter nonlinear temporal-difference
gradients test the predicted phase boundaries, including regimes where adding
agents has negligible value.  These results replace raw agent count by
dependence-adjusted effective parallelism and delineate when adaptive
participation is statistically worthwhile.

**Keywords:** Markov learning; multi-agent reinforcement learning; stochastic
approximation; delayed updates; minimax adaptation.

## Main theorem package

### Main Theorem: sharp finite-resource phase law

For a fixed resource-feasible participation/stride design, aggregate the
agent innovations and define their lag and long-run covariances

\[
K_k(q,b)=\operatorname{Cov}(\bar\xi_0,\bar\xi_k),
\qquad
\Gamma(q,b)=\sum_{k\in\mathbb Z}K_k(q,b).
\]

For delayed vector linear stochastic approximation with additive stationary
Markov innovations, T-037 gives an exact finite-horizon risk expression

\[
R_B(q,b,D)=R_{\rm transient}
 \eta^2\sum_{s,r<T(B,q,b,D)}
 \operatorname{tr}(QH_sK_{s-r}(q,b)H_r^\top),
\]

with explicit message, environment, and wall-clock budgets.  Its corollaries
locate three computable regimes:

1. **speedup:** added agents reduce risk at the resource-matched horizon;
2. **saturation:** dependence caps the effective number of samples;
3. **reversal:** extra cost or staleness raises finite-horizon risk.

For equicorrelated simultaneous innovations, the result recovers

\[
n_{\rm eff}(q,\rho)=\frac{q}{1+(q-1)\rho}
\]

as a special case, rather than assuming it as a proxy.

### Theorem 2: matching predictable-policy lower bound

On the Gaussian common-factor Markov location subclass, T-038 proves the
exact minimax value for every predictable participation/stride policy.  A
deterministic covariance design and generalized least squares attain it, so
data-dependent action selection cannot improve the value.  The expression
retains \(q\), temporal mixing, delay, and both message and environment
budgets; none is absorbed into an unspecified constant.  Comparing a
particular constant-step SA iterate with this GLS value remains a separate
algorithmic-efficiency question.

### Theorem 3: cost of adaptation

On a certified-mixing separated class, define an explicit identification
threshold \(B_{\rm id}\) using the downstream learning-risk gap, observation
information, delay, and remaining dual budgets.  T-017 and T-039 give

\[
B<cB_{\rm id}\Longrightarrow
\text{fallback is minimax optimal},
\qquad
B>CB_{\rm id}\Longrightarrow
\max_j\frac{R_j^{\rm adapt}}{R_j^{\rm oracle}}
\le1+O(\log B/B).
\]

The unrestricted unknown-mixing result remains negative.  The positive result
does not claim uniformity as the mixing coefficient approaches one or the
oracle gap approaches zero, nor second-order equality to the finite-budget
controlled-belief occupation value.

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
games, unrestricted unknown mixing, general actor--critic convergence, and
generic client selection are outside the claimed theorem class. T-042 exactly
decomposes the general multiplicative Markov term into a weighted martingale
transform, boundary terms, an impulse-response increment, and a
delayed-iterate increment with computable finite-state constants. General
multiplicative Markov TD is covered by T-046 only on the explicit bounded
class whose finite-horizon robust small gain is below one and whose risk
envelopes preserve the claimed phase ordering. When that certificate is
vacuous or overlaps, the paper states the exact finite-state result and keeps
the main dimension-free theorem additive; it does not claim a sharp
unrestricted multiplicative-TD rate.

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
