# T-033 low-rank factor-correction audit

## Verdict

Low-rank shared-factor correction is stopped as the ICML algorithm headline.
The algebra is valid and inexpensive, but the core known-factor estimator is
a classical constrained best linear unbiased estimator.  The unconstrained
full-covariance BLUE/GLS estimator weakly dominates it and is itself available
in the same low-rank complexity through a matrix-inversion lemma.  The Markov
delay extension also loses exact cancellation or stability without much
stronger structure.

No scientific trajectory, preregistration, CPU pilot, GPU job, or HPC4 job is
created by this audit.

## Static model

For simultaneous observations

\[
g_t={\bf 1}h(\theta_t)+Bf_t+\varepsilon_t,
\qquad
\operatorname{Cov}(\varepsilon_t)=D,
\]

exact factor-corrected weights solve

\[
\min_w w^\top Dw
\quad\text{subject to}\quad
{\bf1}^\top w=1,\quad B^\top w=0.
\]

Writing \(A=[{\bf1},B]\), the solution is

\[
w_{\rm FC}=D^{-1}A(A^\top D^{-1}A)^{-1}e_1.
\]

It exists exactly when \({\bf1}\notin\operatorname{col}(B)\).  Equal factor
loadings therefore reproduce the common-factor floor instead of permitting a
correction.

## Dominance obstruction

With \(\Sigma=D+B\Omega B^\top\), the BLUE/GLS weights

\[
w_{\rm GLS}=\frac{\Sigma^{-1}{\bf1}}
 {{\bf1}^\top\Sigma^{-1}{\bf1}}
\]

minimize \(w^\top\Sigma w\) over every unbiased linear weight.  Exact
cancellation adds constraints, so

\[
w_{\rm GLS}^\top\Sigma w_{\rm GLS}
\leq
w_{\rm FC}^\top\Sigma w_{\rm FC}.
\]

For rank \(r\), Woodbury evaluation costs
\(O(mr^2+r^3)\) arithmetic and \(O(mr)\) storage; no \(m\)-by-\(m\) inverse is
needed.  In the audited two-agent example with loadings `(0.5,1.5)`, unit
idiosyncratic variance, and factor variance 10, exact cancellation has risk
2.5 while low-rank GLS has risk 2.1667.  Thus the proposed complexity is useful
but not a new statistical optimum.

## Delay obstruction

If agent \(i\) delivers \(B_i f_{t-\tau_i}\), arbitrary factor paths require
one cancellation constraint for each delay cohort.  For two agents with
loadings `(1,2)`, simultaneous delivery admits weights `(2,-1)`, whereas
delays `(0,1)` make pathwise cancellation infeasible.  Buffering restores a
common factor time only by making every update as stale as the slowest cohort;
prediction replaces exact cancellation with factor-model error.

The same weights expose a stability issue in the scalar mean recursion

\[
e_{t+1}=e_t-\alpha\sum_iw_i e_{t-\tau_i}.
\]

At \(\alpha=0.9\), the spectral radius is 0.10 for simultaneous observations
but 1.4296 for delays `(0,1)`.  Negative weights therefore cannot inherit the
usual convex aggregation stability argument.

## Naturalness and scope

Control variates in RL and dynamic factor estimation are established fields.
More importantly, common noise in a standard Markov or mean-field control task
often changes the true state, reward, or conditional value and should not be
deleted as nuisance.  Unbiased cancellation is scientifically justified only
for a certified zero-mean measurement/innovation factor with separated
loadings.  Injecting such a factor into a standard Gym task would not satisfy
the project's natural-benchmark gate.

The factor route could still support a specialized sensor-fusion or signal-
processing paper with an explicit nuisance reference channel.  It does not
currently rescue the ICML algorithm track.

## ICML consequence

After T-030, T-032, and T-033, further participation, subset, weighting, or
factor-correction variants are not authorized as outcome-driven replacements.
The remaining credible ICML attempt must be theory-first: a tight
characterization of when multi-agent Markov data yield linear speedup, when
dependence/delay impose a floor, and when finite-horizon adaptation has
nontrivial value.  Existing positive and negative experiments should test
that phase diagram rather than advertise a universally superior controller.
