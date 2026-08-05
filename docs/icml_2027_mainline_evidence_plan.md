# ICML 2027 mainline and evidence plan

## Recommended claim

The defensible central claim is:

> In delayed multi-agent Markov TD with a fixed nonlinear representation,
> reward-free fingerprint sensing can select participation from a
> correlation--cost phase rule and retain a finite-horizon Lyapunov guarantee
> under the same message, environment, and delay budgets.

This is narrower than general actor--critic or end-to-end MARL, but it has a
complete causal chain: the common-path model determines the variance term, the
two budgets determine the usable horizon, the delayed Lyapunov recursion
determines the finite-time risk, and a reward-free observable estimates the
correlation parameter without using learning outcomes.

The title should avoid an unconditional unknown-mixing claim and avoid saying
that the method gives universal superlinear wall-clock speedup.  A candidate
is **“Reward-Free Correlation-Adaptive Participation for Delayed Multi-Agent
Markov Learning.”** “Beyond Linear Speedup” can be a subtitle only if the
formal resource-normalized comparison supports that wording.

## The theorem package that must appear in the paper

1. **Delayed finite-horizon Lyapunov bound.**  For the affine TD head under a
   stable drift and bounded Markov noise, derive the exact delayed recursion
   bound in the (P)-norm.  The leading noise term must expose
   \(v(q,\rho)=\rho+(1-\rho)/q\), while the contraction and usable horizon
   expose (D) and both residual budgets.

2. **Budget-aware phase rule.**  Substitute
   \(N(q)=\min\{\lfloor B_m/(h+q)\rfloor,
   \lfloor B_e/q\rfloor\}-D\) into the finite-time upper bound and prove the
   selected q minimizes the certified surrogate over the finite catalogue.
   State clearly when the closed-form rule is exact (fixed ray/known
   correlation) and when it is only a certificate-guided action.

3. **Reward-free fingerprint concentration.**  With independent probe blocks,
   prove a time-uniform or fixed-horizon concentration bound for
   \(\hat\rho=K/m\), including the independent-path collision term.  Then bound
   the excess certified risk from selecting the wrong q by the phase-score
   modulus.  Probe cost must be subtracted before applying the bound.

4. **Adaptation cost/lower-bound boundary.**  Retain the existing stopped
   change-of-measure and adaptive occupation lower-bound results to explain why
   unknown-mixing adaptation cannot be claimed uniformly and why probing has a
   real opportunity cost.  This prevents the paper from overselling the
   sensing mechanism.

The SDDE/Lyapunov--Krasovskii construction should be an interpretation layer;
the primary theorem must be the executed discrete delayed Markov-TD recursion.
No theorem should imply convergence for a trainable ReLU encoder or a general
actor--critic unless a separate proof is added.

## Main-text positive evidence

The main experimental table should be populated only after T-063A passes its
frozen formal gates.  It should report, for each of Asterix, Breakout, and
Seaquest and for the aggregate:

- controller/strong-fixed geometric terminal-risk ratio with one-sided cluster
  bootstrap upper bound;
- delay-0 and delay-8 ratios;
- strict improved-cell breadth with its lower bound;
- true-correlation oracle proximity;
- message- and environment-binding cases.

The primary comparison is against the frozen task-by-budget strong fixed-q
baseline with no probe cost.  The controller must be credited for all probe
messages and actor transitions.  No pilot seed, pilot-selected threshold, or
outcome-aware oracle row may enter this table.

## Positive appendix evidence

If T-063A passes, the appendix should contain positive, reproducible
breakdowns rather than a collection of unrelated ablations:

- all seven correlation levels and the selected-q phase diagram;
- both delays and both budget-binding regimes;
- seed-cluster confidence intervals, not endpoint-level standard errors;
- fingerprint calibration, rho-zero collision rate, and q-direction paths;
- controller versus true-rho full-budget gap;
- probe-cost accounting and usable-update fraction;
- CPU time, memory, and O(1) controller arithmetic scaling;
- fixed-q envelope and a no-probe ablation that isolates probe opportunity cost.

The prior negative EXP-017A/T-020 results should be a short diagnostic appendix:
they establish why reward-dependent selection, q=1 absorbing behavior, and a
weak fallback are not acceptable.  They must not be mixed into the positive
formal sample or used to relax any gate.

## Decision tree after T-063A

- **All formal gates pass:** freeze the formal result, write the theorem
  package and main/appendix tables, then consider one new GPU-only learned-
  encoder robustness experiment.  That experiment is optional for the first
  submission and requires its own preregistration.
- **Formal value passes but one robustness gate fails:** report the exact
  qualified scope (for example, taskwise failure on one benchmark) and do not
  claim broad standard-RL superiority.  A new benchmark may be designed only
  after an outcome-free audit.
- **Formal fails:** stop the positive ICML claim, preserve the failure, and
  return to theorem/benchmark redesign.  No gate, seed, comparator, or
  analysis may be changed to make the body positive.

Thus “正文正向、附件正向” is a desired presentation only conditional on
independent formal evidence; it is not an analysis rule.  The current CPU
formal run is the decisive next step, and no GPU is needed yet.
