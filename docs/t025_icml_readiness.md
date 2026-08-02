# T-025 ICML readiness after EXP-018B

## Updated decision

The fixed-participation main line is now a credible ICML 2027 candidate, but
not a submission-ready paper. Two previously separate evidence chains are now
connected:

1. the delayed affine Markov-TD theory and EXP-016B formal finite-horizon
   threshold/transfer evidence; and
2. the exact nonlinear fixed-parameter covariance identity and EXP-018B
   prospective formal calibration.

The appropriate title family remains **Beyond Linear Speedup: Correlation-
and Delay-Limited Participation under Multi-Agent Markov Data**. The word
"beyond" means replacing independent-agent linear speedup by the attainable
factor `q/[1+(q-1)rho]`; it never means superlinear acceleration.

## Claim-to-evidence chain

| Claim | Theoretical support | Prospective evidence | Status |
|---|---|---|---|
| delayed mean-square contraction | Theorem 3 | affine tests/EXP-007B--D | defensible in stated thinned/decorrelated scope |
| affine delayed finite-time error | Theorem 4 | EXP-016B Layer B formal | defensible in stated affine scope |
| correlation-limited speedup and dual-budget threshold | Theorem 5 | EXP-016B Layer A formal | defensible |
| finite-horizon adaptation/opportunity cost | adaptive lower bound and threshold sandwich | EXP-016B formal | defensible for certified separated regimes |
| nonlinear gradient variance transfer | Theorem 9 | EXP-018B formal, exact reproduction | defensible as fixed-parameter identity only |
| unrestricted unknown mixing | impossibility theorem | negative-boundary simulations/design audits | defensible negative result |
| SDDE delay geometry | Lyapunov--Krasovskii representation | mechanism diagnostics | interpretation only |

## What EXP-018B closes

T-021 required a prospective nonlinear direct-gradient covariance endpoint.
EXP-018B closes that exact gate: 192 new seeds, 18,432 finite unique rows,
Bonferroni-controlled co-primary upper bounds, 8/8 statistical/validity gates,
and byte-identical independent reproduction.

It does not validate parameter drift, nonlinear optimization, online
participation selection, or a communication-matched neural learning curve.
Those implications remain prohibited.

## Why the paper is not yet complete

The theorem/controlled-mechanism package is coherent, but an ICML audience
will still expect an external nonlinear learning demonstration. The previous
CartPole/Acrobot benchmark cannot fill that role: T-020 proved that even its
outcome-aware cellwise fixed-q oracle improves the correct strong fallback by
only 0.3846%, so no meaningful adaptive controller can be demonstrated there.

A new benchmark may proceed to GPU only after an outcome-free CPU/static
commit establishes all of the following:

- standard intrinsically stochastic fixed-policy dynamics with unchanged
  per-agent marginals under controlled coupling;
- public budget rays with an internal fixed-q optimum;
- at least 5% aggregate strong-baseline oracle value and directional value in
  at least 60% of cells after all communication/probe costs;
- a frozen observable risk surrogate and nontrivial comparator;
- exact budget, delay, marginal-invariance, and taint tests;
- new pilot seeds and runner/analyzer hashes.

Until those conditions pass, GPU execution would add scale without closing a
claim gap.

## Recommended paper architecture

1. exact correlation-limited speedup law and fixed-q resource phase diagram;
2. delayed affine Markov-TD finite-time theorem;
3. finite-budget identification/opportunity-cost boundary and unknown-mixing
   impossibility;
4. nonlinear fixed-gradient transfer lemma;
5. EXP-016B formal threshold/affine results and EXP-018B formal covariance
   validation;
6. a separately preregistered nonlinear learning benchmark only if its static
   value certificate passes.

The online controller is not a headline contribution. SDDE remains a
Lyapunov--Krasovskii interpretation layer, while the discrete theorem supplies
the convergence statement.

