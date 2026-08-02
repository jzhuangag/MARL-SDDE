# T-040 ICML novelty and rigor audit

## Decision

The T-034 line is now a **conditionally ICML-worthy theory thesis**, but it is
not yet an ICML-ready submission. T-037--T-039 remove three earlier blockers:

1. exact vector finite-horizon risk under cross-agent Markov lag covariance;
2. exact predictable-policy Gaussian Markov minimax value under dual budgets
   and delay;
3. first-order downstream-risk adaptation on a separated class.

The remaining headline blocker is the multiplicative Markov-TD remainder.
The remaining empirical blocker is a fresh prospective standard learning task
that contains both a theorem-predicted positive participation regime and a
theorem-predicted no-value/reversal regime.

No GPU is authorized by this audit. The next experiment is an exact CPU phase
map and must be preregistered separately before execution.

## Primary-literature collision

The closest results establish important pieces but not their present
interaction:

- [Khodadadian et al., ICML 2022](https://proceedings.mlr.press/v162/khodadadian22a.html)
  prove linear speedup for federated TD/Q-learning under Markovian sampling;
  their theorem uses separate-agent sampling and does not characterize
  cross-agent correlation saturation.
- [Dal Fabbro et al., CDC 2024](https://arxiv.org/abs/2403.17247) study
  delay-adaptive multi-agent SA and obtain \(N\)-fold speedup under
  **independent** agent Markov chains; this is the closest direct comparator.
- [Adibi et al., AISTATS 2024](https://proceedings.mlr.press/v238/adibi24a.html)
  give tight delayed-SA dependence on mixing and delay, without
  cross-agent-correlation-limited participation.
- [Nagaraj et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/c22abfa379f38b5b0411bc11fa9bf92f-Abstract.html)
  prove Markov-regression minimax limits and show constant-step SGD can be
  suboptimal; they do not study cross-agent participation and dual costs.
- [Samsonov et al., NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ff45bac2fc44064fd8ae85301b40b41e-Abstract-Conference.html)
  analyze inference for Polyak--Ruppert averaged LSA under Markov noise; this
  makes asymptotic covariance itself insufficient as a novelty claim.

The defensible novelty wedge is therefore not “Markov data,” “delay,”
“multi-agent speedup,” or “long-run covariance” separately. It is the sharp
resource-dependent transition from linear speedup to correlation saturation
and reversal, together with an exact predictable-policy lower bound and the
finite-budget cost of learning which regime applies.

## Theorem audit

| Component | Current status | ICML use |
|---|---|---|
| T-037 additive vector phase identity | exact | main theorem if scope says additive Markov innovations |
| T-036 finite-state affine TD recursion | exact, state-enumerating | transfer theorem / proof benchmark |
| General multiplicative Markov TD | open Poisson/martingale remainder | headline blocker |
| T-038 Gaussian Markov minimax | exact equality, predictable actions | main lower-bound theorem; classical machinery must be acknowledged |
| T-017 finite-budget threshold | proved on compact separated class | main adaptation threshold |
| T-039 oracle matching | first order, \(1+O(\log B/B)\) | main asymptotic adaptation result |
| Unknown mixing near one | negative theorem | valid limitation/contribution |
| SDDE bridge | not quantitative | appendix interpretation only |

### The largest proof risk

For general linear TD,

\[
e_{t+1}=e_t-\eta\bar A e_{t-D}+\eta\xi(Z_t)
         -\eta(A(Z_t)-\bar A)e_{t-D}.
\]

T-037 handles the first three terms exactly but not the final
sample--iterate-coupled term. Replacing it by independent additive noise
would be a methodology error. The next proof task must use a Poisson
decomposition and bound its martingale, coboundary, and iterate-increment
remainders with explicit mixing and delay constants. If that proof is
vacuous or loses the phase dependence, the paper must retain the narrower
additive theorem rather than relabel it as general TD.

## Experiment audit

Existing evidence is useful but not sufficient as a final ICML package:

- EXP-018B formally supports the fixed-parameter nonlinear covariance
  mechanism.
- EXP-016B formally supports a separated finite-budget threshold in Gaussian
  and affine TD classes.
- T-032 is an exact negative/no-value result.
- EXP-017A and the earlier Blackjack/controller studies are honest negative
  design evidence, not support for a universal adaptive controller.

The next prospective sequence is:

1. exact CPU phase map from T-037 and T-038, with predictions frozen before
   calculation;
2. fresh-seed tabular/linear-TD task with unchanged per-agent marginals and
   controlled common randomness;
3. only if both gates pass, one nonlinear GPU transfer with a precomputed
   oracle-value ceiling of at least 5%.

## AI research failure-mode check

| Failure mode | Verdict | Evidence |
|---|---|---|
| Citation hallucination | pass for T-040 sources | primary proceedings/arXiv records and secondary metadata cross-checks recorded in citation_verification_t040.json |
| Implementation bug | no known issue | 36 focused theorem tests, including independent scalar/dense identities |
| Shortcut reliance | contained | additive theorem is not called general TD; multiplicative remainder is explicit |
| Bug treated as insight | pass | no new sampled outcome; prior negative experiments remain negative |
| Methodology fabrication | pass | no scientific trajectory or retrospective gate change |
| Result hallucination | pass | theorem status and unresolved boundaries are separated |
| Pipeline frame-lock | monitored | T-034 hard-stop rules remain active; another algorithm pivot is prohibited |

## Go/no-go gates

Continue toward ICML only if:

1. the multiplicative TD remainder is nonvacuous on the frozen exact grid, or
   the paper is deliberately narrowed and still passes a fresh novelty review;
2. the exact phase map classifies at least 95% of preregistered cells;
3. a standard task supplies both at least 5% positive oracle value and at
   most 1% value in a negative regime;
4. the prospective controller/selector is evaluated against the strongest
   task-by-budget fixed design, with a nontriviality gate.

Failure of any item triggers the already declared theory/signal-processing
fallback, not another ICML mechanism pivot.
