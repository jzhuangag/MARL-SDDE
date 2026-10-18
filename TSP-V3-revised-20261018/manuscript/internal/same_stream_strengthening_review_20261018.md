# Substantive V3 theory--evidence strengthening

## Material passport and decision scope

Human authorization: strengthen V3 novelty and the theory--experiment loop using existing evidence, independently of the running asynchronous MARL comparison. Target: IEEE TSP. Work is confined to the isolated revised manuscript and new analytical sidecars. This is a new derivation and an outcome-known deterministic reanalysis, not a new performance experiment, preregistration, native-MARL optimizer result, external independent review or journal acceptance clearance.

The substantive addition is **same-stream identification-to-learning risk transfer**, including the actual averaged delayed FIFO learner and its exact budget-dependent weights. Gaussian conditioning and Cauchy--Schwarz are standard; their resource-priced selection/learner interface, not the inequalities in isolation, is the contribution being advanced.

## What was missing and what changed

The general learning-value theorem needs valid upper certificates conditional on the entire probe history. The completed Gaussian experiments instead use disjoint but consecutive observations on the same common-factor path. A large posterior common-state mean can change future risk after an identification decision. It is therefore invalid to infer their actual selected risk just from stationary commitment risks and a wrong-decision probability.

The new theorem computes the conditional risk of any frozen linear program exactly:

`J_a + 2 d_a k_a m_j + k_a^2 (m_j^2 - s_j^2)`.

It then bounds covariance with the decision event by `sqrt(delta_j) Gamma_j`. Both upper and lower actual-risk enclosures result. No independent restart or uniform bound over unbounded Gaussian histories is required. The programs, weights, horizons and initialization are fixed before the probe; adaptively fitted downstream programs are outside this theorem.

Exact linear weights are derived for the saved pseudo-observation running mean and for the saved server FIFO affine recursion's **average of applied iterates**, with its real empty-queue initialization. All scheduled observations, including terminal unused increments, remain paid. This is not a substitution of an easier fresh learner or of an unaveraged endpoint.

A second derivation proves the stable averaged FIFO risk is `K_a/s + O(s^-2)` on the fixed budget ray and its probe-selection correction is `O(sqrt(delta_j)/s^2)`. Stability of the registered step is shown from the delayed polynomial's first unit-circle crossing. This supplies a concrete delayed, same-stream realization of the shrinking-risk interface. It does not remove the unaveraged fixed-step noise floor, general Markov uniform-history assumptions, or fixed neural-critic assumptions elsewhere.

## A failure found and preserved, not hidden

The first analytic pass used the old nominal `delta=0.025`. Its audit failed. The frozen probe-length helper uses `log(0.5/delta)`, which gives an **equal-prior average-error** sufficient bound, not the same directional sufficient bound. All104 exact affinities exceed0.025, ranging0.0352270864--0.0497464853; they are below0.05. This does not prove the actual directional errors exceed0.025.

The provisional output, code and failure record are preserved at commit2aec1a5 and `internal/same_stream_evidence_20261018/`. That nominal-directional analysis is **not admitted**. The corrected output is separate: `internal/same_stream_directional_evidence_20261018/`, using each existing probe's exact affinity as its directional upper bound. No probe length, action, source, threshold, seed, observed result or old gate changed. This correction is to the new analytical interpretation, not a retry of a scientific payload. The general theoretical directional guarantee remains conditional on an actually directional probe design.

## Complete coverage and results

The formal metrics and saved summaries were SHA-checked against the original registry/reproduction audit. The analysis covers14,336 policy cells: all8 policies,104 Layer-A/56 Layer-B scenario families, all budget points, both true regimes, and the8 search-censored scenarios. Every existing empirical mean retains192 evaluation seeds. First-seed protocol metadata reconstructs deterministic budget mappings; its numerical risk and likelihood ratio are never read. Its identified label is used only to check that the recorded action equals the corresponding frozen branch, not to choose the counterfactual programs or risk interval. Source arithmetic checks selected actions, scheduled/usable updates, messages and environment ledgers. The original96 per-scenario primary means are recovered from the saved summaries.

For the original practical intermediate-budget estimand:

| Layer | Actual-risk improvement enclosure | Saved empirical mean | Positive / adverse / inconclusive cells |
|---|---:|---:|---:|
| Gaussian running mean | [0.1607242,0.1951514] |0.1767168|323 /101 /22 (446total)|
| Averaged delayed affine learner | [0.0562005,0.0772222] |0.0726347|176 /33 /13 (222total)|

The intervals are population-risk enclosures, not confidence intervals for individual sample means. Their agreement with the saved aggregate observations supports a **mechanism-level closure**, not fresh independent confirmation. Positive aggregate expected improvement survives both same-stream and directional-error corrections. It is not just the original point estimate repeated.

Negative scope is material. Over the whole Layer-B intermediate region, the mean enclosure is [-0.004872,0.025139], so a positive guarantee is not established there. There are provably adverse regime/budget cells even inside the positive original practical aggregate. Above finite value thresholds and below identification, the two policies execute identical programs. Full all-stratum and all-policy outputs, not only the positive practical aggregate, are retained. Across the complete registered Layer-A and Layer-B design, the mean improvement lower bounds are positive (0.0506883 and0.0079930), but this finite design average is not a population-universal guarantee.

## Core added contribution relative to nearby work

| Nearby work and primary record | Its relevant focus | Defensible V3 distinction |
|---|---|---|
| [DASA](https://ieeexplore.ieee.org/document/10886380/) | Delay-adaptive multi-agent stochastic approximation under Markov sampling | V3 prices common-factor redundancy, spacing and identification against a finite remaining learner; new risk transfer explicitly handles the probe/learning decision dependence. No claim of a better general delay rate. |
| [Yu et al., SDDE worker activation](https://arxiv.org/abs/2406.11159) | SDDE-based asynchronous learning/worker activation | SDDE is an interpretation here, not the claimed novelty. Integer paid horizons and the continuing-stream identification-to-risk interface are explicit. |
| [Perazzone et al., Lyapunov FL resource control](https://ieeexplore.ieee.org/document/10885536/) | Convergence-bound-driven client/power resource optimization | Lyapunov resource selection itself is established. V3 adds correlated Markov participation/spacing and an identification decision valued through downstream risk and paid lost updates. |
| [Li et al., parameter-free TD,2026](https://arxiv.org/abs/2603.02577) | Removing problem-dependent tuning through exponential steps and regularization | V3 is **not parameter-free**; its distinct question is finite-resource sensing/learning allocation and selection-risk transfer, not unknown-mixing adaptation. |
| [Liu et al., Poisson--Moreau drift,2026](https://arxiv.org/abs/2605.07104), [Chen--Maguluri survey,2026](https://arxiv.org/abs/2605.31309) | Almost-sure SA rates / broad Lyapunov finite-time framework | V3 does not claim a new general Lyapunov framework or sharper universal SA rate. Its contribution is the paid action/information/downstream-program coupling. |
| [Li--Luo--Zhang, adaptive Markov SA concentration,2026](https://arxiv.org/abs/2609.29833) | Shrinking-tube reliability for projected adaptive Markov SA | V3's new result is an explicit expected squared-risk transfer for fixed Gaussian linear programs, not all-future high-probability concentration. |

Search provenance: bounded public searches on2026-10-18 for delayed Markov SA, finite-budget identification, correlated multi-agent TD and recent Lyapunov/adaptive-Markov work; primary publisher/arXiv records inspected. The2026 records above are abstract-level adjacent-work checks, not exhaustive full-text priority clearance. No blanket 'first', absence claim or experimental superiority over those methods is made. These new background records are documented here without silently editing the preserved Bib file; the main text sharpens comparisons to its three already verified closest resource/delay references. Published methods were not newly rerun in V3, and the independent140h native study is not used to fill that gap.

## Research and statistical validation (11/11 checked)

1. Simpson/aggregation: regime-level adverse cells and the inconclusive whole-TD intermediate mean are reported alongside positive aggregates; no all-cell claim.
2. Ecological inference: design-average risk is not an every-seed, every-task or arbitrary-environment guarantee.
3. Selection/Berkson: retain the original practical mask but also all registered strata; no positive-outcome filtering.
4. Collider: no new outcome-conditioned covariate adjustment or selected control.
5. Base rate: directional errors distinguished from equal-prior average error; both true regimes remain in coverage.
6. Regression to mean: no new selection of extreme seeds or before/after reruns.
7. Survivorship: all registered finite and search-censored cases retained; original failures preserved.
8. Look-elsewhere: full-cell outputs and adverse intervals reported; no new significance test or multiplicity-free p-value.
9. Forking paths: outcome-known plan and the failed nominal pass are preserved; corrected affinity interface is an analytic repair, not an outcome-tuned threshold.
10. Causal scope: exact Gaussian program identity and old paired controlled simulations are kept separate from empirical general-MARL causality.
11. Reverse causality: probe/program freezing precedes observations; no use of future returns in action choice.

Validation level: CAUTION for new independent performance or priority claims, which are not made; the analytical population statements require the explicit fixed-program Gaussian assumptions. This report is a main-agent technical review, not an independent external review.

## Publication disposition

The manuscript now has a more concrete statistical signal-processing contribution: **what can be learned from correlated streams per paid budget, whether identifying that dependence is worth its cost, and how its decision transfers to the actual delayed learner**. This is a meaningful TSP-oriented theoretical/evidence improvement, but not evidence of a broadly superior new asynchronous MARL optimizer. Original V3, all old data and unsuccessful studies remain unchanged. Complete proofs are preserved in a companion supplement rather than compressed typography. Human ORCIDs, related/prior submission disclosures and all-author final approval remain author-release requirements; no acceptance or submission is claimed.
