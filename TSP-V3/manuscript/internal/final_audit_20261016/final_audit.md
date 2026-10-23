# V3 TSP presubmission final audit — 2026-10-16

## Decision: NOT RELEASED FOR SUBMISSION

The existing V3 has positive, traceable empirical findings and remains the TSP-first manuscript. The new asynchronous actor study is independent and must not delay this manuscript's repair path. However, this fresh audit finds a definite headline-denominator misdescription and proof gaps. The earlier clean-audit descriptions must not be treated as a current submission clearance.

**No V3 manuscript, bibliography, figure or PDF was overwritten. No new experiment was run for this audit.** Findings and arithmetic are saved separately. This is a completed defect-finding review, not a claim that every ARS final-release gate has passed.

## 1. What remains genuinely positive

| Finding | Fresh audit evidence | Honest interpretation |
|---|---|---|
| MPE controller versus fixed-q catalogue | TSP/internal/marl_mpe_qgrid_audit_gate.json, 80 registered records | Regime-wise deficits -0.3572%/-0.8892%; positive unknown-regime normalized comparison, not universal dominance |
| MaMuJoCo fresh confirmation | TSP-V3/results/marl_mamujoco_calibrated_confirmation_20260930/gate.json and summary.csv | Mixture AUC +8.340817%, one-sided lower +1.107360%, 7/8 positive pairs, p=.03515625; final +4.636745%; prespecified task-specific catalogue {2,8} |
| Resource-indexed fixed-policy learning curves | TSP/internal/convergence_curve_validation.json and TSP/data/convergence_curve_summary.csv | Parameter/return-estimation AUC ratios .900913/.886789; 64 held-out seeds; not an actor-return theorem |
| Robust shield and objective ablations | TSP-V3/results/certificate_robust_shield_confirmation_20261003/{summary.json,paired_comparisons.csv} | 24/24 stress-cell upper means covered; all nominal actions retained; terminal parameter risk 12.20% below myopic and 64.22% below correlation-only |
| Learning-aware probing | Immutable EXP-016B cell_summary.csv, hash 08a72e82075a4896ab41635f9b01341850ed8c6715448ed652bb33cd53026076 | Positive effect retained, but manuscript's percentage interpretation requires the correction below |

The myopic ablation is **not** a win on every metric: finite-horizon/myopic parameter AUC=1.008316, return-estimation AUC=1.013570, terminal return-estimation risk=1.024412; their intervals include one. The paper's specific terminal-parameter claim is supported. Do not expand it into all-metric superiority.

### Important distinction between the two reported MARL percentage estimands

The MPE 2.319499% figure is reconstructed exactly as the mean of per-seed, per-regime relative AUC gains versus q8, with equal regime weights. Normalizing **after** mixing the two regime returns gives 2.779808% instead; the ratio of aggregate means gives 2.811688%. MaMuJoCo's 8.340817% uses the paired mixture-first definition. Both reported effects are positive, but the two percentages are not identically normalized estimands. Define the formulas explicitly rather than presenting the two numbers as interchangeable instances of one unspecified metric. Do not silently substitute a new primary metric.

The MPE fixed q2/q4 cells were a baseline-completeness extension after the original confirmation. The retained controller evidence and the later comparator extension should not be described as a single newly preregistered 80-cell confirmation. The old validation report states this distinction correctly.

## 2. Required corrections, ordered by submission risk

### IL-SERIOUS-1 — EXP-016B denominator and headline claim

Locations: abstract main.tex:36; contribution main.tex:112; simulations main.tex:774,780; Fig. 3 and its generator.

The frozen analyzer calculates

    (risk_information_only - risk_learning_aware) / risk_always_all,

not the relative reduction divided by risk_information_only. The saved data give:

| Subset | Currently advertised % | Actual relative reduction versus information-only % |
|---|---:|---:|
| Layer A | 54.426655 | 35.244340 |
| Affine TD | 10.852328 | 9.789897 |
| Delay active | 49.121173 | 32.940442 |
| Message binding | 77.687224 | 43.721334 |
| Environment binding | 43.130250 | 30.133567 |

All remain positive. Preserve all frozen tests, outcomes and thresholds. Either label the existing values **fallback-normalized paired risk improvements** everywhere, or use the actual relative reductions in the prose and regenerate only the display from the same saved cells. This is correction of reported meaning, not permission to alter an experiment. Exact means and formulas are in readonly_checks_complete.json.

### IL-SERIOUS-2 — Threshold proof uses a false pointwise KL bound

Appendix F's compactness argument does not bound Gaussian predictive-mean differences on every history. A fully explicit compact separated counterexample and an expected-information repair are in theory_repair_memo.md:T1. This is a proof-step failure; it does not establish that the stated threshold result is impossible.

### IL-SERIOUS-3 — Existence and claimed exact scalar solve

Proposition 2 invokes compactness after deleting a stability boundary. That inference is invalid in the written domain, and eta_min=0 has a concrete counterexample. Nonempty closed margins/coercivity or an epsilon-minimizer formulation is needed. A bounded local scalar optimizer also needs a global-error argument before an exact continuous-oracle claim is made. See memo:T2.

### IL-SERIOUS-4 — Sliding-envelope / tracking / delay interface

The fixed-target proof's two-case envelope argument is not the unconditional affine block recurrence used later. Positive-part contraction can repair part of the algebra, but random-delay selection and mixed-target delayed histories also need explicit assumptions or terms. See memo:T3. This is especially important before treating the critic-tracking extension as a guaranteed policy-learning interface.

### IL-MEDIUM-1 — Safety and coverage scope

The finite-budget comparison is against a fallback **upper certificate**, not automatically its actual MSE. The high-instance statement already says certified risk; apply that precision consistently to the low-instance allowance. Coverage failure probability alone does not add directly to unbounded squared loss. See memo:T4.

### IL-MEDIUM-2 — Phase and computational applicability

Identify the learner supplying the threshold theorem's uniform 1/s risk expansion, distinguish exact spectral moments from cheap conservative scalar bounds, and specify initial delayed history. These are theorem-to-implementation questions, not reasons to discard positive curves. See memo:T5.

### IL-MEDIUM-3 — Submission author metadata

The source still says Anonymous Authors. Current SPS journals use single-anonymized review; TSP requires author details and ORCID metadata. Names, order, affiliations, corresponding author, consent and disclosure information must come from the authors, not be inferred or invented. Submission itself has not been authorized or attempted.

### IL-MINOR-1 — Equation-reference style

40 source lines contain bare or long-form equation references rather than the requested Eq.~\\eqref convention. Every affected line is enumerated in readonly_checks_complete.json. This is style, not a mathematical failure.

### IL-MINOR-2 — Fig. 2 tick crowding and exact-calibration prose

Fig. 2's 0.9/0.98 persistence ticks are crowded at column width; categorical equal spacing would improve legibility without changing data. Its plotted action arrays are hard-coded but match all twelve saved selected actions in docs/validation_exp010b.md. Prefer loading that table to avoid future transcription drift. No definite clipping was found.

## 3. Bibliographic integrity and IEEE style — separate results

- All 37 cited works obtain fresh positive authoritative metadata matches; no orphan/dangling/duplicate keys were found. No reference was invented, replaced or silently repaired.
- Crossref DOI metadata and official proceedings pages were used. Semantic Scholar was unavailable at its first request; OpenAlex returned two records then became unavailable. Remaining requests were marked unavailable/stopped, not passed. Resolver availability and raw metadata are preserved in final_citation_audit_20261016_a1.
- The automated 36-pass/1-conflict result concerns Paulin's article number79 versus Crossref issue-none. The author's institutional record verifies article79, volume20, 32 pages, September2015 and the DOI. This is a representation judgment, not an unverified or fabricated work. Prefer explicit Art. no.79 formatting; the .bib remains untouched.
- The two retained PettingZoo/FACMAC arXiv identifiers were freshly verified as the same works; their 2020 preprint submission dates do not replace the published2021 dates.
- IEEE style uses this project's IEEEtran.bst and existing compact canonical venues. No definite missing/duplicate/empty-field error was found. PMLR location/month omissions and NeurIPS Adv. convention are project judgments, not silently rewritten.
- Closest-work context was freshly checked against Yu/Chen/Poor's 2025 SDDE paper, Perazzone et al.'s Lyapunov scheduling paper and Cummins et al.'s feedback-participation paper. The latter two already have principled/theory-derived participation mechanisms. V3's defensible distinction is the specific correlated-Markov finite-budget value/information/cost interface, not the first theoretical Lyapunov participation controller.

Full100% citation-context and >=50% paragraph-originality release screening is **not certified complete** by this blocked audit. Four characteristic-text searches found no exact matching manuscript sentences; that is only a limited screening sample, not a plagiarism clearance. Author-dependent self-overlap/disclosures cannot be completed with Anonymous Authors. Re-run the final ARS release gate after the substantive corrections; do not recycle a stale clean-audit label.

## 4. Build, typography and visual review

- Existing PDF preflight PASS: 13 readable letter-size pages, no encryption, missing page or parser warning. PDF hash 6fe44327eb90d286de354c0698f49611fce0082bea09cec4dd7d7d5d7ba46bfd.
- All13 pages were rendered and visually inspected. No blank overflow page, clipped equation, figure/caption collision or malformed glyph was found. Fig.4/5 panel sizes, below-axis labels and fraction ticks 0.00/0.25/0.50/0.75/1.00 are consistent.
- All four experimental figure PDFs embed Times New Roman; italic math uses the matching Times New Roman face. Shared generator specifies grid alpha .28, linewidth .55 and band alpha .14. No figures were regenerated.
- Built-in preview compile was attempted but blocked before TeX execution by Windows CreateProcessAsUserW access denied. This is an environment error, not evidence of a source defect.
- Isolated pdflatex **draftmode**, BibTeX and two further draft passes exited0 without writing a PDF. Final diagnostics: zero undefined references/citations, zero overfull boxes, four underfull hbox warnings. The original PDF stayed unchanged. Do not repeat the old claim of zero warnings.
- Abstract has213 whitespace-counted words, within the user's150–220 requirement; five keywords; introduction and Simulations structure are contribution-led. No standalone Related Work or Experiments heading needs repair.

## 5. Venue judgment and workflow decision

TSP-first remains reasonable for **correlated Markov estimation, delayed stochastic approximation and finite-budget Lyapunov resource design**. A forced wireless case is not required to describe that contribution honestly. Generic stochastic optimization alone, however, is not automatic scope admission or acceptance. The paper should lead with its specific statistical-learning/estimation mechanism and distinguish it from its closest TSP SDDE antecedent.

The new40-hour comparison runs117 complete cells (three tasks, thirteen methods, three seeds) plus six replays under unchanged scientific gates. It includes fresh HAPPO controls, MAPPO, delayed/online-KL, joint MA-Trace and defensive PG adaptations, full method and three ablations. It remains development, not confirmation and not a promised positive result. Its outputs cannot be used to patch the current V3 claims before final validation.

**Next for V3 is a non-GPU mathematical/reporting correction round, not more environments.** The bounded repair order is in theory_repair_memo.md. This track need not wait for the actor study. Because preserving V3 was explicit, this turn delivers a separate audit rather than silently replacing the existing manuscript or its evidence. No submission clearance is issued until the substantive findings and author metadata are resolved.

## 6. Seven-failure-mode research audit

| Mode | Status | Reason |
|---|---|---|
| Implementation bug | INSUFFICIENT for new actor performance; V3 proof interface issues identified | No new native integration result is invented; mathematical assumptions need repairs |
| Citation hallucination | CLEAR for existence of all37 retained works | Fresh authoritative matches; index outages disclosed |
| Result hallucination | SUSPECTED/CONFIRMED misdescription | EXP-016B saved normalized improvement mislabeled as baseline-relative reduction |
| Shortcut | CLEAR in this turn | No threshold relaxation, removed task, retry, outcome peek or manuscript import |
| Bug-as-insight | CLEAR in this turn | Denominator/proof defects reported as defects, not new scientific discoveries |
| Methods fabrication | INSUFFICIENT for broad claims | Exact continuous solve/cheap exact spectral calibration need narrower interfaces |
| Frame-lock | CLEAR in decision | TSP evidence preserved, new asynchronous route not represented as already positive or ICML-ready |

## Primary sources freshly consulted

- TSP scope: https://signalprocessingsociety.org/publications-resources/ieee-transactions-signal-processing
- SPS author policy: https://signalprocessingsociety.org/publications-resources/information-authors
- SDDE antecedent: https://ieeexplore.ieee.org/document/10909566/
- Lyapunov participation antecedent: https://arxiv.org/abs/2503.00569
- Feedback participation: https://proceedings.mlr.press/v283/cummins25a.html
- Paulin metadata: https://www.research.ed.ac.uk/en/publications/concentration-inequalities-for-markov-chains-by-marton-couplings-/
- Retained eprints: https://arxiv.org/abs/2009.14471 and https://arxiv.org/abs/2003.06709

This check verifies disclosure and claim-to-provenance fidelity. It does not judge whether the experiment was correctly designed, run, statistically adequate, or reproducible by ARS. The separate arithmetic/proof/visual findings above describe their actual scope; no claim is made that the remote160-run study was rerun or re-audited remotely today.
