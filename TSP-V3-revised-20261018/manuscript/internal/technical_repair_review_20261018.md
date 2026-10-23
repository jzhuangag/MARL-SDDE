# V3 targeted technical repair — verification review

Material Passport: authorized isolated V3 revision, 2026-10-18. Original V3,
all saved outcomes/gates, and the independent asynchronous MARL study are preserved.
This is a technical pre-submission review, not an actual TSP editorial decision.
Decision: TECHNICAL_REPAIRS_VERIFIED_AUTHOR_RELEASE_PENDING. The final gate passes
13 pages,218 abstract words/five keywords, zero undefined/overfull items,27
underfull warnings, all fonts embedded and all13 pages visually reviewed. The
underfull warnings correspond to narrow-column spacing, not clipping. Final PDF
SHA256:60da714cbd7fc6fb43f3d00b5cfa6c18e9c8d390030b3dcf6b3099a78df84350.
The machine decision and final PDF/render hashes are in
`submission_release_check_20261018.json`; do not substitute an earlier working PDF.

## Judge record and review scope

ARS re-review mode is applied inline to the original audit/repair ledger, revised
main text/appendices, source-to-clock audit, saved evidence checks and fresh
reference records. The same primary agent performs the targeted revision and
verification; there is no claim of five independent reviewers, external-model
verification, full-text plagiarism clearance or a calibrated acceptance probability.
The earlier internal audit has no independent-reviewer provenance block.

Skill-mandated single-family disclosure, quoted from the re-review protocol:
“This verification round ran on the same model family that drove the revisions;
over-optimization to this judge's latent biases is possible (Ren et al. 2026,
arXiv:2607.13104 §8.1.2).” This protocol quotation is not an added manuscript
reference or independently used research evidence.

## Required revision traceability

| Required issue / author's repair claim | Verification location and reasoning | Verdict |
|---|---|---|
| Distinguish stale-state delay from terminal exclusions and physical latency | Model dual-budget subsection defines D, u, ell separately. The usable horizon subtracts u AFTER budget divisions; ell is a physical charge BEFORE division. Main decomposition compares the same full-budget delayed fallback and charges each overhead once. | FULLY_ADDRESSED within the declared protocols |
| Map repaired clocks to actual old implementation | `clock_protocol_audit_20261018.json` checks 1,600 pure integer cases and all36 saved affine selected actions without native execution. EXP010B/convergence use scalar h+q+b, u=ell=0; EXP016B uses dual budgets, u=D, ell=0. Old implementation/gates remain unchanged. | FULLY_ADDRESSED |
| Correct retained-transition endpoint and mixing clock | The model uses a Markov augmentation containing TD transition pairs. Old starts0,b,... have final endpoint(K-1)b+1<=Kb; the saved exact pair-chain mixing power is b. Fresh masks do not increase total variation, and the q+1 source chains admit the union bound. First-tuple conditioning/independent stationary initialization is stated. | FULLY_ADDRESSED |
| Replace impossible pathwise Gaussian information bound | Appendix F proves uniform L2 bounds on the adaptively sampled state and both candidate predictive means, then bounds EXPECTED Gaussian KL. Bounded stopping gives expected total information; no uniform bound over arbitrary unbounded histories is asserted. | FULLY_ADDRESSED |
| Close existence / solver interface | Positive non-strict certification margins on a compact positive-step interval give closed, nonempty certified domains; empty domains have explicit fallback/no-update behavior. Local scalar candidates are not global certificates. Main admission uses frozen ACTUAL selected-action upper risks, eliminating a hidden global-solver prerequisite. | FULLY_ADDRESSED |
| Repair delayed-window and moving-target proofs | Appendix A proves the positive-part replacement recursion about d/(1-c), including incomplete terminal blocks and zero negative-time increments. Moving-policy tracking explicitly requires the uniform mixed-history drift or the corresponding reset/no-old-update protocol. | FULLY_ADDRESSED under the stated history condition |
| Correct performance/safety baseline | Main theorem and Appendix E bound loss against the fallback CERTIFICATE, conditional on full probe history with uniform certificates for correct and wrong commitments. Calibration coverage alone is not presented as unconditional unbounded-MSE safety. | FULLY_ADDRESSED |
| Repair finite-budget threshold scope | Phase hypotheses separately state the 1/s expansion, budget-shift inequality, high-instance selected-action gap, low-instance coefficient ordering and wrong-action penalty. Appendix F derives a concrete independently initialized Gaussian-mean realization and the missing scale/safety terms; it does not infer a 1/s rate from a fixed positive-step floor. | FULLY_ADDRESSED for this separated risk class |
| Correct reported risk denominator | Main simulations and Fig.3 use information-only risk:35.24% and9.79%, with subset32.94/43.72/30.13%. Saved gate, outcomes and original analyzer files are not rewritten. | FULLY_ADDRESSED |
| Separate unequal transfer estimands | MPE2.319% normalizes regime/seed pairs before averaging; MaMuJoCo8.341% forms each seed's equal-regime mixture before normalization. Comparators remain strongest SINGLE fixed actions, not arbitrary strong asynchronous algorithms. | FULLY_ADDRESSED |
| Correct scope/cost claims | Rank-one aggregation O(qd) is separated from spectral moment calibration. EXP016B's consecutive Gaussian probe/learning data and averaged FIFO learner are explicitly empirical mechanism evidence, not a test satisfying all repaired conditional-theorem hypotheses. Robust shields preserve the action; positive ablation gains concern terminal parameter risk. | FULLY_ADDRESSED through exact claim boundaries |
| Verify closest-work comparison and bibliography | All37 identities were freshly resolved today, with raw36 PASS plus a separately resolved article-number representation. Context matrix records primary abstract/intro-level support and access limits. Prior Lyapunov bound-driven client/power control, adaptive rates and SDDE worker activation are acknowledged; no absolute priority claim is made. | FULLY_ADDRESSED at the recorded claim/context scope |

The phase theorem and critic theorem remain conditional mathematical results;
their declared assumptions are not certificates of general neural MARL behavior.
The independent 140h study is neither evidence for this revision nor a prerequisite
to releasing it for author review.

## Formatting and bibliography judgments

The current SPS author instructions require a 10-point two-column manuscript,
one-inch margins and at most13 pages for an initial regular submission, including
proofs and references. The revised source enlarges the default IEEEtran margins
to one inch. Repeated exposition and a manual reference-column break are removed;
font sizes, line/reference spacing and proof-critical assumptions are not shrunk.
All substantive proofs remain in the counted manuscript.

The abstract has no abbreviations, remains within150–220 words, and has exactly
five keywords. Source lint checks left alignment, one sentence per physical line
and the required equation-reference syntax. The final artifact gate records the
compiled page count, warnings, embedded fonts and the13 visually reviewed pages.

The37-entry BibTeX remains byte-identical to the original. Paulin is article79,
not an issue79; the existing local IEEE rendering prints “no.79”. The authoritative
identity/volume/article/pages are resolved in `citation_resolution_20261018.json`.
This is a rendering-style judgment, not an unresolved work/DOI conflict. No
ambiguous BibTeX entry has been silently rewritten. Self-citation classification
cannot be certified until real author names are supplied.

Official policies checked2026-10-18:
- https://signalprocessingsociety.org/publications-resources/information-authors
- https://signalprocessingsociety.org/publications-resources/ieee-transactions-signal-processing

The TSP case is adaptive statistical learning/policy evaluation with a resource
design theorem; a fabricated communications application is neither required nor
added. Journal scope/novelty acceptance remains the editor's judgment.

## Author-release boundary

The machine gate, not an earlier intermediate checklist, determines current
technical artifact status. Authentic author order, affiliations, corresponding
author details, ORCIDs, funding/disclosure facts, related prior submissions and
all-author consent remain human inputs. Insert these only from supplied facts,
then rebuild and recheck pagination. Do not submit “Anonymous Authors” or claim
that an internal technical review guarantees TSP acceptance.

No new native data were run for these repairs. No old result was retried,
re-ranked, removed or altered, and no incomplete 140h return was inspected.
