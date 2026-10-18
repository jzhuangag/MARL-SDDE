# V3 isolated repair: second-pass technical review

## Material Passport and decision

Human-authorized independent repair, 2026-10-18. Routes: academic-research-suite
(targeted revision and evidence checking), IEEE TSP writing, publishable academic
writing, IEEE bibliography style, and PDF visual review. This is NOT a submission
release. Original V3 and every scientific source/result remain untouched. No
incomplete sealed-bridge outcome was read or imported.

Status: REPAIR_IN_PROGRESS; evidence/protocol alignment remains a release gate.
This is not an experimental NO_GO and does not stop the independent native study.

## Substantive source changes and their scope

1. The probe/delay decomposition now telescopes to the SAME full-budget delayed
   fallback certificate used by the main theorem. The previous displayed sum
   telescoped to the zero-delay fallback instead. The delay adjustment is signed;
   it is not claimed to be a nonnegative extra cost in every instance. No saved
   admission decision or experimental gate was recomputed or changed.
2. The main performance proof now conditions first on the complete probe history.
   It explicitly requires action-wise conditional certificates, a declared
   learning initialization/common R0, and cross-instance certificate validity
   for wrong commitments. Multiplying an unconditional risk certificate by a
   correct-decision event would not justify the result.
3. The phase theorem now states the required uniform budget-shift inequality,
   coefficient gap and wrong-commit bound mathematically. A uniform asymptotic
   expansion along a ray alone does not imply a local budget-perturbation bound.
   The proof gives explicit sufficient scale inequalities; BS is an eventual
   admission threshold rather than assuming that the first admission is monotone.
4. Adaptive Gaussian sampling is no longer called stationary merely because the
   underlying process was stationary initially. Both filters have finite expected
   information through L2 recursions, not pathwise bounded Gaussian predictions.
   The common-factor model also no longer claims unchanged marginal variance
   across different theta hypotheses.
5. The 2.319% MPE result is identified as mean regime--seed normalized gain.
   The 8.341% MaMuJoCo result forms the paired regime mixture before normalization.
   These are distinct estimands, not two interchangeable mixture ratios.
6. Equation references are normalized throughout. Duplicate exposition and an
   unreferenced secondary SDDE appendix were removed; proof-critical conditions
   were retained. The PDF is 13 pages with an ordinary IEEE reference-column
   break, no font/margin/spacing compression. Abstract:220 words; keywords:5.
7. The spacing plot uses explicitly labelled categorical persistence positions
   to separate the 0.9 and 0.98 labels. Its numerical ordinates are unchanged.

## Completed checks

- Full latexmk build exits0; zero undefined references/citations, zero overfull
  boxes and three underfull warnings in the current log.
- Current PDF SHA256:9751de9b6d12bb3340be659f76190871ea9152ee71c05e708099ee0d1a465679.
  All13 pages rendered and visually inspected; no clipped/overlapping content
  observed. Corrected risk and spacing figures are legible; reference columns
  are balanced. Rendering is in third_pass_render; preflight is PASS.
- revision_checks_third_pass_20261018.json records every original source/Bib/PDF
  hash unchanged, zero leading whitespace/bare equation references/qquad.
- All36 saved EXP-010B actions have strictly positive containment margins:
  eta>=5.6734676540672174e-5, mu_delta>=0.09001859123854217,
  a_delta>=0.9910577518512226, 1-c>=3.1161071624152825e-6.
  This is a retrospective containment witness for these actions, not a new
  ex-ante experimental constraint or proof of the scalar solver's globality.
- Fresh37-reference existence/metadata evidence and IEEE local-style audit
  from today remain preserved. Raw36-PASS/one article-number representation
  conflict is not rewritten as37 raw passes. Paulin's article79 resolution is
  separate; the BibTeX is unchanged. Fresh arXiv pages confirm provenance of
  PettingZoo2009.14471 and FACMAC2003.06709; proceedings remain the primary
  cited versions. PettingZoo's preprint author order/names differ from the
  proceedings record and must not overwrite that published author list.
  URLs:https://arxiv.org/abs/2009.14471 and https://arxiv.org/abs/2003.06709.

## Remaining concrete release work (do not skip)

The source-to-theorem clock mapping needs a dedicated audit before clearance.
The manuscript's current usable horizon subtracts D physical clock units before
division by b, while run_adaptation_cost_pilot.py:available_updates subtracts
D RETAINED updates after both budget divisions. run_exp016b_pilot.py similarly
sets usable=max(0,scheduled-D) and charges every scheduled message/transition.
For example, with nonbinding messages, environment budget20,b=4,D=2, these
conventions give4 versus3 usable updates. Neither can silently stand in for the
other. This is a reporting/model-alignment finding, not permission to alter old
saved code, gates, actions, or outcomes.

Next repair: distinguish the recursion's update-lag bound, any withheld terminal
update count, and physical terminal-latency charge; map each retained study to
its actual convention. Also verify the retained-transition endpoint and mixing
clock against actual sample collection, and distinguish scalarized h+q+b cost
studies from separate dual budgets. Recompute theoretical certificates from
immutable saved moments only if needed, with new audit outputs; no native rerun
or revised performance selection. The corrected decomposition itself must be
checked against the old admission implementation before implying exact identity
between the repaired theoretical controller and every saved simulation.

Complete full citation-context/nearest-work comparison verification, sentence
line lint and submission metadata checks as separate gates. No full-text
originality or TSP scope/admissibility certification is claimed here. Real
author/coauthor information still must come from the human. Keep the new actor
science independent and preserve every old positive and negative record.
