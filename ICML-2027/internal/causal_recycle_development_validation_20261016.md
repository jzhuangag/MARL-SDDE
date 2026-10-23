# Complete-algorithm development: verified NO_GO and substantive correction

## Completed evidence

The frozen study at d543363 completed all 60 registered learning cells, six
semantic replays, two byte-identical analyses and the complete all-method native
and communication return figure. Total driver CPU was 8675.40625 seconds, below
the registered 12000-second bound. No GPU/HPC4 was used, no failed cell was
retried, no task/seed/method was removed, and no frozen source or gate changed.

Relevant software regression: the four frozen causal-recycle interface tests
pass, and the four new deterministic current-packet protocol/algebra tests pass.
The independent offline audit was also executed twice in disjoint roots; all
four resulting CSV/JSON files are byte-identical. The all-method return PNG was
visually inspected. No return trajectory was altered or selectively smoothed.

Offline verification is reproducible with
`experiments/audit_causal_recycle_complete.py --output <new-unused-root>`.
The checked output is `results/causal_recycle_development_audit_20261016`.
It verifies frozen source/runtime hashes, 60+6 unique mappings and file hashes,
finite values, exact native/communication/optimizer/proxy costs, independently
generated service times, source-version identities, latest-packet eligibility,
terminal incoming reservations and semantic replay. The frozen integrity
analyzer additionally checks identical initial policies, controller cost/queue
recurrences and exclusion of evaluation episodes from controller input.
Training/evaluation totals are 960000/288000 native transitions for science,
plus 96000/28800 for replay. These are counted separately, not hidden overhead.

## Frozen task-level decision

| Task | Strongest control | Native AUC change | Positive seeds | Final-return change | Communication AUC change | Control learning signal |
|---|---|---:|---:|---:|---:|---:|
| Spread | HAPPO LR x2 | -1.149852% | 0/3 | -2.131208% | -1.448020% | -5.952969% |
| Reference | HAPPO LR x2 | -0.778903% | 1/3 | -0.360229% | -0.622595% | +4.939511% |

The preregistered decision is NO_GO. BOTH native AUC means are negative, so this
is not merely an overly strict positive margin. Spread also fails the learning
signal and final-return requirements. Reference passes those two requirements
but fails mean AUC and positive-seed frequency. No confirmation or manuscript
admission is allowed. Two native MPE tasks are one environment family, not the
requested broad multi-family evidence.

## Full-mechanism diagnosis, with evidence separated from hypotheses

1. Learning signal: one actor update/backward pass per bank and 40 rounds did
   not meet the registered Spread learning gate. HAPPO LR x2 ranks strongest
   in both tasks. This suggests that base learning response deserves repair
   before another scheduler contest; it does not prove one-pass learning is
   the sole cause. Different checkpoint world seeds confound unpaired initial-
   to-final interpretation; a frozen-policy same-world control was absent.
2. Old-gradient opportunity: service and actual joint-version staleness pass
   audit, but not all launched work is usable. Spread launches 120 gradients,
   with mean 74.333 latest-eligible packets; Reference launches 80, with mean
   50.667. Collisions/supersession discard work regardless of the controller.
   No counterfactual maximum is inferred from this count or a noisy menu.
3. Queue pricing and sunk work: full method accepts mean 25.333/74.333 eligible
   Spread packets and 16.667/50.667 Reference packets. About two thirds of ready
   latest packets are declined. Yet proposal Adam calls are 148.667/101.333,
   and all 120/80 source gradients remain paid. Total communication decreases
   only 7.61%/10.75% versus the fresh HAPPO anchor. Rejection does not reclaim
   the main source/trajectory/advantage/incoming computation and messages.
4. Value-price scaling: mean absolute selected normalized score is
   .00047738/.00032175, while mean full-broadcast queue price is
   .01515258/.01910543. These are diagnostic magnitudes, not proof that a
   retrospectively chosen V or looser budget would beat the baselines.
5. Score transfer: current-bank feedback arrives too late to choose its own
   receipt action in the frozen design. The optimistic weights describe
   historical action patterns, not current packets. Removing the old hint
   yields exactly identical actions/final policies for two of three Spread
   seeds, but not all Spread seeds and not Reference. The hint is therefore
   neither universally inert nor demonstrated to identify useful current
   decisions. An estimator-only variance pass cannot repair this timing.
6. Comparator/timing: fresh sequential HAPPO immediately uses new data; it is
   an ideal zero-transport anchor, not the same ready-packet information set.
   Full control scores before committing while immediate asynchronous controls
   commit before the next bank. Every original comparison remains reported;
   the mismatch motivates a prospective matched control, not posthoc removal
   of the strongest anchor or threshold relaxation.

## Concrete correction completed, and what remains

Implemented a separate current-packet value controller. Probe allocation is
predictable; the current actual-return vector selects the current receipt
action; selection bias costs at most 2V sqrt(S/m) in conditional expectation.
A joint (m, lambda) probe loss follows from Young's inequality and receives
unbiased full-vector variance feedback. Two resource queues price all common,
probe and candidate costs, with immutable candidate/commit identity checks.
The derivation and finite-time conditional drift inequality are in
`current_packet_value_design_20261016.md`. Four deterministic interface/algebra
tests pass. This is a substantive causal correction, not a new positive
performance label. It has not yet been integrated or validated on native MARL.

Precision is still a serious feasibility issue, not solved by that derivation.
Using the OLD mean sample traces .00013022/.00003488 only as diagnostic point
values, m>=4S/epsilon^2 gives approximately 521/140 episodes for epsilon=.001,
or 21/6 for epsilon=.005. These are not certified requirements, since the
observed average trace is neither a population bound nor uniform over new
policies. They do show why a new selector must establish useful discrimination
at affordable cost, not merely report a variance reduction or small error in
absolute units. Instantaneous normalized gain tolerance is not equivalent to a
1% trajectory-AUC gain; do not convert one into the other.

The next bounded stage is native standard-learner integration plus useful-score
precision/cost feasibility, not an immediate repeat of the failed 60-cell run.
Learning-response pilots, if needed, must be explicitly development and not
called outcome-free qualification. New scientific experiments need separate
source freezing, fresh seeds, matched eligible reference and unchanged
prospective comparisons. No return advantage or ICML readiness is asserted.

## Interpretation audit

- All tasks/seeds/methods remain visible; no pooled positive metric hides a
  failing task, seed pattern, or communication result (Simpson/aggregation).
- Three development seeds do not support universal inferiority or superiority;
  task-group means are not per-seed causal evidence (ecological inference).
- No outcome-selected positive cells or exclusions are used (selection/collider).
- Base-rate fallacy is not applicable to this continuous paired metric; no
  classifier precision/acceptance interpretation is introduced.
- Checkpoint world differences can mimic learning changes; no posthoc endpoint
  selection or regression-to-the-mean rescue is used.
- Complete predefined curves and end points are retained, not a favorable
  time slice (cherry-picking/survivorship).
- Multiple baselines and posthoc diagnosis do not create confirmation or a
  valid one-sided significance claim (look-elsewhere/HARKing).
- Ablations suggest mechanisms but do not identify an unmeasured causal effect
  of queue pricing across diverging policy trajectories (causal overreach).
- Variance is not a confidence upper bound; selected scores are not unbiased
  after data-dependent argmax (estimation/selection bias).
- Semantic replay confirms reproducibility, not scientific truth or external
  validity; native MPE evidence is not a cross-family MARL claim.
- Conditional own-trajectory drift/menu guarantees are not independent-learner
  HAPPO superiority, gradient stationarity or physical backlog stability.
