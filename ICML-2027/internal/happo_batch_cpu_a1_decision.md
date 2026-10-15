# CPU decision: asynchronous HAPPO policy-block allocation

## Material Passport

- Date: 2026-10-15.
- Experiment: `ICML27-HAPPO-ASYNC-BATCH-CPU-001-A1`.
- Frozen scientific source: `65bb5ca4a2895384c005e52f980da718edf2e214`.
- Scope: end-to-end cooperative MPE learning, two development and four holdout
  seeds per task, six controls and an outcome-aware diagnostic.
- Verification: complete scientific execution and registered deterministic
  replay; internal feasibility evidence, not manuscript evidence.
- AI assisted implementation and analysis; no human-read literature assertion
  is made by this result record.

## Decision

**Stage A fails. Stage B and GPU performance experiments are not authorized by
this experiment.** All 80 registered payloads completed with exit code zero.
Every frozen integrity check passed, the six registered payload replays match
all 18 scientific files byte for byte, and the two complete analyzer outputs
match for `metrics.csv`, `summary.json`, and `integrity.json`.

Ten of the sixteen frozen scientific gates pass. The six failures concern AUC
improvement and positive-seed consistency. Every final-return lower threshold,
noncyclic-action threshold, and fresh-HAPPO learning threshold passes.
The result does not establish a useful advantage over the strong comparator.

| Task | Development-selected comparator | Diagnostic AUC gain | Positive holdout seeds | Final-return change | Fresh-HAPPO learning from initialization |
|---|---|---:|---:|---:|---:|
| Spread | HAPPO full cadence | 0.7115% | 3/4 | 0.6304% | 5.4797% |
| Reference | History surrogate | -0.0334% | 2/4 | -0.1591% | 4.4604% |

The separate comparison with cyclic HAPPO gives AUC gains of 0.3985% on spread
and 0.4946% on reference; positive-seed counts are respectively 2/4 and 3/4.
The preregistered minimum is 2% in both tasks, so it is not met.
Percentages follow the registered per-seed signed-return normalization, not
ratios of task-averaged returns. Four holdout seeds form a feasibility screen;
no significance or universal-dominance claim follows from these numbers.

## What the failure diagnoses

### 1. Search selection does not transfer reliably to independent episodes

The menu includes the zero update. Consequently, maximizing search return
always gives a nonnegative search margin, including when the winner was
selected through noise. The registered independent checkpoint diagnostic
measures the selected update against its actual pre-update policy on disjoint
evaluation episodes:

| Task | Mean selected search margin at checkpoints | Mean independent increment | Positive independent increments |
|---|---:|---:|---:|
| Spread | 0.197889 | -0.000836 | 11/24 |
| Reference | 0.158783 | -0.038434 | 11/24 |

These are six checkpoints per seed, not 24 independent training seeds.
The discrepancy is evidence that the search ranking does not reliably
transfer at the available precision. It cannot distinguish zero intrinsic
allocation value from an opportunity that this noisy search fails to select.
Calling this finite, one-step greedy menu a mathematical oracle ceiling would
be incorrect. The stopped decision applies to the tested implementation.

### 2. The search cost is already too large for the observed improvement

Every method receives 9,600 training joint transitions per seed. The menu adds
mean search costs of 443,200 joint transitions on spread and 111,200 on
reference: respectively 46.17 and 11.58 times its training allowance. It also
uses extra counterfactual optimizer work and 9,600 independent diagnostic
transitions. The charts and ledgers report those costs explicitly.
The sub-1% AUC changes are not deployable equal-total-resource superiority.
Increasing the search budget without a precision-versus-value calculation
would compound this issue.

### 3. Reference contains a deterministic delay/schedule alias

For two actors with one cyclic launch per block, the selected actor is
`block mod 2`. Rolling the ages `[0,8]` by the same block index assigns age zero
to exactly that actor. Thus the nominal stale-cyclic comparator launches zero
stale packets. Age-first makes the same choice, and fresh parallel with one
actor is the same update as cyclic HAPPO. The four methods have numerically
identical curves on all four holdout seeds.

This is an experimental-design defect missed by the original qualification:
exact accounting does not prove mechanism separation. Reference cannot be
presented as an independent validation of stale-versus-fresh scheduling.
It remains in the result set and its frozen failure is preserved.

### 4. A modest learning signal exists, but the sampled training phase is narrow

Both HAPPO learning gates pass. Therefore the failure cannot be dismissed as
the baseline learning nothing. The budget covers 24 blocks and 9,600 joint
transitions, which supports an early-training screen rather than a conclusion
about every learning phase or standard full-budget MARL performance.

### 5. Old Adam-state replacement is a confound that has not been isolated

The tested stale packet includes an old source-version parameter increment
and Adam state. Applying it replaces the server actor's optimizer moments with
the old state. That is a registered algorithm choice, not an accounting
failure. Its contribution to performance was not separately randomized here,
so it must not be asserted as the observed failure's proven cause.

## Substantive correction already implemented

`experiments/causal_packet_ledger.py` supplies a new causal transport interface:

1. A launch creates a unique job with the entire source policy-version vector,
   due time, computation charge, and reserved message bytes.
2. Each actor has at most one in-flight job; service times use a separate
   per-actor launch-ordinal stream, independent of the cyclic action clock.
3. Completion transmits and charges a packet once, whether accepted or rejected.
4. Each packet can be consumed once. Rejection refunds neither compute nor
   transmission. All agents' native environment steps are charged.
5. Packets carry raw gradients; server optimizer moments remain at the server.
   Receipt-time control can choose an eligible packet and its step, while
   pre-launch control owns future computation and communication expenditure.

Four transport tests exercise rejection charges, unique completion/consumption,
overlapping jobs and cross-actor policy motion, and delay-stream independence.
The complete ICML CPU suite passes 32 tests. This verifies the corrected
interface; it is not a new learning-performance result or a novelty claim.

The associated theory and the next experimental entry criteria are in
`causal_packet_redesign_after_a1.md`. The next result must use the corrected
event model, an observable score compared with a matched reference action,
and a resource-feasible precision calculation. No new scientific run is
authorized merely by assigning another experiment identifier.

## Statistical interpretation checks

All eleven ARS categories were reviewed:

| Check | Assessment |
|---|---|
| Simpson's paradox | Both task-level comparisons are shown; no pooled positive claim substitutes for a failed task. |
| Ecological fallacy | Inference is limited to task-level four-seed feasibility; checkpoints are not treated as seeds. |
| Selection/Berkson bias | The development-selected comparator is fixed by the original rule; all registered payloads remain. |
| Collider bias | No post-treatment covariate adjustment or outcome-based sample filtering was applied. |
| Base-rate neglect | Positive-seed and checkpoint-increment denominators are explicit. |
| Regression to mean | No seeds were selected for extreme returns; initialization and concurrent controls are retained. |
| Survivorship bias | All 80 registered A1 payloads completed and enter analysis. Superseded pre-A1 payloads are documented separately. |
| Look-elsewhere effect | All frozen gates and methods are retained. Diagnostic search maximization bias is explicitly identified. |
| Forking paths | A1 was frozen before its outcomes; gates and task membership are unchanged. Subsequent diagnostics are labelled descriptive. |
| Correlation/causation | Search-transfer and optimizer-state explanations are distinguished from experimentally isolated causes. |
| Reverse causality | Delays and action roles precede outcomes; no return-based ex-post configuration changes are used. |

No p-values, post-hoc power claim, or equivalence conclusion is introduced.

## Reproduction and provenance

Primary run: `ICML-2027/results/happo_async_batch_a1_scientific_20261015`.
Replay: `ICML-2027/results/happo_async_batch_a1_registered_replay_20261015`.
Final analysis: `ICML-2027/results/happo_async_batch_a1_analysis_20261015`.
Analyzer replay: `ICML-2027/results/happo_async_batch_a1_analysis_replay_20261015`.
Supplementary diagnosis: `ICML-2027/results/happo_async_batch_a1_diagnosis_20261015.json`.
Figures: `ICML-2027/results/happo_async_batch_a1_figures_20261015`.

The figure plots every registered holdout method without smoothing. Its x-axis
is training joint-transition fraction; separate costs show paid search and
diagnostic transitions. Four-seed minima/maxima are preserved in the exported
curve table. Neither figure is admitted into a manuscript as positive evidence.

Byte-identical analyzer SHA-256 values:

- `metrics.csv`: `2f5d2bfa0d80df80f4ab64cd080ff4e77800ef314cc9822c0ce4a96d97ba6af9`.
- `summary.json`: `3f04b13427105dc59119b35223eaf2e5cdb476753bb1e56553f503e1b16bf685`.
- `integrity.json`: `bbf6947af9a9e407e777975b7639b14b265b39dab2e83287e3433456ebac0935`.

The frozen TSP manuscripts and earlier experiment results are untouched.
