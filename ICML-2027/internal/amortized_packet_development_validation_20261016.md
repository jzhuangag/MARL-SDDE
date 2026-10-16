# Complete amortized MARL development: final bounded NO_GO

## Material Passport

ARS experiment-agent/validate. Verification Status: VERIFIED for the complete
registered60-cell payload study, six registered semantic replays and byte-identical
analyses. Scientific conclusion: development NO_GO; no confirmation/manuscript
admission or ICML-readiness claim. Three seeds/task are an exploratory screen.

## Completion, integrity and provenance

Scientific source/preregistration dd279792735ecf0c91b4bb3e86420c3c7a732736 was
pushed before native data. Complete driver exited0, PID28264 is gone, all60 cells
and six fixed replays completed; total active CPU8014.6875s below12000. No retry,
source/gate/seed/baseline change, GPU/HPC4, cleanup or manuscript edit occurred.
Lock SHA256: bb8f2f1c854caa77e80878c6d9ce8eba4898fb6646821896678f9d2fdff17a9d.

The completed driver owns two byte-identical metrics.csv/summary.json/integrity.json
outputs. Independent post-completion audit checks every66 payload hash and exact
mapping, registered-commit source bytes, current pinned runtime, complete ledgers,
unique independent service/versions/receipts, five recomputed PPO epochs, server
moments, pre-bank representation/meta, exact disjoint tail/evaluation exclusion,
same initial policies/world controls and every registered non-timing semantic
replay. It reconstructs native AUC/final comparison from raw completed records.
It does not rerun the native experiment or overwrite frozen outputs.

Science consumed720000 native training and180000 evaluator/control transitions;
registered replays add72000+18000, total990000 joint environment transitions.
Agent-row totals are separately stored, not confused with joint transitions or
logical parallel ticks. All stored costs are in the raw ledgers and the derived
cost_counts.csv. Logical bytes/declared work proxies are not wire bytes/FLOPs.

Evidence: results/amortized_packet_development_20261016 with preserved _replay,
_analysis_a and _analysis_b siblings. Derived audit:
results/amortized_packet_development_audit_20261016/validation.json and
all_method_table.csv. Only the six prospectively registered cells were replayed;
do not describe all60 learning cells as individually replayed.

## Frozen task decisions

| Task | Strongest of7 controls | Native AUC gain | Positive seeds | Final-return change | Baseline learning | Independent gain sum mean | Independent positive seeds |
|---|---|---:|---:|---:|---:|---:|---:|
| Spread |fresh HAPPO x1|-0.867157%|1/3|-7.045926%|5.212161%|-0.064809|1/3|
| Reference |online KL-adaptive|-0.178870%|1/3|-2.796061%|8.429456%|1.034459|2/3|

Both tasks fail the original>=1% AUC,>=2/3 positive paired seeds and final>=-2%
gates. Spread also fails independent selected-response mean/seed gates. Both
pass baseline learning, nondegenerate updates, cost ratios and integrity/replay.
Independent nonidle minima are16/14 receipts, well above the original4 gate.

This is not merely an overly strict1% threshold: both primary mean differences
are negative, and both endpoint comparisons miss the separate noninferiority
gate. The baseline-learning control now passes in BOTH tasks, so weak baseline
learning cannot explain away this result. All seven controls and both ablations
remain reported. No task removal, pooled rescue or posthoc primary-method swap.

## Complete resource comparisons

| Task | Full payload bytes / strong control | Full messages / strong control | Full work proxy / strong control | Full mean CPU / strong control |
|---|---:|---:|---:|---:|
| Spread |7923814.667 /6717984 (1.179493x)|270 /240|340089.073 /362032.556 (0.939388x)|157.552s /141.411s|
| Reference |6318506.667 /5176560 (1.220600x)|193.667 /153|269730.385 /250599.180 (1.076342x)|103.120s /96.609s|

The registered1.5x byte/work gates pass. Actual CPU differs from the work proxy:
the negative Spread proxy overhead does NOT imply hardware speedup. Per-cell CPU
is descriptive/non-reproducible timing, including process initialization effects.
Full's early probes/tail queries add communication but no EXTRA native training
episodes: both are slices of the same already charged20-episode banks. Therefore
it is incorrect to blame lost native learning on an uncounted sampling budget.

Shared-message-horizon AUC (full / strongest): Spread -26.467984 /-26.434998,
Reference -26.905871 /-26.702313. Shared-payload-horizon AUC: Spread
-26.494035 /-26.385757, Reference -26.903074 /-26.692480. Thus communication
indexing does not turn the primary failure into a strong-baseline win. Whole
all-method resource tables, own-horizon metrics and shared horizons remain stored.

## Full-mechanism diagnosis: evidence versus inference

1. **Learning signal:** strong baselines learn5.21%/8.43% against same-world
   frozen policies. Native sampling/initialization/evaluation-role gates pass.
   The experiment is not dismissed as an inert task or an operational failure.
2. **Delayed-update primitive:** genuine joint-version staleness is present;
   full-method mean arrived-packet ages span1.0--1.81 versions by seed. The
   direct eligible-all primitive trails fresh/strong online controls. Delay,
   update cadence and sequential-factor differences are bundled here; this
   study does not identify a pure causal "delay-only" effect.
3. **Finite-menu opportunity:** the menu contains unit-proposal parameter rays,
   not arbitrary corrected gradients or fresh sequential HAPPO programs. Its
   sampled scores are not a true oracle. Only the SELECTED action has independent
   tail feedback, so we cannot infer the best candidate's true return, prove
   absence of headroom, or use a noisy maximum as an oracle upper bound.
4. **Score transfer / selection optimism:** Spread's training-selected gain sum
   averages6.404324 but the independent counterpart averages-0.064809; optimism
   is6.469133. Reference averages3.252142 versus1.034459; optimism2.217683.
   These are descriptive three-seed realized contrasts, not population bias
   estimates with a claimed confidence certificate. Reference's independent
   response gate passes, yet its full learning still loses to KL-adaptive. Thus
   positive local selected gains alone do not ensure strongest-baseline learning.
5. **Probe/copy tax and risk geometry:** exact one-unit PPO amortization works;
   the former three-PPO duplication is gone. Nevertheless communication rises
  17.95%/22.06%. Across eligible full-method frames, m=16 occurs88/131 times;
   response-basis geometry occurs95/131. Recorded selected expected-risk feedback
   means range7.06--329.14 per eligible frame, while mean training-selected gains
   are about.13--.40. These sample quantities show that the valid conditional
   bound is not a numerically persuasive positive performance guarantee here;
   empirical variance/remainder feedback is never promoted to a population UCB.
6. **Queue pricing:** full and no_price have identical action AND committed-policy
   paths in ALL six task/seed cells, and identical native learning curves. At
   every full-method bank the priced candidate index equals the unpriced score
   argmax. Queues can be positive (communication maximum.521, compute maximum.169)
   without changing the action. Most already paid program/proposal work is common
   sunk cost; positive alphas share the same broadcast/tail-data cost for a given
   active subset. Lyapunov prices therefore have no demonstrated decision-level
   benefit in this evaluated design. Raising prices posthoc is not a scientific
   rescue and cannot fix the observed noisy-score/update-direction problem.

Full does improve native AUC relative to simplest eligible-all by about1.15% in
Spread and1.04% in Reference. This is local component evidence, not a win over
the strongest controls. Full-vector-only improves primary AUC relative to full
in both tasks; in Reference it is only about0.182% above the strongest control,
below the original1% gate, with final return still worse by about2.80%. It cannot
be substituted for the frozen primary method after seeing these outcomes.

## Final bounded design judgment and stop

NO_GO for widening, confirming or publishing THIS receipt-only exhaustive
parameter-ray selector as the ICML primary algorithm. The one amortization
correction was implemented and tested end to end; its lower primitive cost does
not solve selection optimism or improve the stale update's available directions.
Neither more environments/seeds, looser gates, posthoc alpha catalogues, stronger
prices nor another variance-only experiment is authorized as an automatic rescue.

The concrete next algorithmic requirement is to change the **stale-data update
primitive**, with an explicit joint-policy KL/off-policy correction that controls
the committed displacement, rather than keep maximizing a noisy menu of the same
PPO rays. Resource/Lyapunov actions must also have genuinely avoidable prospective
cost before proposal/launch, if resource allocation remains a claimed contribution.
Merely adding a KL cap to the same rays is insufficient: online_kl already does
that and is a registered strong control. A successor must change the actual
behavior/joint-version-corrected update program, not rename that baseline.
This is a substantive research direction/feasibility requirement, NOT an already
proved novel method or a promised positive result. A new controller would require
its own novelty/interface argument and human-authorized bounded plan; no new
numbered run or auxiliary-test series is launched from this stopped workflow.

All frozen evidence, old NO_GO studies and TSP manuscripts remain preserved.
The current heartbeat terminates after this verified report is pushed.

## Statistical interpretation and11/11 fallacy scan

No significance, population superiority/inferiority or95% coverage claim is
made from three seeds. Effect sizes are paired normalized AUC/final differences;
seed, not episode/frame/agent, is the comparison unit. Plot bands are one SEM.
There are no posthoc inferential p-value tests requiring a multiple-test rescue.
All predetermined practical gates are applied and every method is shown.

- Simpson: both task-level primary means fail; no pooled reversal claim.
- Ecological: no team-average result is attributed to individual agents.
- Berkson: all60 registered cells are retained; qualification returns not ranked.
- Collider: primary analyses do not condition on favorable scores/accepted returns.
- Base rates: no classification/accuracy claim; not applicable.
- Regression to mean: matched same-world frozen-policy controls are reported.
- Survivorship: no dropped seed/task/method/payload; six fixed replays retained.
- Look-elsewhere: all controls/ablations/failed gates remain visible; no winning
  ablation is relabelled as the preregistered primary algorithm.
- Forking paths: source/config/seeds/gates were pushed before data and unchanged.
- Correlation/causation: component diagnosis is descriptive except exact software
  path/cost identities; no unmeasured optimal-menu or delay-only causality claimed.
- Reverse causality: tail data follows selection and never controls that receipt;
  positive local gains are not interpreted as proof of long-horizon superiority.
