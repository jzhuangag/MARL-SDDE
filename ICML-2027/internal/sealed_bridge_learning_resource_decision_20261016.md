# Complete sealed-prefix MARL: terminal resource decision

## Material Passport

ARS experiment-agent/validate,2026-10-16. Verification Status: ANALYZED;
independent hash/source/runtime/cost/ledger/forecast reconciliation PASS.
This is a cost-only resource decision, NOT a native scientific replay,
performance validation, learning failure, ICML-readiness or paper admission.
The bounded registration is terminal RESOURCE_NO_GO. No science-admission
commit or117-cell scientific study was started; no scientific seed was used.
All frozen sources, settings, tasks, methods, gates, prior results and TSP
manuscripts remain unchanged. No GPU/HPC4, cleanup or retry occurred.

## Clear outcome

All39 registered method-by-task cost cells completed successfully and the
single owner exited0. Measured complete workflow CPU:1033.421875s (17.224min),
within its3000s profiling cap. The frozen conservative forecast is110529.41796875s,
or30.7026 CPU hours, versus the registered total cap12000s (3.3333 CPU hours).
It exceeds that cap by9.2108x. The driver correctly stopped BEFORE science.

This is not evidence that the proposed algorithm loses to HAPPO/MAPPO, and is
not a scientific NO_GO. Numerical returns/gradients/selection-risk feedback
were neither retained in profile artifacts nor ranked/tuned. Those values
cannot enter V3 or an ICML paper as positive evidence. Complete learning,
equal-cost advantage, useful long-horizon overlap and strongest-reference
return superiority remain UNTESTED.

## Provenance and checks

Registration/source commit pushed before native data:
27cd2874a86b285aeb3e7d08b013d83b29c937a3.
Lock:internal/sealed_bridge_learning_lock_20261016.json;
SHA2568e5aca6054c1bc78096da61eb3b4a8000d099c1ae91e57321e6014bc3ccf8183.
Original root:results/sealed_bridge_learning_cost_20261016.
Execution session44508, driver PID27472; process is gone and the session
returned exit0 with state=complete,decision=RESOURCE_NO_GO.
execution.json is the immutable START record; workflow_complete.json is the
terminal record. It is not an active job because the start record says running.

The actual command is preserved in execution.json:
`python ICML-2027/experiments/sealed_bridge_learning/workflow.py --lock E:/HKUST-study/vin/SDDE/ICML-2027/internal/sealed_bridge_learning_lock_20261016.json --root E:/HKUST-study/vin/SDDE/ICML-2027/results/sealed_bridge_learning_cost_20261016 --phase profile`.

Independent verifier:internal/verify_sealed_bridge_learning_cost_20261016.py.
It reads ONLY the registered cost/protocol artifacts and runtime/source files;
it never constructs/steps an environment, invokes the learning driver, or reruns
the profile/scientific analyzers. Both fresh output roots
results/sealed_bridge_learning_cost_20261016_validation_a and_validation_b
contain byte-identical integrity.json and decision.json:

- integrity.json:ee7f625d9dcb7f92794069d59927134f8721a4ea58c18d152a55f7b99fbb4999.
- decision.json:f933a5d265453fe29d3834cc6d67cc649553f1489a8c26ba2e4bc1b7e46859e8.

Checks:39/39 exact mapping and payload hashes; no missing/extra profile cell;
39 monotone progress records; finite nonnegative costs; full charge-event
aggregation; hard native reservation/debit/drain reconstruction; exact training
and separately charged evaluation budgets; outcome-free persisted schema;
source commit content matches; clean pinned HARL commit and all runtime tree/
version fingerprints match; all7 original qualification source hashes unchanged.
The locked driver additionally audited complete source/initialization/role and
controller recurrence before stripping outcomes. The stripped cost artifacts
cannot independently reproduce those full hidden payloads; their digests and
runtime attestations are preserved, not described as a native replay.

The terminal session contains no crash/traceback and no workflow_failed.json
exists. Benign environment-registration/read-only-input warnings are in the
execution tool history; there is no separate durable full stdout/stderr file.
No rerun was performed merely to obtain a new log.

## Actual paid profile resources

| Resource | Total across39 cells |
|---|---:|
| Native training transitions |873600|
| Benchmark evaluation transitions |81900|
| Same-world frozen-policy transitions |81900|
| All native transitions |1037400|
| Declared neural work rows |24277900|
| Logical payload bytes |264408728|
| Messages |1497|
| Neural/normalizer tensor-copy bytes |8353392|
| Retained gradient-array bytes |171990656|
| Actor Adam calls / SGD calls |450 /85|
| Critic Adam calls |500|

Work rows are declared proxies, not FLOPs/measured CPU; logical bytes are not
physical network traffic. Tensor-copy/retained-gradient bytes are explicit
declared events, not every transient allocator event. Actual CPU is separate.
Every method retains the same source/calibration/full-budget charging rules.

## Why the resource gate failed

| Task | Actual profile cell CPU sum | Raw projected science+replay CPU |
|---|---:|---:|
| MPE Spread |38.796875s|15265.625s|
| MPE Reference |25.015625s|9880.859375s|
| Official HalfCheetah2x3 |962.40625s|47452.1484375s|
| Total |1026.21875s|72598.6328125s|

Whole-workflow CPU is slightly larger than cell sums because imports, guards,
audits and final fingerprints are included. The frozen calculation retains
profile initialization/evaluation costs rather than subtracting them:

    forecast = 1.5 * sum(profile_cell_CPU * science_transitions/profile_transitions
                        * (3 science seeds + registered replay count))
               + already_spent_profile_CPU + 600s analysis reserve
             = 110529.41796875s.

The implementation has13 methods,3 tasks,3 seeds=117 complete learning cells,
plus6 fixed semantic replays. Proposed paid training is54.6m+2.8m=57.4m native
transitions before separately charged evaluation. A1000-step locomotion episode
costs substantially more here than an operational25-step episode. HalfCheetah
accounts for about65.4% of the raw forecast, but the two MPE tasks alone also
exceed the existing cap in aggregate. Removing a task/method or returning to
25-step locomotion would change the scientific question, not solve the registered
resource problem. Neither action was taken.

The1.5 factor and600s reserve were frozen before data. This linear planning
forecast is not a statistical upper confidence bound, measured future runtime,
or proof that a faster implementation is impossible. Long-run option frequencies,
initialization amortization and host load can change actual costs. It nevertheless
fails the predeclared admission rule by a large margin; no posthoc factor or cap
change may convert this registration into a pass.

## Concrete next options: human resource decision required

1. Recommended for retaining this exact local runtime: authorize a NEW resource-
   only amendment with up to144000 CPU seconds (40 CPU hours), still one worker.
   Keep all scientific tasks/horizons/methods/unused science seeds/controller
   settings/performance gates unchanged; preserve this RESOURCE_NO_GO record;
   reuse this cost evidence rather than rerun it. Freeze/push the operational
   amendment and separate scientific admission BEFORE any science. Forty hours
   is planning headroom above30.7h, not a runtime or positive-result guarantee.
2. If elapsed time matters more, separately authorize a CPU-server execution
   package with independent cells in parallel. This needs explicit remote scope,
   native runtime/cost/accounting qualification and a new operational lock; no
   server/GPU action is authorized or performed now. Four workers do not reduce
   total CPU cost, and queue/setup time means30.7/4h is only an idealized planning
   division, not an ETA. Do not silently bypass the current one-worker restriction.
3. If submission speed is the priority, keep the confirmed TSP V3 package as the
   submission track and leave this new actor-correction experiment separate.
   No unconfirmed asynchronous-actor claim is added to V3. This cost decision
   neither invalidates V3's existing confirmations nor improves the ICML case.

No new idea/threshold change is justified by a cost-only stop. Conversely, it
would be unjustified to promise that more CPU will make the method outperform.
Once resources are explicitly decided, the next deliverable should be the
complete locked learning benchmark, not another estimator-only test sequence.

## Statistical/fallacy checks and terminal handling

11/11 ARS checks covered. Simpson/ecological:task costs are separated, no pooled
return inference. Berkson/collider:no outcome filtering or conditioning.
Base-rate:diagnostic accuracy is not evaluated. Regression-to-mean:no selected
extreme return is reused. Survivorship:all39 cells remain, zero dropped.
Look-elsewhere:no return ranking or significance search. Forking paths:source,
forecast factor, caps and stopping rule were frozen before data. Causation and
reverse causality:no return causal claim follows from CPU observations.
No p-values, confidence intervals or population-superiority inference apply.

The periodic follow-up is deleted at this bounded terminal resource decision.
Await explicit resource direction; do not start science or another profiling
attempt automatically. This report, both deterministic verification outputs,
all39 cost payloads and original terminal records are committed/pushed together.
