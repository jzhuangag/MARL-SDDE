# Additional100 CPU-hour resource continuation

## Material Passport / prospective authorization

On2026-10-18 the human explicitly authorized100 additional CPU hours, preservation
of completed results, no whole-study restart, completion of all controls/replays/
audits, truthful reporting, and independent V3 TSP repair. Total cumulative CPU
cap is504000 seconds (140h), including old profile, every old science/guard cost,
the interrupted cell's lost work, and continuation/import/replay/analysis costs.

The original40h driver has no hot-extension or mid-cell state checkpoint interface.
Do not inject code or mutate its plan/lock. Let its frozen resource check stop it;
preserve its stop, partial costs, logs and all completed payloads unchanged. The
NEW wrapper is outside the frozen scientific package. It may start only after
the same kernel mutex is free and the prior failure is exactly the registered
whole-CPU-cap TimeoutError, never a scientific or other payload failure.

## Exact continuation boundary

The completed prefix must agree exactly with the last progress generation,
registered mapping and per-file SHA256 manifest. No missing/extra complete cell
is eligible. Copy that prefix byte-for-byte into a fresh continuation root and
verify both original and imported SHA256 values; never overwrite the old root.
Only remaining cells run. The resource-interrupted cell has no recoverable full
policy/optimizer/RNG state; regenerate that ONE unfinished cell from its original
registered seed/initialization, paying both the abandoned and regenerated work.
This exception is prospectively disclosed, resource-only, and independent of
all scientific outcomes. No completed cell is retried or selected by its return.

All117 scientific mappings, all9 controls/full/3 ablations,3 tasks/two families,
three seeds,200k/200k/1m native budgets,25/25/1000 horizons, constants, accounting,
the six fixed full-budget replays and every original gate remain frozen. Import
validation checks integrity/finiteness without ranking partial-study returns.
The original frozen analyzer runs twice in fresh disjoint directories and its
metrics.csv/summary.json/integrity.json must be byte-identical. Frozen all-method
native/message/logical-byte plots are required. No performance claims before the
complete workflow and independent audit. No confirmation or manuscript admission.

## Budget forecast and stop policy

Cost-only extrapolation at11:27 CST used the first complete HalfCheetah cell:
8154.984375 CPU seconds versus1116.455078125 profile-predicted seconds, factor7.304.
Applying this factor to the unchanged per-method remaining HalfCheetah costs,
using actual MPE costs for its four replay cells, and a600-second analysis reserve
gave about94.5 additional CPU hours and134.3 cumulative hours. This is a rough
forecast from one long-task cell, NOT a confidence bound or completion guarantee.
The additional100h leaves limited headroom and includes interrupted-work loss.
No automatic extra cap, task removal, budget shrinkage or silent retry is allowed.
Any new failure stops and preserves evidence. One CPU native owner; no GPU/HPC4.

Freeze/push new wrapper, amendment, source/runtime/config/prefix lock and admission
BEFORE continuation native data. Keep V3 revisions in a separate preserved copy;
their correctness/citation/PDF release does not depend on this experiment.
