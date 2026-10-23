# Complete-loop qualification: PASS

Material Passport: operational qualification with complete deterministic replay;
not return-performance evidence. Verified before any scientific development.

Twenty unique cells complete: both native tasks and all ten methods, six rounds
each. The twenty-cell replay reproduces every semantic payload exactly, excluding
only measured CPU duration. Two independent analyses give byte-identical files:

- decision.json SHA256: 81f46193b08f6578f6126246de8dde9d3147e237dbbe4b1a97cc4a6a15249f52.
- integrity.json SHA256: 81b98a2e1311fab471d0dd6d2c19261244d25685e82111c2af8d7b9ad44e33b6.

The frozen analyzer reports QUALIFIED, replay_cells=20,
all_accounting_causal_payload_checks=true, performance_evidence=false.
Both full-controller task cells exhibit actual nonzero version staleness.
All schedulers share the independent launch-ordinal service stream; no aliasing
of the selection clock is accepted. Native sampling, source gradients, critic
steps, proposals, score forwards, potential work, trajectory/advantage/policy/
gradient payloads and message counts reconcile. Source parameter snapshot
assertions and single-use packet/terminal incoming-byte checks all pass.

Each root uses48000 training and3000 evaluation joint transitions. Neither
qualification return nor evaluation return was used to choose constants,
tasks, methods, catalogue, learning rates or gates. Active CPU duration:
383.703125 seconds original and385.375 seconds replay. Four current interface
tests pass, including the projection inequality and server-owned Adam moments.

Live locked runtime: HARL b1af98b0dbab72a2eee9d160751cd09aedbb8ce2, clean;
MPE source tree cc403c5864789f6c520fa930e3119ce2435847bf4416f62cb5069215b560732f;
NumPy2.2.2 and PyTorch2.6.0+cpu. The runtime versions are taken from the live lock,
not an earlier environment description.

Original/replay roots, hashes, complete analyses and the original source lock
are retained. This pass permits only separately preregistered complete-learning
development. It does not overturn any A1/estimator failure or establish ICML
novelty, HAPPO dominance, stationarity or multi-family generalization.
