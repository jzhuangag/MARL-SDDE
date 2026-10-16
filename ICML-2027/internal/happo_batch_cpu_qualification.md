# HAPPO asynchronous batch CPU qualification

Date: 2026-10-15.

The first 12-block operational attempt used two worker processes and stopped
when PettingZoo/Pygame could not allocate a rendering surface under the host's
low free-memory state. It did not use a registered scientific seed and did not
complete the qualification payload set. Its partial directory is preserved as
`happo_async_batch_qualification_final_20261014`.

The unchanged protocol was rerun with one worker in the fresh directory
`happo_async_batch_qualification_serial_20261015`. All 14 task--method
payloads completed. The machine-readable integrity audit records:

- complete task--method mapping;
- identical initial parameters within each task;
- finite learning-curve values;
- exact joint and per-agent transition accounting;
- exact actor, critic, packet, proxy, oracle-search, counterfactual, and
  independent diagnostic charges;
- no benchmark-evaluation leakage into any controller;
- unique per-block training-trajectory identity hashes; and
- matching per-payload SHA-256 manifests and current runner/config hashes.

The qualification result is operational only. No qualification return was
used to select a method, threshold, task, seed, or hyperparameter. The
scientific source and decision gates are frozen before any development or
holdout execution.
