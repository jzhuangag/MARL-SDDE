# Q-grid infrastructure amendment 1

The first MPE submission, Slurm array `1861758`, could not open its requested
stdout/stderr paths because the new scratch root's `logs` directory did not
exist before Slurm launched the batch script.  Tasks 0--5 failed before the
payload entered Python; tasks 6--7 were cancelled once the common cause was
identified.  The root contained zero `tsp_bridge_metadata.json` files and no
scientific trajectory.

The recovery creates only the pre-existing `logs`, `artifacts`, and `tmp`
directories.  It changes no source file, hash, task, method, seed, budget,
model, evaluation rule, or gate.  The identical frozen script was resubmitted
for offset zero as Slurm array `1861769`.  Array `1861758` remains in the HPC4
accounting record and is excluded because it contains no completed metadata,
not because of an observed scientific outcome.

Array `1861769` then exposed a second pre-payload environment error: the shared
Two-Clocks Python runtime does not contain the pinned PettingZoo/SuperSuit
stack used by TSP-MARL-CONF-001.  All eight tasks exited during environment
construction with `ModuleNotFoundError: supersuit`; they produced zero
completed metadata records and no usable scientific trajectory.  The MPE
script is therefore amended to use the predecessor's preserved
`/scratch/jzhuangag/MARL-SDDE-TSP-MARL-CONF-001/venv` and matching HARL
checkout, with user-site packages disabled.  Recovery output is isolated by a
required `ARTIFACT_SET` label.  This amendment changes only the execution
environment and output namespace; the frozen task, q-grid, seeds, budgets,
model, evaluation, and gates remain unchanged.

Amended MPE Slurm payload SHA-256:
`30b02c45f2f1efd2719be88f7c17b2db7b5d08c798d227c765681f0f3f1f0e8d`.
The original pre-outcome hash remains recorded in the execution freeze rather
than being overwritten.
