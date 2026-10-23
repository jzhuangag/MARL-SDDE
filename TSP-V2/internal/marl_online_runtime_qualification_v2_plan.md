# TSP-V2 second HARL runtime qualification plan

## Scope and provenance

`TSP-V2-MARL-ONLINE-QUAL-002` is a new outcome-free software qualification.
It does not reuse the stopped `QUAL-001` root, seeds, or artifacts and cannot be
used as development, confirmation, tuning, or manuscript evidence.

The frozen implementation commit is
`ef08d1f56e62f3094dded54299a5acfa361344db`.
It preserves the stopped runners and adds a second bridge that follows pinned
HARL's exact `available_actions[0] is not None` convention while explicitly
testing both the MPE mask and MaMuJoCo all-`None` representations.

## Grid and gates

The qualification again crosses both registered environments, both coupling
modes, and controller/fixed-q execution branches, for eight fresh jobs.
Each job must construct a real pinned runner, complete initial evaluation,
complete at least one training update, complete post-update evaluation, and
write valid metadata, progress, and checksum artifacts.

The controller branch must additionally execute its disjoint before/after
validation, Lyapunov decision, state synchronization, and exact dual-resource
charging.
All values must be finite, all eight jobs must exit with code zero, and the
pinned HARL commit must remain unchanged.

Jobs write only below
`/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-QUAL-002`.
Any failure stops this identifier without retry or gate modification.
Only a complete pass authorizes a separately preregistered scientific
development experiment with new seeds.
