# TSP-V2 online HARL runtime qualification plan

## Scope

`TSP-V2-MARL-ONLINE-QUAL-001` is an outcome-free software qualification.
It is not a development, confirmation, or manuscript experiment, and its
returns must not be used to tune the controller or support a scientific claim.

The stopped executable remains unchanged.
The corrected entry point replaces only the deterministic evaluation bridge,
using the pinned HARL interface `runner.args["env"]` instead of the nonexistent
`runner.env_name` attribute.
The frozen implementation commit is
`291b402f854568553d747946d584cff315759c34`.

## Frozen qualification grid

The grid crosses both registered environments, both registered coupling modes,
and the controller/fixed-q execution branches, for eight jobs in total.
Each job constructs the real pinned HARL runner, executes initial evaluation,
at least one training update, and post-update evaluation.
Controller jobs additionally execute the disjoint before/after validation path,
candidate-state synchronization, a Lyapunov decision, and exact cost charging.

All qualification seeds are isolated from the stopped development seeds.
The jobs write only below
`/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-QUAL-001` and never write to
`/project`.

## Mandatory pass conditions

Qualification passes only if all eight jobs exit with code zero; every expected
artifact and checksum exists; all reported values are finite; the pinned HARL
commit is unchanged; each job reports at least one completed training update;
the controller jobs expose a finite validation observation; and message and
environment charges equal the formulas in the executable.

Any failure stops qualification without retry under this identifier.
A scientific development preregistration may be created only after a complete
qualification pass.
