# Multi-task participation-phase operational amendment 1

## Trigger

Outcome-free G0 job `1889967` failed with exit code `1:0` before producing an
output artifact.
The MaMuJoCo runtime lacks the `supersuit` dependency required by the registered
MPE environment, so all four MPE worker subprocesses exited during import.
No trajectory, return, win rate, gradient, or policy update was produced.

## Permitted correction

G0 is split into a two-cell Slurm array.
The Speaker--Listener cell uses the already pinned and previously validated MPE
runtime under `/scratch/jzhuangag/MARL-SDDE-TSP-MARL-CONF-001`.
The MaMuJoCo cell uses the already pinned runtime under
`/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902`.
Its MuJoCo path, key path, and library search path are exported exactly as in
the previously validated Two-Clocks MaMuJoCo Slurm payload.

An intermediate read-only import preflight stopped before submission because
these environment variables were not yet exported in the login shell.
It created no Slurm job or scientific output and motivated adding the frozen
runtime paths to the task-specific payload.

The task definitions, HARL commit, coupling rules, seeds, budgets, Stage-A
training lattice, gates, and stopping rule remain unchanged.
Each G0 cell writes a task-specific compatibility artifact, and both must pass
before Stage A can be submitted.

The failed job and its logs are retained under the registered scratch root.
