# Multi-task participation-phase operational amendment 2

## Trigger

The first Stage-A submission attempt was rejected before job creation with
`QOSMaxSubmitJobPerUserLimit` because the frozen 32-cell lattice was expressed
as one 32-element Slurm array.
No Stage-A job ID, trajectory, return, or result directory was created.

## Permitted correction

The global lattice remains indexed from 0 through 31 exactly as frozen.
The Slurm payload now accepts a mandatory `OFFSET` and maps each eight-element
array to global indices `OFFSET + SLURM_ARRAY_TASK_ID`.
Four disjoint submissions with offsets `0`, `8`, `16`, and `24` therefore
execute each frozen cell exactly once.
Each submission limits concurrency to two cells.

Task definitions, resource rays, couplings, participation endpoints, seeds,
policy hyperparameters, budgets, output names, analyzer, and mandatory gates
are unchanged.
Later offsets are submitted only after earlier arrays leave the queue so the
user's QOS limit is respected.

Amended Slurm SHA-256:
`5f3e9687c114d57135cdfccd02cb335853df487f153e9e8fe42ac57c350a47bb`.
