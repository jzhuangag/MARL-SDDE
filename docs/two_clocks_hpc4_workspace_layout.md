# Two Clocks HPC4 workspace layout

The independent active root is
`/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902`. It does not reuse or modify
the historical `/scratch/jzhuangag/MARL-SDDE` checkout.

| Directory | Contract |
|---|---|
| `code/` | Clean GitHub checkout at an exact commit. |
| `external/HARL/` | Clean upstream HARL checkout pinned by commit. |
| `external/SMACv2/` | Clean upstream SMACv2 checkout pinned by commit. |
| `external/downloads/` | Source archives retained with SHA-256 provenance. |
| `envs/` | Runtime overlay and caches; no cache is written to home. |
| `results/` | Immutable per-job outputs; an existing directory is never overwritten. |
| `plots/` | Plot contract and, only after preregistration, generated figures. |
| `logs/` | Slurm stdout and stderr. |
| `provenance/` | Source checksums and environment records. |

The G0 job is outcome-free. It may report versions, tensor shapes, transition
counts, process teardown, timing, and memory, but not rewards, returns,
win-rates, or any method comparison. Passing G0 does not authorize formal
seeds. A scientific pilot requires a later immutable preregistration commit.
