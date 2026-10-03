# TSP-MARL-SMACV2-HEADROOM-001 Amendment 1

## Trigger and preserved evidence

Array job `1851296` completed seven of its eight frozen cells with exit code zero.
Cell 7 (`q=8`, shared coupling, seed `81102`) stopped after update 1200 of 2666 because the common inverse-CDF sampler returned action index 16 for a 16-action categorical distribution.
The incomplete output directory is retained unchanged and is excluded from scientific analysis because it contains no completed bridge metadata.
The seven completed cells, the frozen design, seeds, budgets, evaluation schedule, baselines, and gates remain unchanged.

## Deterministic implementation correction

HARL supplies normalized float32 categorical probabilities.
Finite-precision cumulative summation can leave the final cumulative probability a few units in the last place below one, so a valid uniform draw can fall into a spurious interval above the final category.
The corrected sampler verifies that every probability row is finite and normalized, preserves every interior inverse-CDF boundary, and sets only the final cumulative boundary to exactly one.
This closes the numerical interval without changing the intended categorical law or the public-uniform coupling.

## Recovery rule

Only the failed cell may be rerun.
The recovery uses the identical task, policy architecture, optimizer, `q=8`, shared coupling, seed `81102`, seed registry, message and environment budgets, evaluation schedule, upstream dependencies, and frozen gates.
It writes to the new directory `fixed-q8-shared-seed81102-recovery1` and does not overwrite the incomplete output.
The recovery is admissible only if it exits with code zero, passes file checksums and exact resource accounting, and leaves exactly one completed metadata record for the failed scientific cell.
