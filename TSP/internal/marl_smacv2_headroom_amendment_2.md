# TSP-MARL-SMACV2-HEADROOM-001 Amendment 2

## Trigger and preserved evidence

Recovery job `1854988` launched the frozen `q=8`, shared-coupling, seed-`81102` cell on node `gpu02`.
Before the first scheduled evaluation completed, two StarCraft II evaluation processes lost their WebSocket connections and their Python workers became defunct.
The remaining workers and training parent stayed alive while waiting for the failed vector environment: after 1 hour 17 minutes, `progress.txt` was still empty and no completed bridge metadata existed.
The job was therefore cancelled as an infrastructure deadlock rather than allowed to occupy its full 16-hour allocation.
Its log and incomplete `fixed-q8-shared-seed81102-recovery1` directory are retained unchanged and excluded from scientific analysis.

## Second recovery rule

Only the same failed scientific cell may be launched again.
Recovery 2 preserves the task, policy architecture, optimizer, `q=8`, shared coupling, seed `81102`, seed registry, rollout length, message and environment budgets, evaluation schedule, dependency versions, code, and frozen gates from Amendment 1.
The Slurm allocation excludes `gpu02`, the node on which the evaluation-process deadlock occurred; node placement is not a scientific factor in the registered design.
The job writes to `fixed-q8-shared-seed81102-recovery2`, leaving every prior directory untouched.
It is admissible only after exit code zero, checksum verification, exact budget accounting, and confirmation that the analyzer sees one and only one completed metadata record for the cell.
