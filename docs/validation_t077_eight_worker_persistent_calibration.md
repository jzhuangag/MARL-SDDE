# T-077 eight-worker execution validation

T-077 failed compute gate F3. Eight workers stayed responsive and balanced,
but the frozen 432-cell grid did not complete within ten minutes. At timeout,
worker CPU times ranged from 428.52 to 453.95 seconds and memory remained near
35 MB per worker. Ten exact task processes were terminated; no result directory
or partial outcome existed.

This rules out further local worker-count escalation for the frozen full grid.
The next admissible experiment must use a separately frozen, theory-selected
workload rather than an extended timeout. Selecting delay extremes 0 and 3
while retaining every other factorial level gives 288 cells and 9,216 old-seed
endpoints. The selection is defined by the delay-stability question, not by
observed performance.
