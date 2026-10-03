# T-076 parallel execution validation

T-076 failed compute gate E3. Four local workers remained responsive and
balanced, but the run did not finish within the frozen 12-minute wall-clock
limit. It was terminated shortly after the limit was observed, at about 13.4
minutes. No result directory had been created, so E2, E4, and E5 remain false
and there is no scientific grid outcome to interpret.

Worker memory stayed near 35 MB each and the parent near 87 MB. The failure is
therefore throughput, not memory, process-pool stability, or partial I/O.
T-076 cannot be retried with an extended timeout. A new execution-only audit
may use all eight detected logical processors while preserving the exact
T-074 scientific configuration, ordered map, and no-partial-output rule.
