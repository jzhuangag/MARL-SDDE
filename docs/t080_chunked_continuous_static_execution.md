# T-080 atomic execution wrapper for T-079

T-080 changes only recoverability.  It partitions the unchanged ordered T-079
workload into twelve contiguous 36-cell chunks.  Each chunk calls the frozen
T-079 cell function with eight local CPU workers and writes only after all 36
cells finish.  The final merge calls the frozen T-079 analyzer.

No partial chunk may be interpreted, and no T-079 scientific setting changes.
The wrapper exists because the original monolithic session ended after about
103 minutes without creating an output directory, losing all completed in-memory
cells.

T-080 is local-CPU only.  It does not authorize sampled seeds, formal evidence,
nonlinear benchmarks, GPU, HPC4, or remote storage writes.
