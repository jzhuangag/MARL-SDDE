# T-077 eight-worker execution amendment

T-077 uses all eight detected logical processors to execute the unchanged
T-074 endpoint function. It preserves ordered output, chunk size sixteen, and
the rule that no output directory is created until every endpoint completes.
The hard wall-clock gate is ten minutes. Scientific parameters, P1--P11,
source rows, seeds, and analysis are hash-locked to T-074.

This is the final local orchestration attempt for this frozen grid. Failure
does not authorize a larger timeout, GPU, or HPC4; it would require batching or
a smaller separately preregistered workload. Passing still produces tainted
architecture evidence only.
