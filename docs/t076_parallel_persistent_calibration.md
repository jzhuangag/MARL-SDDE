# T-076 parallel execution amendment

T-076 changes only grid orchestration. It imports the frozen T-074 controller,
source rows, analysis, constants, and P1--P11 without modification. Four local
CPU processes evaluate independent endpoints through an ordered map with chunk
size eight. Results are written only after every endpoint returns.

The machine exposes eight logical processors and approximately 17 GB physical
memory. Four workers leave headroom and are expected to use far below 1 GB in
aggregate. E1--E5 freeze full scientific-gate preservation, complete ordered
coverage, a 12-minute wall-clock limit, byte-exact four-worker reproduction,
and full repository regression. Passing still yields outcome-informed design
evidence only; it does not authorize new seeds, formal evidence, nonlinear
benchmarks, GPU, or HPC4.
