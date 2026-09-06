## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: development experiment plan
- Origin Date: 2026-09-07
- Verification Status: DEVELOPMENT-ONLY; NOT PREREGISTERED EVIDENCE
- Version Label: pistonball_gpu_development_v1

# Pistonball policy-freshness GPU development matrix

The systems smoke passed on A30 job 1825368.  This next matrix is allowed to
guide debugging and the later pilot design, so it is intentionally labeled
non-confirmatory.  Its seed 79001 must be excluded from all pilot and formal
populations.

Five methods share identical initialization, random streams, 327,680 charged
actor transitions, 2,048 available refresh units, randomized Pistonball
initialization, rollout horizon four, maximum receipt delay four, and nine
evaluation checkpoints with three episodes each:

- composite signed learning drift + exact cache energy + communication queue;
- signed learning drift without cache energy;
- maximum parameter mismatch;
- prefix-feasible complete-burst refresh;
- no optional refresh.

Each method runs as a separate one-A30 Slurm array element.  Outputs and logs
remain under
`/scratch/jzhuangag/causal-policy-freshness-icml2027`; no checkpoint is written
because this phase needs only curves and accounting.  No `/project` write or
cleanup is authorized.

This phase asks whether learning occurs, whether the composite method avoids
both null and full-refresh collapse, whether signed value adds anything beyond
mismatch debt, and whether runtime is practical.  It has no pass threshold,
p-value, or paper claim.  If these results justify a pilot, the next commit
must freeze new seeds, exact hashes, strong comparator selection, return/AUC/
budget/overhead gates, and a stopping rule before any pilot job starts.

## Operational amendment 1

The first submission passed a comma-delimited scheduler list through Slurm's
`--export`, whose own delimiter is comma.  Task 0 received the first method and
continued normally; tasks 1--4 stopped in shell parsing before Python and
created no result JSON.  The batch script now uses a colon-delimited list.
The scientific runner, seed, methods, configurations, and analyzer are
unchanged.  Only failed task indices may be resubmitted; task 0 must not be
rerun or overwritten.

## Operational amendment 2: bounded image preprocessing memory

The corrected array began Python normally, but signed, signed-only, and
no-refresh tasks were killed by the 32 GB host-memory cgroup before producing
JSON; mismatch and complete-burst were stopped before the same failure.  No
return or curve was observed.  A code audit identified a dominant avoidable
allocation candidate: repeated full-resolution uint8-to-float32 CPU image
interpolation before replay compression.  The preprocessing now uses deterministic center-index uint8
subsampling, whose largest intermediate is bounded by the compressed spatial
axis rather than a full float image batch.

This is a learner implementation change, so none of the killed/stopped tasks
may be combined with later results.  A new commit and new output root must first
pass a bounded 1,024-launch memory qualification.  Only then may the original
development matrix be restarted from scratch; no pilot/formal population is
affected because none exists.

The first qualification execution completed the learner in 129.06 seconds with
2,765,264 KiB peak process RSS, but its shell validator failed because it had
incorrectly required every rollout segment to contain four cycles.  Pistonball
can terminate within a segment, so the fully charged actor-transition count is
instead 20 times the sum of the realized segment lengths.  The runner's charge
was correct; the qualification assertion was not.  The replacement validator
checks this realized accounting identity and equality between launched and
received gradient-packet counts after the terminal drain.  The failed
qualification output remains diagnostic only and cannot authorize the matrix.
