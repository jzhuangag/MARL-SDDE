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

## Operational amendment 3: matched actor-transition horizon

The corrected memory qualification revealed that Pistonball may terminate
inside a four-cycle rollout segment.  Fixing the number of launch events alone
therefore does not fix the number of charged actor transitions: a scheduler
whose policies terminate an episode earlier can consume fewer transitions.
The development comparison is now defined by an exact four-cycle launch
horizon.  After a terminal transition the same owner worker is reset with the
next deterministic seed and collects the remaining cycles before the launch
ends.  Hence every method receives exactly 327,680 actor transitions and the
same evaluation transition grid.  This learner change requires a fresh memory
qualification and a completely new development output root; the successful
pre-amendment qualification is retained as systems evidence only.

## Operational amendment 4: deterministic GPU placebo contract

The first matched-transition matrix completed, but `signed_only` selected no
refreshes while its training reward trace differed from `no_refresh`.  The two
methods were byte-identical under the same CPU seed, so the GPU discrepancy is
not an algorithmic effect.  The runner now enables deterministic PyTorch
algorithms and disables cuDNN benchmarking, while every Pistonball Slurm entry
sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts.  The result object
records the deterministic-algorithm state and batch validation requires it.
The completed pre-amendment matrix remains useful only for this diagnosis; its
return differences cannot be used as method evidence.  A fresh qualification
and fresh matrix root are required.

The first strict-determinism qualification then stopped before producing an
output because CUDA does not implement a deterministic backward pass for
adaptive average pooling.  Both encoders receive fixed compressed geometry, so
their adaptive layers are replaced by fixed `AvgPool2d(4,4)` layers with the
same 4-by-2 actor and 4-by-4 critic output shapes.  Determinism remains strict;
the error is not downgraded to a warning.

Before spending five full development runs, a two-method placebo qualification
uses development-only seed 79002 for 512 launches.  If `signed_only` selects no
edge, its complete launch trace, training reward, and evaluation rows must be
identical to `no_refresh`.  A mismatch blocks the full matrix even when both
jobs otherwise finish.
