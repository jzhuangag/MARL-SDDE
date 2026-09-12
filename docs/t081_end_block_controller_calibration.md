# T-081 causal block-end primal-dual controller calibration

T-081 repairs the action-timing mismatch exposed after T-080.  Every one of a
block's ten observations first updates the local learner.  The completed short
trajectory then supplies a target fingerprint and a persistent Markov-noise
certificate.  A four-row simplex QP selects the collaboration matrix at block
end, and that mixed parameter affects only future learning.  Consequently no
future sample enters the action.

The virtual safety queue is updated from the observable paired excess loss of
the mixed learner and its same-data local shadow.  Its value enters the next
QP as a penalty toward the local simplex vertex.  This is a delayed
drift-plus-penalty mechanism rather than a post-outcome rollback.

The primary class is fixed before T-081 execution: nonstationary schedules,
target scale 0.3 or 0.6, and temporal correlation zero, with every registered
initialization, noise level, spatial correlation, and delay retained.  This is
the separated signal/mixing class suggested by the existing identifiability
theory, not a claim of uniform benefit.  All remaining registered cells are
retained as stationary, low-signal, or high-correlation boundary controls.

This calibration reuses all 32 old T-071A seeds and is explicitly tainted
architecture evidence.  Its strong comparator uses the frozen per-cell T-079
continuous static matrix under exactly the same block-end timing.  The
controller constants remain the previously declared scale-free rules; no
T-080 cell outcome tunes them.  Only if all C1--C14 gates pass may a separate
commit preregister a new-seed CPU pilot.  T-081 itself authorizes neither new
seeds nor formal, nonlinear, GPU, or HPC4 execution.
