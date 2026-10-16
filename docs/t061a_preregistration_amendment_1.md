# T-061A preregistration Amendment 1: comparator split label

The first run attempt stopped before any endpoint, CSV, summary, result
directory, or console statistic was produced.  The T-060 fixed-q helper used
for the strong comparator requested a `selection_seeds` registry solely to
write its historical split label, while T-061A has no selection split.  The
controller risk calculated immediately before the exception remained only in
process memory and was neither printed nor read.

The prospective fix passes an empty split registry to that comparator helper.
It changes no trajectory, q, risk, budget, probe, comparator, seed, gate, or
analysis value; the returned split label is not included in a T-061A endpoint.
A mini end-to-end regression now calls `run_endpoint` and verifies both the
controller and fixed comparator paths before scientific execution.

All scientific choices and the original stop rule remain frozen.  This
amendment does not authorize formal seeds, GPU, HPC4, or `/project` output.
