# Validation: PDSG-COMP-002 local-factor alignment cost

Date: 2026-09-06

## Decision

The materially sparse implementation passes L1--L4 and fails L5--L6.  The
registered stopping rule therefore forbids clean reproduction and any
successor learning experiment justified by this cost audit.

The primary run verifies the intended autograd support: all local donor
derivatives are nonzero and every probed non-neighbor parameter is absent from
the alignment graph.  It also observes a median HVP-to-local-gradient time
ratio of 1.7441 and a maximum of 2.0277.  These facts are descriptive; the
complete preregistered low-complexity timing certificate failed.

## Provenance

- estimator-scope commit: `d717b434c3aa152d7b020b0a0a443feae67cff05`;
- preregistration commit: `900709a9a2556e2b513da8f818191617eba7b848`;
- runner freeze commit: `a931d477599cfe24c76a280f51fc620e58b01061`;
- manifest SHA-256:
  `E7CD1D83650E5F9EE41CAEE9E4C54369B95D20C218C39368F2074DB52A940309`;
- runner SHA-256:
  `CD05124EA6B330A1B327DD1E8335F17A67D30B78E97307437F31279329C8BE63`;
- primary result SHA-256:
  `CACB73C33E1D3F2F3F6F83AD209AF25572EBBD0462E76120A3466130258A08A3`;
- 108 configuration rows;
- Python 3.11.13 and PyTorch 2.6.0 CPU with four threads;
- no environment, trajectory, learning outcome, GPU, or HPC4 operation.

The ignored raw output is
`tmp/policy_dependency_sync/sparse_hvp_cost_primary/summary.json`.

## Gate ledger

| Gate | Result | Observation |
|---|---:|---|
| L1 finite positive values | pass | all 108 rows |
| L2 local nonzero, outside unused | pass | all 108 rows |
| L3 median HVP/baseline <= 2.5 | pass | `1.744127` |
| L4 maximum HVP/baseline <= 4.0 | pass | `2.027689` |
| L5 fixed-degree global scaling <= 1.35 | **fail** | `1.712166` |
| L6 adjacent-degree scaling <= 2.2 | **fail** | `2.246813` |
| L7 reproduction | not run | blocked by L5--L6 |

The L5 maximum occurs at hidden width 128, batch 256, and degree four.  The
across-seed median HVP time is 10.223 ms at 64 agents and 5.9708 ms at 256
agents, even though neither the forward graph nor differentiated parameter
set contains non-neighbor blocks.  The L6 maximum occurs at 64 agents, width
128, batch 256: 4.55 ms at degree two and 10.223 ms at degree four.

The timing inversion across global agent counts shows that short Windows CPU
wall-clock measurements contain system/cache variability not represented by a
pure operation-count claim.  This is a diagnosis, not grounds to change the
frozen gates or run until favorable timings appear.

## Research consequence

The project stops opening further micro-timing identifiers.  The structural
complexity claim is limited to the mechanically verified autograd support:
the statistic touches `Delta` donor blocks and no non-neighbor blocks.  The
observed 1.7441 median and 2.0277 maximum ratios may guide engineering, but are
not promoted to a passed certificate.

The next admissible work is mathematical and statistical:

1. complete the expectation-level finite-time theorem with the sparse-tail
   term;
2. construct and audit the signed alignment estimator's error, not its timing;
3. use an outcome-free benchmark-tail audit to decide whether any standard
   MARL tasks support the sparse theorem scope.

No learning runner is authorized yet.
