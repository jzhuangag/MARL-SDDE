# Validation: PDSG-COMP-001 cross-policy alignment cost

Date: 2026-09-06

## Decision

The frozen primary CPU timing passes C1--C4 and fails C5.  Under the registered
stopping rule, the overall result is a failure: no clean timing reproduction is
run and the current interface is not certified as having acceptable
degree-scaling overhead.

The result rejects the stronger claim that increasing the donor degree from
two to four changes the joint reverse-derivative time by at most 35% in every
registered model cell.  It does not show that a separate backward pass was
performed per edge: the observed worst factor is 1.5759 rather than two, and
the code performs one joint derivative call.  This descriptive observation
does not alter the failed gate.

## Frozen provenance

- preregistration commit: `aaa0288bc2338b6c483dfc96c74ab2ff83b17c4a`;
- runner freeze commit: `e622189ac80ebdf28607533164fa23e3301704df`;
- manifest SHA-256:
  `C33FDC006AD53E541AB9EA0332EDBCDD286C4326A8D0FA1821CC6521D4654292`;
- runner SHA-256:
  `E519C40BE029E76450028D030502C3B8E676A2C483D4A94511AA390A73C08D26`;
- primary result SHA-256:
  `C8B11F85337486797F26077FB189B3735128129CFDBB67FDE23DC18F6BB7FFA7`;
- 72 configurations, three registered seeds per architecture cell;
- Python 3.11.13, PyTorch 2.6.0 CPU, four PyTorch threads;
- Windows 10 build 19045;
- no environment, trajectory, learning outcome, GPU, or HPC4 operation.

The ignored raw result is
`tmp/policy_dependency_sync/hvp_cost_primary/summary.json`.

## Gate ledger

| Gate | Result | Observation |
|---|---:|---|
| C1 finite positive timing and norms | pass | all 72 rows finite and positive |
| C2 nonzero donor derivative | pass | every row nonzero |
| C3 median HVP/baseline ratio <= 4 | pass | `1.501992` |
| C4 maximum HVP/baseline ratio <= 6 | pass | `1.822653` |
| C5 maximum degree scaling <= 1.35 | **fail** | `1.575931` |
| C6 same-machine reproduction | not run | blocked by C5 |

The C5 maximum occurs at eight agents, hidden width 128, and batch size 256:
the across-seed median HVP times are about 3.18 ms at degree two and 5.01 ms
at degree four.  Eleven of the twelve registered architecture cells have a
degree-scaling factor between 1.0734 and 1.3058; this is descriptive and cannot
replace the maximum-cell gate.

## Consequence for the algorithm

The expected-drift controller remains a theoretical candidate, but its neural
implementation must not be advertised as degree-insensitive or as having
passed the registered low-complexity certificate.  The defensible complexity
is linear in the number of differentiated donor parameter blocks, with an
observed aggregate HVP/baseline ratio reported only as design-stage evidence.

A successor may proceed only after changing the estimator itself, for example
by imposing a prospectively declared sparse dependency mask or by deriving a
sketched scalar edge statistic.  Merely relaxing C5 or rerunning until the
timing falls below 1.35 is forbidden.
