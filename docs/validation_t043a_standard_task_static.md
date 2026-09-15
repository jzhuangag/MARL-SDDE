# T-043A validation: standard-task phase feasibility

## Decision

T-043A is an honest static failure. It permanently stops EXP-020A under the
registered FrozenLake/CliffWalking terminal-risk design. No sampled learning
trajectory, seed, confidence interval, pilot, formal run, GPU, or HPC4 job was
used.

The independent/environment and fully correlated/environment limits behave
as the theorem predicts, but message charging does not create the registered
reversal. After optimizing the frozen normalized step grid, `q=16` is the
oracle in all 72 message-binding cells, including every `rho=1` cell. The
strongest fixed q is therefore already the cellwise oracle and the adaptation
ceiling is exactly zero.

## Frozen gates

| Gate | Result | Verdict |
|---|---:|---|
| S1 task/kernel/fixed-point validity | exact checks pass | PASS |
| S2 finite, stable, charged rows | 1,296/1,296 | PASS |
| S3 independent/environment speedup | 1.000 | PASS |
| S4 correlated/environment no-value | maximum 0.000 | PASS |
| S5 correlated/message reversal | 0.000 | **FAIL** |
| S6 per-task q support | `{1,16}` for both | PASS |
| S7 message oracle improvement | 0.000% | **FAIL** |
| S8 strict message-cell improvement | 0.000% | **FAIL** |
| S9 outcome-free execution | zero trajectories | PASS |
| S10 reproduction/provenance | byte-identical | PASS |

The overall result is 7/10 gates, with all three adaptation-value gates
failing. The thresholds and task grid are unchanged.

## Mechanistic diagnosis

At `rho=1` all q values have the same aggregate noise per update. Message
charging gives `q=16` fewer updates than `q=1`. In these constant-step
finite-terminal-risk cells, fewer updates can reduce accumulated steady-state
noise enough to dominate slower bias contraction. The optimized `q=16/q=1`
risk ratios range from approximately 0.204 to 0.531 across the registered
`rho=1` message cells. Thus update-count reversal is not monotone when the
objective includes the full bias--variance transient; the noise-dominated
cost proxy cannot decide its direction alone.

This is a scientific limit, not a power issue: T-043A is deterministic and
uses the exact registered scalar surrogate. It also explains why a controller
cannot add value here: a single fixed `q=16` realizes every message-cell
oracle.

## Scope of the negative result

T-043A did not run the Stage-B metric frozen earlier in T-034, namely
resource-normalized MSVE area under the curve. It registered terminal scalar
risk as a necessary feasibility screen. Therefore the result forbids the
specific EXP-020A terminal-risk design but is not sampled evidence against all
standard-task AUC formulations. Any future AUC study must remain a new
preregistration justified by the pre-existing T-034 specification; T-043A
cells or thresholds cannot be retuned or relabeled.

## Reproduction

- preregistration commit: `490e83b`;
- primary runtime: 9.070 seconds;
- clean reproduction runtime: 9.002 seconds;
- `rows.csv` SHA-256:
  `696c3741c9ce67968f11d612b77628ad62c2fe5638a24c70ce08a0d443d8f747`;
- `task_constants.json` SHA-256:
  `034103f50a17c2638d61ebd120bbd59cba13e9a635d5d4f71fa4900ed20afae8`;
- `summary.json` SHA-256:
  `f8af70b70fd8f74f799dc863c2f4cc237fc6e08ff77fa647e500ebd942708afb`.

All three artifacts are byte-identical in the isolated reproduction. The
reproduction directory remains local and is excluded from Git.
