# EXP-016B independent formal replication preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-01
- Verification Status: PREREGISTERED
- Version Label: exp016b_formal_v1

## Scientific boundary

The EXP-016B pilot passed P1--P12 but was implemented after its static design
commit.  It is mechanism-development evidence and is not promoted to formal
paper evidence.  This preregistration freezes the completed implementation,
analysis, inputs, and an independent seed block before formal outcomes exist.

The formal replication changes no scenario, policy, budget point, estimand,
confidence family, practical-effect threshold, coverage denominator, safety
metric, or mandatory gate.  In particular, the primary contrast remains the
paired finite-Z risk difference between information-only probing and
learning-aware fallback.  Layer B remains an affine delayed Markov-TD transfer
test rather than a general nonlinear MARL claim.

## Frozen protocol

- Base configuration: `6dfdf87521700c2ddae9b81947e0ecc01ee33ebcf5fcda34b09e9e3c3f7f7ee5`
- Formal protocol: `98ddaf121e718ea6d390efa8795b4f06e7916a43de0a98a57806205b44988c76`
- Formal seeds: 192 consecutive seeds beginning at `20440101`
- Seed-list SHA-256: `5ad675ea1e60b9d0dfad8967e9590d74fe31afb8012ec747b2f7ad1bab3023c5`
- Expected rows: 2,752,512
- Expected disk: approximately 2.2 GB plus reproducibility output
- Compute: local CPU; no GPU, HPC4, or `/project`

Exact runner, analyzer, wrapper, input, and seed hashes are stored in
`docs/exp016b_formal_registry.json`.  The wrapper refuses execution if any
implementation hash changes.  Formal output must be written to a new directory
and never overwrite pilot or reproduction artifacts.

## Decision rule

All P1--P12 remain mandatory.  No failed gate may be rounded, amended, replaced,
or rescued by a post hoc subset.  A clean same-seed replication must reproduce
the four core artifacts byte for byte.  Passing permits use of this experiment
as formal synthetic/affine evidence; it does not by itself authorize a broad
nonlinear MARL or unrestricted unknown-mixing claim.

At the time of this freeze, no formal trajectory, metric, aggregate, or outcome
had been generated.
