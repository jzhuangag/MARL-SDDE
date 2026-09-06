# PDSG-SIGN-001 preregistration

Date: 2026-09-06

Status: frozen before any PDSG-SIGN-001 trajectory or result directory is
created.
This is a local CPU exact-model confirmation, not a standard MARL experiment.

## Research question

In an asynchronous factored Markov potential game with random packet delays
and a binding communication budget, does a fixed-horizon signed
policy-version graph chosen by paired Lyapunov drift retain material terminal
learning value against a resource-feasible envelope of strong static and
online freshness schedulers?

## Frozen mechanism

Each of six distinct owner policies launches a horizon-four packet from its
current owner block and cached teammate versions.
The active donor follows a lazy directed Markov chain whose regime changes
online.
The server selects the null action or one edge in the launch-time causal cone
by minimizing

\[
V\widehat D_p(a)+Q_pc_p(a),
\qquad V=\sqrt{160},
\]

where this confirmation uses the exact quadratic signed drift `D_p`.
The returned packet is applied at the fixed certified cap `0.04` after an
independent random delay.
Every launch consumes four environment transitions; no unused control split
is charged or generated.

The true drift makes this an oracle existence test.
It cannot establish that a neural centralized critic estimates the score.

## Frozen population

- 24 cells: coupling `{0,0.6,0.9}`, maximum extra delay `{2,5}`, message
  budget `{0.5,1.0}`, environment budget `4`, and switch probability
  `{0.03,0.12}`.
- Sixteen active cells and eight uncoupled controls.
- Sixteen new common-random-number seeds `51001--51016`.
- 160 launches for every policy/cell/seed trajectory.
- 25 policies and 9,600 endpoint rows.

Development seeds `43001--43004` are excluded.
Each seed applies one common random permutation to the initial policy and
target coordinates, shared by every compared policy.

## Frozen strong envelope

The 24 baselines are no refresh; unsigned packet-debt; largest mismatch;
active donor; largest coefficient; top-two mismatch and coefficient;
round-robin; periodic full refresh; all five fixed one-offset graphs; and all
ten fixed two-offset graphs.
Fixed schedulers use token buckets at the same declared message budget.

The AUC envelope and terminal envelope are selected separately by the smallest
geometric risk among resource-feasible baselines in each cell.
This outcome-wise cell envelope is deliberately favorable to the baselines.
The signed policy is never a baseline candidate.

## Frozen gates

The machine-readable definitions in
`docs/pdsg_sign001_manifest_20260906.json` are authoritative.
S1--S11 require complete finite outputs, exact environment charging, message-
budget feasibility, at least 10% active median terminal improvement, broad
terminal and AUC directionality, positive groupwise terminal effects, exact
uncoupled no-harm, nontrivial graph variation, complete baselines, and frozen
provenance.
S12 requires byte-identical endpoints, cells, and core summary in an isolated
rerun.

Any S1--S11 failure is retained without changing seeds, cells, policies,
thresholds, or metrics.
The isolated reproduction still verifies either outcome.
A failure forbids integration of the estimator under this identifier.

## Frozen provenance

- manifest SHA-256:
  `1AC08E5407DCF653670E6C316CD46B8A6AF1D7868F661A4A20D97C5B157D0B6B`;
- runner SHA-256:
  `92AFBE9B2DF1155D3DA7C58453274CB4B5EDCE82387A3E3557FC3F48DBE6A7E8`;
- asynchronous game SHA-256:
  `AF0A1765405B2A8C1B7E21DB0166A5B75EA231ABE48ADE5A22765A3ACAF7DC36`;
- signed theorem component SHA-256:
  `D73C341676C05C0165782334F5F51C721B0CE1F0A5AA18CDBC34A7EE411F9A44`.

Before this preregistration, both
`experiments/policy_dependency_sync/results/pdsg_sign001_primary` and
`experiments/policy_dependency_sync/results/pdsg_sign001_reproduction` were
absent.
Static validation reported 24 cells, 25 policies, 16 seeds, and 9,600 expected
endpoints.
Four manifest tests passed.

## Commands after the preregistration commit

```text
.venv/Scripts/python.exe -m experiments.policy_dependency_sync.run_pdsg_sign001 run --manifest docs/pdsg_sign001_manifest_20260906.json --output experiments/policy_dependency_sync/results/pdsg_sign001_primary --workers 4
.venv/Scripts/python.exe -m experiments.policy_dependency_sync.run_pdsg_sign001 run --manifest docs/pdsg_sign001_manifest_20260906.json --output experiments/policy_dependency_sync/results/pdsg_sign001_reproduction --workers 4
```

No GPU, HPC4 job, formal paper evidence, or standard-benchmark learning run is
authorized by this preregistration.
