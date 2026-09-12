## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-07
- Verification Status: VERIFIED
- Version Label: pdsg_obs001_preregistration_v1

# PDSG-OBS-001 preregistration

Status: frozen before any PDSG-OBS-001 trajectory or result directory is
created.  This is a local-CPU exact-model confirmation, not a standard MARL
benchmark and not formal paper evidence.

## Research question

Can a causal, translation-equivariant estimator built only from completed
ordinary training packets recover a material fraction of the exact signed
policy-version graph oracle's learning value, while retaining the same
communication/environment accounting and defeating the same strong static and
online envelope?

## Frozen mechanism

Six distinct owner policies asynchronously launch fixed horizon-four Markov
packets from mixed teammate-policy caches.  At each launch the action is the
null graph or one edge in the public causal cone.  Communication is priced by
a virtual queue.  Receipt uses the fixed stable cap `0.04`.

The online reference estimate starts at the current parameter vector, making
the rule equivariant to translations of parameter coordinates.  Each completed
ordinary packet supplies one normalized-LMS update with gain `0.25`.  No
separate sensor, probe trajectory, future random variable, or true target is
available to the online scorer.  The selected edge minimizes the estimated
signed paired-Lyapunov index in exact `O(Delta)` time.

The exact-target signed controller is executed only as a diagnostic oracle.  It
cannot enter either baseline envelope and cannot determine the pass/fail
outcome except through the separately frozen headroom-recovery gate O12.

## Frozen population

- 24 cells: coupling `{0,0.6,0.9}`, maximum extra delay `{2,5}`, message
  budget `{0.5,1.0}`, environment budget `4`, switch probability
  `{0.03,0.12}`;
- 16 active cells and eight coupling-zero controls;
- 16 new common-random-number seeds `61001--61016`;
- 160 launches per policy/cell/seed;
- 25 core policies plus one diagnostic oracle, for 9,984 endpoints.

All development seeds `43001--43012` and oracle-confirmation seeds
`51001--51016` are excluded.

## Frozen comparison

The 24 baselines are unchanged from PDSG-SIGN-001: no refresh; unsigned
packet-debt; mismatch-, coefficient-, active-donor-, round-robin-, and
periodic-full rules; all five fixed one-offset graphs; and all ten fixed
two-offset graphs.  Fixed graph policies use a token bucket.  AUC and terminal
envelopes are selected separately, within each cell, from resource-feasible
baselines.  This outcome-wise rule favors the comparator.

## Frozen gates

The machine-readable definitions in
`docs/pdsg_obs001_manifest_20260906.json` are authoritative.  O1--O13 require
complete finite output, seed isolation, exact trajectory charging, message
feasibility, at least 10% active median terminal improvement, broad terminal
and AUC directionality, positive subgroup effects, exact uncoupled controls,
dynamic graph variation, all 24 baselines, frozen implementation provenance,
at least 75% median oracle-headroom recovery, and finite estimator/action-trace
diagnostics.  O14 requires byte-identical endpoints, cells, and summary in an
isolated rerun.

Any O1--O13 failure is retained without changing seeds, cells, code,
thresholds, policies, or metrics.  O14 reproduces the same pass or failure.
No result from PDSG-SIGN-001 or development enters tuning after this commit.

## Frozen provenance

- manifest SHA-256:
  `32C7148C74CF0EA086A94FD85C3676107B32422AAF9C0819AC0487E187FA95A4`;
- runner SHA-256:
  `A8172C45ECA8536340E8B671E12C156251B9EA4C1F92A0ABC5786D79856F2E31`;
- manifest-test SHA-256:
  `7ECFADF3A7CAC49E22438612AB09B4E0E5DFC030CFE1E7BAB906E9E035B4BCC3`;
- asynchronous-game SHA-256:
  `0D9623C90CF16CC8BE50CA5E691C480748C736820440980893325F7362143DAF`;
- signed-theorem component SHA-256:
  `D73C341676C05C0165782334F5F51C721B0CE1F0A5AA18CDBC34A7EE411F9A44`.

Immediately before this preregistration, both result directories were absent.
Static validation reported 24 cells, 25 core policies, one oracle diagnostic,
16 seeds, and 9,984 expected endpoints.  Seventeen targeted static tests
passed.  No scientific trajectory was generated.

## Authorized commands after this commit

```text
.venv/Scripts/python.exe -m experiments.policy_dependency_sync.run_pdsg_obs001 run --manifest docs/pdsg_obs001_manifest_20260906.json --output experiments/policy_dependency_sync/results/pdsg_obs001_primary --workers 4
.venv/Scripts/python.exe -m experiments.policy_dependency_sync.run_pdsg_obs001 run --manifest docs/pdsg_obs001_manifest_20260906.json --output experiments/policy_dependency_sync/results/pdsg_obs001_reproduction --workers 4
```

No GPU, HPC4, standard-benchmark run, or ICML-readiness claim is authorized by
this preregistration.
