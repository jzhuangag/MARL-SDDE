# EXP-018A CPU pilot validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-02
- Verification Status: VERIFIED WITH PREREGISTERED FAILURE
- Preregistration Commit: `13736469743fc8d5773b976661c1e01ecd540ee6`

## Decision

EXP-018A is an honest **5/7 pilot failure**. Formal preregistration is not
authorized. No threshold, seed, task, rho, q, checkpoint, or analyzer was
changed after observing the result.

## Frozen-gate ledger

| Gate | Result | Value |
|---|---:|---:|
| G1 shape/finite/unique | PASS | 6,144/6,144 unique finite rows |
| G2 manifest/parameter freeze | PASS | all hashes matched; zero parameter changes |
| G3 pairwise sharing | PASS | rho 0/0.5/0.9: 0 / 0.496062 / 0.898398 |
| G4 variance identity | PASS | median/p90 relative error 0.073035 / 0.323833 |
| G5 q=1 rho invariance | **FAIL** | median 0.327756 > 0.30; p90 0.613102 < 0.75 |
| G6 monotone q paths | **FAIL** | 256/384 = 0.666667 < 0.80 |
| G7 scope boundary | PASS | mechanism-only claim retained |

The maximum calibration error was 1.579061 and is descriptive, as frozen in
the analysis plan. The main calibration distribution passed its registered
median and p90 thresholds, but the conjunctive gate ledger still fails.

## Post-hoc location diagnosis

The diagnosis changes no gate. Monotone-path fractions by rho were 1.0000,
0.5547, and 0.4453 for rho 0, 0.5, and 0.9. The dominant failures were the
near-tied `q=16` versus `q=32` comparison: its theoretical variance factors are
0.53125 versus 0.515625 at rho 0.5 and 0.90625 versus 0.903125 at rho 0.9.
Thus G6 asked 64 seeds to order differences of about 2.94% and 0.35% while the
registered practical-effect scale elsewhere was 5%.

G5 used different, identically distributed q=1 source assignments across rho.
Its expectation is invariant, but the 64-seed sample variances fluctuate; the
median spread missed the frozen threshold by 0.027756. This explains the
failure but does not reverse it.

## Execution and reproduction

- pilot runtime: 160.8 seconds on local CPU;
- deterministic reproduction runtime: 139.6 seconds;
- pilot and reproduction `projections.csv`: byte-identical;
- pilot and reproduction `static_manifest.json`: byte-identical;
- pilot and reproduction `validation.md`: byte-identical;
- `summary.json`: not byte-identical because the analyzer records the absolute
  pilot/reproduction input path; a one-line diff confirms all scientific
  values and checksums are identical.

Because the preregistration required both CSV and summary byte identity, the
path-dependent summary is recorded as a provenance defect even though the
scientific CSV is exactly reproducible.

## Artifact provenance

- pilot CSV SHA-256:
  `73ed8fde7d5bf394aa0dc5d93a5902cc4c6a4194c16d15e6efc77158f91a2a35`;
- static manifest SHA-256:
  `02d997eb6e860068b17313fd81967c96c5dee1ac140012914c2482735740d123`;
- pilot summary SHA-256:
  `85b8ba31c04e53c072baced7b88a07fc93436d24afc2ea431dd350c24202ff6a`;
- reproduction summary SHA-256:
  `4c92d1218a8f20796e185aa90c8cf81086e46af3964c1a37641707efac428a7a`.

Raw artifacts remain local under the ignored result directories. No GPU,
HPC4, `/project`, or `/scratch` resource was used.

## Test status

- preregistration commit regression: `265 passed, 7 skipped`;
- post-result EXP-018A targeted regression: `20 passed`;
- post-result full-repository regression: `TIMEOUT/UNVERIFIED` after 600 seconds.

The timed-out pytest process continued consuming one CPU core with stable
memory and emitted no assertion failure before its monitor connection ended;
it was terminated at the hard timeout. The full-repository state is therefore
not reported as passing. Files outside the targeted EXP-018A scope were
unchanged from the preregistration commit that passed the full regression.
