# SMACv2 coupling calibration validation

## Decision

`TSP-SMACV2-COUPLING-CAL-001` failed one of six frozen gates, so the original
24-run `TSP-MARL-SMACV2-DEV-001` array is not authorized and was not
submitted.  The failed gate is retained without threshold or seed changes.

## Results

All four A30 jobs (`1851283_0`--`1851283_3`) completed with exit code `0:0` in
3:59--4:22.  Each collected 64 blocks from eight rollout workers, performed
zero policy updates, and did not compute a return or win rate.

The two independent median pairwise correlations were `-0.05478` and
`0.02509`; the two common-seed/shared values were `0.42850` and `0.57282`.
Their regime medians were `-0.01484` and `0.50066`, giving separation
`0.51550`.  Both independent runs selected `q=8` and both shared runs selected
`q=1`.  Thus finiteness, independent correlation, separation, and both
selection gates passed.  The preregistered shared-correlation floor `0.75`
failed.

This is not evidence that SMACv2 is unusable.  It establishes that synchronized
procedural seeds and action uniforms induce a reproducible *moderately*
correlated regime after StarCraft dynamics amplify small path differences,
not the near-replication regime assumed by DEV-001.  A future performance
study must pose that moderate-dependence question explicitly and use new
seeds; it cannot relabel DEV-001 as passed.

## Provenance

The preregistration commit is `7e783a63898c204634b203c21bde4f604a34abb1`.
Raw artifacts remain under
`/scratch/jzhuangag/MARL-SDDE-TSP-SMACV2-DEV-001/artifacts/coupling-calibration`
and occupy 1.9 MB.  The analyzer result is
`/scratch/jzhuangag/MARL-SDDE-TSP-SMACV2-DEV-001/provenance/coupling_calibration_result.json`
with SHA-256
`917df9603fc36cf6efc709d2d20e3f45aa1420eadd42b9cd0c9542e8bf7b1296`.

The first-generation `SHA256SUMS` files self-listed and therefore report only
their own entry as failed; every payload entry verifies.  They are preserved
unchanged.  Subsequent scripts exclude `SHA256SUMS` from its own manifest.
