# EXP-018A preregistration: frozen nonlinear TD gradient identity

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-02
- Verification Status: STATICALLY VERIFIED
- Version Label: exp018a_preregistration_v1

## Status and scientific scope

EXP-018A is a local-CPU implementation pilot for the fixed-parameter nonlinear
variance identity in Theorem 9 of `proof_program_joint_ms.md`. It is not a
learning-curve experiment and does not train a participation controller. It
tests whether the variance of an averaged nonlinear TD gradient follows

\[
v(q,\rho)=\rho+\frac{1-\rho}{q}
\]

when complete Markov streams have unchanged single-agent marginals and two
agents share the common stream with probability `rho`.

This preregistration authorizes the 64-seed CPU pilot only after the immutable
preregistration commit. It allocates no formal seed and authorizes no GPU,
HPC4, delayed-learning, dual-budget, controller, or nonlinear-convergence
claim.

## Frozen design

- tasks: `CartPole-v1`, `Acrobot-v1`;
- fixed behavior policies: the public state-feedback policies in the runner;
- independent regeneration probabilities: 0.20 and 0.05;
- frozen network checkpoints: two deterministic `MLP-ReLU-64-64-1`
  initializations;
- `q in {1,4,16,32}` and `rho in {0,0.5,0.9}`;
- 64 transitions per complete source block;
- one common and 32 iid private streams per seed/task/mixing cell;
- 16 frozen normalized Rademacher projections of every TD gradient;
- 64 new deterministic pilot seeds; formal seeds remain `null`.

Every source has its own regeneration clock. Agent `i` uses common source zero
with probability `sqrt(rho)` and otherwise uses private source `i`. Therefore
the probability that two agents use the same common source is `rho`. No
observation noise is added.

## Endpoints and analysis

The runner performs backward passes at frozen model parameters but applies
zero parameter updates. For every seed/cell/q it stores 16 scalar projections
of the averaged TD gradient. The analyzer estimates projection-wise sample
variances across the 64 seed clusters, divides each `q>1` variance by the
same-cell `q=1` variance, and compares the ratio with `v(q,rho)`.

The pilot also records empirical pairwise common-source rates, the variation
of the `q=1` variance across rho, monotonicity in q, manifest integrity, and
parameter hashes before and after all backward passes. Gradient projections
are repeated measurements, not independent seeds.

## Frozen mandatory gates

1. **G1 shape/finite/unique:** exactly 6,144 unique seed-cell-q rows and all
   projection values finite.
2. **G2 freeze:** every manifest hash matches and every before/after parameter
   hash is identical.
3. **G3 coupling:** maximum absolute pairwise-share error across registered rho
   is at most 0.10.
4. **G4 variance identity:** median and p90 relative calibration errors are at
   most 0.20 and 0.50.
5. **G5 marginal invariance:** median and p90 spreads of `q=1` variance across
   rho are at most 0.30 and 0.75.
6. **G6 direction:** empirical variance is nonincreasing over
   `q={1,4,16,32}` in at least 80% of registered task/mixing/checkpoint/rho/
   projection paths.
7. **G7 scope:** the result remains a frozen-gradient mechanism pilot with no
   delay, budget, controller, convergence, formal, or significance claim.

All seven gates must pass before a separate formal preregistration may be
designed. Thresholds, seeds, tasks, cells, runner, and analyzer must not be
changed after observing the pilot.

## Frozen workload and commands

- environment transitions: 540,672;
- source-gradient evaluations: 16,896;
- output rows: 6,144;
- projected CSV: 8 MB;
- projected peak memory: 512 MB;
- compute: local CPU, no GPU.

```text
conda run -n ust2 python experiments/nonlinear_markov_td/run_exp018a_direct_gradient.py --pilot --output-dir experiments/nonlinear_markov_td/results/exp018a_pilot_20260802
conda run -n ust2 python experiments/nonlinear_markov_td/analyze_exp018a_direct_gradient.py --input experiments/nonlinear_markov_td/results/exp018a_pilot_20260802/projections.csv --output-dir experiments/nonlinear_markov_td/results/exp018a_pilot_20260802
```

The deterministic reproduction must use a new output directory and must match
`projections.csv` and `summary.json` byte-for-byte. Wall time is excluded from
scientific outputs.

## Frozen code provenance

- parent: `4ab73b536b0aa94b6c0ebbcd4d3fc18b48ab80e9`;
- static manifest hash:
  `421b3f433671ee7f35920eeac20746786893bfe10fe598140cbc84cd8dc4fc0d`;
- config SHA-256:
  `c0d80dd725c328a121f5f27c1abde5c3247a0e92b4928ed4f6fc096183e6970e`;
- runner SHA-256:
  `7096d7a1e195922eabf58db16593daf76691182af2565211c2ab2ee1f0a44998`;
- analyzer SHA-256:
  `869809c9444fc6ae9815ae11d0f965ae2cf2c8b30c3900ff43a88093f832c5f9`;
- test SHA-256:
  `f5d44c7510a0c4d8a71799492029c2122ea8a8a2af848f4bb90e14be3fa95a40`.
