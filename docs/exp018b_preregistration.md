# EXP-018B formal preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-02
- Verification Status: STATICALLY VERIFIED
- Version Label: exp018b_formal_preregistration_v1

## Objective and boundary

EXP-018B is a 192-seed local-CPU formal calibration of the frozen-parameter
nonlinear TD gradient variance identity. It uses EXP-018A only as disclosed
design information. EXP-018A remains a 5/7 pilot failure and supplies no formal
outcome.

The formal estimand is the ratio between the variance of a q-agent averaged
gradient and the q=1 variance, compared with
`rho+(1-rho)/q`. The study makes no online-controller, delayed-learning,
dual-budget, nonlinear-convergence, actor-critic, or GPU claim.

## Frozen corrections from the pilot

1. The private source-1 q=1 gradient is reused byte-for-byte for every rho row.
   This is a common-random-number implementation of the exact fact that rho has
   no pairwise meaning at q=1.
2. Strict adjacent-q ordering is descriptive only when the theoretical
   separation is at least 5%. It is never a mandatory gate.
3. Scientific summaries contain `projections.csv`, not an absolute input path,
   and must be byte-identical on deterministic reproduction.

These changes correct estimand and provenance defects; no failed EXP-018A
threshold is loosened.

## Frozen design

- 192 new formal seeds, disjoint from all 64 EXP-018A pilot seeds;
- CartPole-v1 and Acrobot-v1;
- independent regeneration probabilities 0.20 and 0.05;
- two frozen MLP-ReLU-64-64-1 initializations;
- q in {1,4,16,32}, rho in {0,0.5,0.9};
- 64 transitions per independent complete source stream;
- 16 fixed normalized Rademacher projections;
- 18,432 output rows, 50,688 gradient evaluations, 1,622,016 environment
  transitions;
- zero optimizer step and zero parameter update.

## Co-primary endpoints and multiplicity

For each of 72 fixed task/mixing/checkpoint/rho/q>1 ratios, average the 16
projection variances, divide by the same stratum/rho q=1 variance, and compute
relative calibration error. The two co-primary functionals are the median and
p90 errors across the 72 ratios.

Resample the 192 seed clusters jointly 5,000 times using seed `18240101`. The
one-sided 97.5% bootstrap upper bound is computed for each endpoint, giving
Bonferroni familywise alpha 0.05. Both conditions are mandatory:

- median upper bound <= 0.20;
- p90 upper bound <= 0.50.

No p-value, cell selection, projection-as-seed analysis, or replacement
endpoint is permitted.

## Mandatory gates

1. F1: exactly 18,432 finite, unique registered rows and the exact seed set.
2. F2: manifest matches and all before/after parameter hashes are equal.
3. F3: every q=1 projection is exactly equal across rho within each seed and
   stratum.
4. F4: maximum pairwise-share absolute error is at most 0.06.
5. F5: median-error 97.5% upper bound is at most 0.20.
6. F6: p90-error 97.5% upper bound is at most 0.50.
7. F7: summary schema is path-independent.
8. F8: scientific scope remains frozen-gradient calibration only.
9. R1: independent rerun produces byte-identical `projections.csv`,
   `static_manifest.json`, and `summary.json`.

F1-F8 must all pass before reproduction. A formal claim is authorized only
after R1 also passes. Any failure is terminal for EXP-018B.

## Frozen commands and resources

```text
conda run -n ust2 python experiments/nonlinear_markov_td/run_exp018b_direct_gradient_formal.py --formal --output-dir experiments/nonlinear_markov_td/results/exp018b_formal_20260802
conda run -n ust2 python experiments/nonlinear_markov_td/analyze_exp018b_direct_gradient_formal.py --input experiments/nonlinear_markov_td/results/exp018b_formal_20260802/projections.csv --output-dir experiments/nonlinear_markov_td/results/exp018b_formal_20260802
```

Projected runtime is 8.1 minutes, peak memory 768 MB, and CSV size 24 MB.
Execution is local CPU only.

## Provenance

- parent: `c58ed5d60afb2e32b07ab21b6e3c0c5a5450a19a`;
- manifest hash: `be491d5175f6c2af36e1d65d2497595f67472e9c37bfd069d561d82a500d2418`;
- config SHA-256: `b669cdc77d0b2bd86a73ee945a689387b06d2e3756acffd0cedb82c027869355`;
- runner SHA-256: `9909ee779331bc49ff4d750623e0edca5900cb4c249c84932f3e6f36c1534b58`;
- analyzer SHA-256: `55d92b0cfbc3a04ba428dee0e0a36a35ba31953b2f3bacee70f6b284d5671c0f`;
- test SHA-256: `a3d278511594ed380004c8e7d0f5a4dc7337797ee4fc8742abe0272a8e6dce50`.
