# Exact multi-state clocked-async MPG confirmation validation

## Decision

All frozen C1--C10 gates pass.  The exact multi-state confirmation supports
the service-rate--coupling phase and authorizes design of a separate stochastic
CPU packet study.  It is not formal evidence, does not establish sampled-RL
performance and does not authorize a standard MARL GPU benchmark.

The preregistration commit is `d55c736`.  Frozen sources and configuration were
unchanged during both executions.

## Registered results

- endpoints: 2,048 (`16 cells * 64 seeds * 2 policies`);
- target endpoint coverage: 1.000;
- primary cells: 12 (service ratios 2, 4 and 8 at four couplings);
- primary cells with strictly smaller median async time: 12/12;
- geometric primary async/shadow target-time ratio: `0.4445572803`;
- geometric primary final normalized-gap ratio: `0.5281518167`;
- maximum realized/registered event delay: `9/12`;
- every directional heterogeneity and coupling-cost check passed;
- all nine pre-reproduction gates passed without threshold changes;
- endpoints and summaries reproduced byte-for-byte in a separate directory.

The homogeneous-service controls are deliberately retained.  Their median
time ratios across increasing coupling are `0.9608`, `0.7850`, `1.1553` and
`1.1243`: async is not universally better.  The positive result is a phase
claim under heterogeneous service, not a no-harm claim.

## Artifact integrity

```text
config SHA-256
2d2af65f81d50b488c93eb26f553412c8740446421d2612a71b09469007f5177

endpoints.jsonl SHA-256 (both runs)
c7a91998b1f5ab6c7666d8bdf6c588e20336808af04b8daf11868a4eff31cc76

summary.json SHA-256 (both runs)
65e1c22a2694f3691440a7692a6f1ed1787cfe923b01baccb6a55c304e4f4e57
```

Local ignored artifacts are preserved at:

```text
experiments/clocked_async_mpg/results/exact_confirmation_20260901
experiments/clocked_async_mpg/results/exact_confirmation_20260901_reproduction
```

No HPC4, GPU, remote storage or old formal seed was used.

Final verification on the validation source state:

```text
theorem-facing package: 46 passed in 0.91s
complete experiments regression: 970 passed, 7 skipped in 119.81s
```

## Interpretation

The result confirms the causal mechanism under exact gradients:

1. a single-flight packet is self-fresh and only pays teammate-policy drift;
2. heterogeneous local completions provide fresh adaptive policy queries;
3. a frozen-policy shadow barrier uses all completed work but cannot turn
   duplicate exact packets into additional query depth;
4. stronger coupling raises the asynchronous delay cost;
5. greater service heterogeneity enlarges the barrier-removal gain.

The structural correction is scientifically material.  Before self-freshness
was proved, the generic history incorrectly charged diagonal sensitivity and
the first tainted development grid lost in all 16 cells.  That failure remains
recorded and never enters the confirmation population.

## Resource-accounting qualification

The exact-value stage has no environment-transition estimator.  Both workers
are continuously occupied in elapsed time, and every **completed** packet is
recorded.  The shadow implementation may cancel an unfinished frozen-policy
packet at a barrier; its partial service is included in elapsed wall-clock but
not represented as a completed packet count.  Therefore C8 is valid under its
frozen completed-packet definition, but this stage must not claim equality of
sample transitions.

The next stochastic runner must record attempted transitions, completed
transitions, partial/cancelled work, applied updates and discarded/stale
packets separately.  A result lacking this accounting is invalid regardless
of return or wall-clock performance.

## Remaining ICML gates

- replace exact gradients by fully charged fixed-horizon Markov trajectory
  packets and verify the bias/noise terms;
- include the same shadow barrier plus raw async and a competitive staleness-
  aware asynchronous baseline;
- keep service, samples and partial work identical or explicitly accounted;
- independently confirm on fresh seeds;
- only then design standard CTDE MARL benchmarks and request GPU/HPC4;
- complete a contribution-led manuscript and citation-integrity audit.

The current story is now coherent and positively supported at the exact-theory
level, but it is not yet ICML-ready.
