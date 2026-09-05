# Clocked MPG Layer-0 fresh-seed CPU pilot preregistration

## Purpose and separation

This pilot is the final CPU feasibility gate before any standard GPU benchmark
is designed or run.  It tests one unified claim: under charged asynchronous
strategic updates, a sample-split Lyapunov-debt controller can trade a small
amount of mean performance for nontrivial lower-tail protection without
falling back to zero or full updates.

Development seeds 1701, 2701, and 2702 and every prior Layer-0 output are
excluded.  They informed the architecture and exposed the zero-baseline and
inactive-debt failures.  They are not inferential evidence.  The pilot uses
eight fresh CRN seeds 3701--3708.  Its primary and isolated reproduction runs
must execute the same committed code and configuration.

## Frozen execution

- Environment: HARL continuous `simple_spread_v2`, upstream commit
  `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.
- Policies: three non-shared HARL `StochasticPolicy` blocks with `[32, 32]`
  hidden layers.
- Service profiles: balanced `(1,1,1)` and heterogeneous `(1,1.55,4)`.
- Methods: `strategic_split`, equal-cost `raw_full_data`, and equal-cost
  `raw_half_data`.
- Per run: 16 fully charged baseline episodes, 400 packets, two independent
  25-step trajectories per packet, and 61,200 actor transitions.
- Controller: `epsilon=V=10^-4`, curvature upper 5, mixed-drift coefficient 1,
  learning rate `5e-4`, and maximum proposal norm `0.05`.
- Compute: local CPU only, four workers.  No GPU, HPC4, formal seed, or remote
  artifact is authorized by this preregistration.

The frozen machine-readable configuration is
`experiments/clocked_async_mpg/harl_layer0_cpu_pilot_config.json`.  The runner
and analyzer are committed before outcome access.  Generated primary and
reproduction summaries remain ignored; their SHA-256 values and the validation
decision will be recorded after execution.

## Mandatory gates

1. **L1 validity/accounting:** 48 finite rows, exactly 61,200 charged and
   completed actor transitions per row, and zero self-fresh error.
2. **L2 positive learning:** strategic mean return change is positive in both
   service profiles.
3. **L3 heterogeneous lower tail:** strategic lower-quartile mean final return
   exceeds full-data raw async by at least 2% of the absolute raw lower-tail
   return.
4. **L4 heterogeneous mean safety:** strategic mean final return is no more
   than 1% below full-data raw async.
5. **L5 directionality:** strategic strictly beats paired full-data raw async in
   at least five of eight heterogeneous seeds.
6. **L6 nontrivial debt:** in each profile, the mean fraction of strategic
   packet scales strictly between zero and one lies in `[0.1,0.8]`.
7. **L7 balanced safety:** strategic mean final return is no more than 1% below
   full-data raw async under balanced service.
8. **L8 split-cost value:** strategic exceeds raw-half-data in both mean and
   lower-quartile final return under both profiles.
9. **L9 reproducibility:** primary and isolated reproduction summaries are
   byte-identical.

All gates are mandatory.  Any failure prevents GPU authorization.  Gates,
seeds, thresholds, profiles, methods, and budgets cannot be changed after
outcome access.  A failure may motivate a new theory/algorithm version, but
cannot be relabeled or rerun into a pass.

## Frozen hashes

| artifact | SHA-256 |
|---|---|
| configuration | `98d0afe3a461d100aff14863ecae400d3fbd2175ec0fbcc588309a87fed56e35` |
| comparison runner | `00cdbf660b9868fa5adb78427612651c657a51d67434ab542244456a1aff59c1` |
| gate analyzer | `86b7b5aace1097f68a25ea1302e05eb3c11b33a083b34bce4b7019774ca1a926` |
