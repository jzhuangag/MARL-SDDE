# Strategic-drift oracle-headroom audit

## Decision

**Pass the architecture-headroom gate, not the algorithm-evidence gate.**  An
arrival-time controller with exact birth-policy directional derivatives can
recover all of the raw asynchronous learner's early time-to-target performance
while improving substantially over the conservative pathwise constant step.
This justifies implementing an observable sampled certificate.  Because the
directional derivative is oracle information in this audit, none of these
numbers are formal evidence for the final algorithm.

## Frozen development choices

- namespace: `strategic-drift-oracle-development-v1`;
- eight development seeds, disjoint from both stochastic confirmations;
- couplings `{0,.08,.16,.24}` and service ratios `{1,2,4,8}`;
- fixed-horizon Markov REINFORCE with horizon 16 and batch size 16;
- target normalized potential gap `.3`, terminal time 180;
- debt controller fixed before the full grid at `V=10`, per-arrival risk
  budget `.001`;
- comparators: exact-gradient hard shield, pathwise constant certificate, raw
  common-step asynchronous learning, and fully utilized shadow barrier.

The preliminary choice of `V` and budget used one seed in three cells and is
therefore development tuning.  No value from this audit may be presented as
an independently confirmed hyperparameter result.

## Headroom results

All policies reached the target in all 16 cells and all eight seeds.  Ratios
below compare the budgeted oracle controller with each comparator.

| Population | Comparator | Time ratio | Charged-work ratio | Final-gap ratio | Faster cells |
|---|---|---:|---:|---:|---:|
| all 16 cells | hard shield | 0.57975 | 0.58461 | 0.19160 | 16/16 |
| all 16 cells | pathwise constant | **0.67394** | **0.67671** | **0.61994** | 12/16 |
| all 16 cells | raw async | **1.00000** | **1.00000** | 0.98274 | 0/16 |
| all 16 cells | shadow barrier | **0.37204** | **0.37694** | 0.65104 | 16/16 |
| 12 heterogeneous cells | hard shield | 0.57577 | 0.58139 | 0.20632 | 12/12 |
| 12 heterogeneous cells | pathwise constant | **0.63015** | **0.63306** | **0.63205** | 9/12 |
| 12 heterogeneous cells | raw async | **1.00000** | **1.00000** | 1.08097 | 0/12 |
| 12 heterogeneous cells | shadow barrier | **0.27151** | **0.27496** | 0.58425 | 12/12 |

The equality with raw async is mechanistic rather than rounded coincidence.
The debt controller selects the maximum scale during the high-signal phase, so
its trajectory is identical to raw async until the registered target is first
reached.  Later, as directional gain falls and certificate debt rises, its
mean scale over the full run falls to `0.2300--0.3143` across cells.  Mean
terminal debt lies in `8.546--9.883`; it is finite but is not evidence of an
asymptotic queue-stability theorem.

The heterogeneous final-gap ratio against raw async is `1.08097`, so the
controller does not dominate raw async.  Separate two-seed stress probes at
larger coupling and service ratios likewise did not exhibit a robust raw-async
failure.  The paper must not motivate the controller by claiming that raw
async empirically diverges in this benign tabular family.  The supported claim
is instead that the online scalar rule removes much of the conservative
certificate cost and exposes a tunable risk--speed frontier.

## Reproduction record

Ignored artifact:

`experiments/clocked_async_mpg/results/strategic_drift_oracle_development_v1/summary.json`

SHA-256:

`8cdb1ffa896e70b6483ec36c7a92eee3e2651745f52313a01abd4c9ac7b793de`

Source hashes:

- controller: `d922b5f3f242bdf39dbc96a2ea7752b8a32a6f066812b0f186a44244d973594b`;
- oracle simulator: `151b76192d8deb48d57a581e5c53c84a2c08b174276742a4f205d65cdffb94c5`;
- runner: `fb70bddb3d729985d744745bfa0ef71276cc664bfe6f1059fbcd30f3758fc11b`.

Command:

```text
.venv/Scripts/python.exe -m experiments.clocked_async_mpg.run_strategic_drift_oracle_development --output experiments/clocked_async_mpg/results/strategic_drift_oracle_development_v1/summary.json --seeds 8 --workers 4 --namespace strategic-drift-oracle-development-v1
```

The clocked-MPG package regression after the run passed `74` tests in
`134.85 s`.

## Next gate

The next CPU task is not another oracle performance scan.  It is to replace
the exact directional derivative by a fully charged, predictable Markov-packet
confidence interface and verify its coverage and nontrivial acceptance.  Only
after that interface and the standard-benchmark protocol are frozen may a GPU
pilot be launched.
