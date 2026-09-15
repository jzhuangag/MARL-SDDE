# TSP-MARL-SMACV2-HEADROOM-001 validation

## Decision

The frozen development-only headroom experiment stops.
It does not authorize a SMACv2 controller experiment or new confirmatory seeds, and its result is not included as positive manuscript evidence.

The data show a clear benefit of broad participation in both registered dependence regimes rather than the intended regime reversal.
Under independent coupling, fixed `q=8` improves team-return AUC over fixed `q=1` by 20.5014%.
Under shared coupling, fixed `q=1` is 19.4300% worse than fixed `q=8`; equivalently, `q=8` improves over `q=1` by 24.1156%.
The strong static method is consequently fixed `q=8` in both regimes, so the regime oracle has zero gain over the strong static method.

## Frozen design and execution provenance

- Experiment: `TSP-MARL-SMACV2-HEADROOM-001`.
- Task: SMACv2 `terran_10_vs_10` with MAPPO and centralized training/decentralized evaluation.
- Preregistration commit: `d0b4a93f2368e5223cd2504d26b08c5478ccd3ab`.
- Sampler-boundary amendment commit: `7757a07f91975d84e2a18b5c6c8ea78a5a119074`.
- Infrastructure-recovery amendment commit: `527e9c24157b2b3b285ea8ffc277c1f630ceaeca`.
- Original array: `1851296`; seven cells completed, while the original `q=8`, shared, seed-`81102` cell failed at update 1200 because float32 cumulative probability left a spurious out-of-support interval.
- Recovery 1: `1854988`; cancelled after 1:17:29 because two StarCraft evaluation processes crashed before the first evaluation completed and left the vector environment deadlocked.
- Recovery 2: `1856702`; completed on `gpu14` in 3:07:34 with exit code `0:0`.
- Scientific output root: `/scratch/jzhuangag/MARL-SDDE-TSP-SMACV2-DEV-001/artifacts/headroom`.
- Analysis root: `/scratch/jzhuangag/MARL-SDDE-TSP-SMACV2-DEV-001/analysis/headroom-20260914`.
- No output was written to `/project` or `/home`.

Both incomplete directories are preserved but contain no `tsp_bridge_metadata.json` and are excluded mechanically by the frozen analyzer.
The recovery changes neither seed, task, coupling, participation, policy architecture, optimizer, budgets, evaluation schedule, nor gate.

## Validity and reproduction

- Exactly 8 unique completed metadata records match the frozen `2 regimes x 2 actions x 2 seeds` design.
- Every completed run passes its `SHA256SUMS` manifest.
- Every reported metric is finite.
- Exact message and environment accounting passes for every run.
- The upstream HARL commit is `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2` in every run and the upstream checkout is clean.
- Two independent analyzer executions produce byte-identical summary CSV, figure PDF, and pre-replay gate JSON.
- Final gate JSON SHA-256: `35fdc8e28396efba9bc3acce34d72133745080f4577937475f9fb261493f02f1`.
- Final summary CSV SHA-256: `a8ff6198ffa4bedfd9ad41d7a37ad5de91f50e49c71542f55177f69ca1183f03`.
- Final curve PDF SHA-256: `946fbb0482f9e088256986a24a64e56e8a79e7a6eff4755eda595c87597a6d51`.
- Analysis manifest SHA-256: `474cb133e7302b4662edaad468f914cea9241beeb1c4053c1d200f58e8b367f0`.

## Registered metrics

| Coupling | Method | Team-return AUC | Terminal return | Win-rate AUC |
|---|---:|---:|---:|---:|
| Independent | fixed `q=1` | 8.874307 | 10.423948 | 0.046721 |
| Independent | fixed `q=8` | 10.693662 | 11.143614 | 0.144271 |
| Shared | fixed `q=1` | 8.935954 | 9.229113 | 0.061932 |
| Shared | fixed `q=8` | 11.090919 | 12.631863 | 0.167365 |

The regime-oracle and strong-static team-return AUC are both 10.892291 because fixed `q=8` wins both regimes.

## Gate ledger

| Mandatory gate | Result |
|---|---:|
| Finite, complete, exact accounting, clean upstream | Pass |
| Independent fixed `q=8` AUC gain at least 5% | Pass; 20.5014% |
| Shared fixed `q=1` AUC gain at least 5% | **Fail; -19.4300%** |
| Regime oracle gain over strong static at least 3% | **Fail; 0%** |
| Terminal-return directions agree with AUC | **Fail in shared regime** |
| Analysis replay byte-identical | Pass |

Final decision: `stop` (`3/6` mandatory gates pass).

## Scientific interpretation

The registered shared-randomness intervention changes the rollout dependence while preserving the intended individual environment and policy marginals, but on this task and budget it does not make local participation preferable.
The additional actor transitions obtained by `q=8` dominate its message-induced horizon reduction in both regimes.
This is a benchmark-specific lack of adaptation headroom, not a numerical-power ambiguity: even the outcome-aware regime oracle selects the same fixed action everywhere.

The result leaves the existing positive PettingZoo MAPPO confirmation unchanged.
It should remain in internal provenance as a stopped task-diversity extension and should not be converted into a positive claim, used to alter the frozen thresholds, or followed by a controller stage under the same experiment identifier.
