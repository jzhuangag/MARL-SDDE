# Real-trajectory MAPPO response development decision

Date: 2026-10-14
Experiment: `ICML27-MAPPO-OBSERVABLE-RESPONSE-CPU-001`
Preregistration commit: `4e5d1c3`
Status: **stop the frozen design; do not authorize MARL performance runs**

## Frozen-run integrity

The CPU run used the clean pinned HARL commit
`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`, PyTorch 2.6.0 CPU, and isolated
PettingZoo 1.24.3. It covered the preregistered two MPE tasks, four seeds, three
whole-trajectory coupling regimes, two training snapshots, all four q values,
and both charged work modes. It produced 24 profiles, 384 candidate rows, and
31,200 joint environment transitions (78,000 individual-agent transitions).
Role seed overlap was zero. The compact payload hashes are recorded in
`results/marl_response_development_20261013/SHA256SUMS`; raw role buffers remain
local and ignored by Git. The run completed in 550 seconds. There was no failure
record, no evaluation-return input to the candidate score, and no GPU run.

| Frozen criterion | Measured | Decision |
|---|---:|---|
| finite, disjoint, exactly charged payload | 24 profiles; 384 rows; 31,200 joint steps; zero role overlap | pass |
| identical candidate initialization | all rows | pass |
| all q in {1,2,4,8} | all 48 stage contexts × 2 work modes | pass |
| shared-data fixed-step parameter invariance, max | 1.22e-6 (limit 1e-5) | pass |
| critic same-sensor quadratic relative error, median | 0.0273 (limit 0.10) | pass |
| critic direction on independent current-stage trajectories | 0.8099 (minimum 0.75) | pass |
| previous-stage critic direction on the next-stage truth | 0.5052 (minimum 0.75) | **fail** |
| shadow-candidate q-risk normalized regret, median | 0.000 across 96 contexts; 24 ties | descriptive only |

The frozen stop rule is triggered by cross-stage prediction failure. No replay,
seed retry, tuning, task removal, gate edit, or performance experiment followed.

## Interpretation and limits

Within a fixed training snapshot, the critic's local quadratic response agrees
well with its independent-trajectory risk change and gets its direction right
about 81% of the time. The actor diagnostic is weaker: its independent-truth
direction accuracy is 61.7%, and its objective is an unclipped PPO diagnostic,
not episodic return. Shadow-candidate q ranking has zero median normalized
regret across all contexts and 0.0828 among the 72 non-tied contexts. Predicted
q counts were {1: 43, 2: 19, 4: 14, 8: 20}; held-out best-q counts were {1: 38,
2: 17, 4: 9, 8: 32}. These are descriptive only: the runner observes each q
candidate's realized parameter displacement before scoring it. This is a
shadow-candidate diagnostic, not a pre-action controller or evidence for a
deployable q policy. The 24 near-tie contexts further limit interpretation.

The previous-stage prediction fails across both tasks. At stage 1, critic sign
accuracy grouped by task and coupling ranges from 21.9% to 71.9%; the failure is
not confined to one task or one correlation regime. This is evidence that the
validation objective moves during training, so a local expansion should not be
reused across policy updates without a fresh sensor or an independently proved
bound on objective motion. This diagnosis is consistent with the exact transfer
decomposition in `theory/normalized_mappo_response.md`.

The 384 rows share only four independent seed values and are nested candidates,
not 384 independent replications. No inferential performance claim or MARL
return claim is made. These observations qualify one local critic-response
interface in two MPE tasks; they do not establish a Markov confidence bound,
finite-time return guarantee, low-overhead deployment, or ICML readiness.

## Next admissible step

Any continuation needs a new preregistration and fresh seeds. The most direct
test is a **refresh-at-each-snapshot** rule: collect and charge a new sensor
buffer after the policy changes, then predict candidate critic-risk changes
only at that snapshot. Carry forward no previous-stage certificate. Keep the
same q catalogue, tasks, independent truth data, and all resource ledgers. First
test on CPU whether fresh-sensor direction and q ranking replicate on new seeds;
only a pass can motivate a separate return experiment. If local prediction
fails on fresh seeds, stop this controller path and redesign its observable
response model before any GPU allocation.

Refreshing a sensor at every snapshot is necessary for this local predictor,
but by itself does not make the score actionable. The current runner forms a
candidate displacement by actually training each q-specific shadow update and
then scores that displacement. A deployed controller must forecast candidate
displacements before collecting q-specific data. Therefore the next CPU gate
must first test a pre-action score built only from already charged history
(for example, per-stream gradient moments with an explicit stale-policy
correction), against fresh, disjoint outcomes on new seeds. It must include
the actual cost and uncertainty of that score. If no pre-action predictor can
rank the full q set without acquiring every candidate batch, stop the q
controller path; do not move the current shadow-candidate score into an online
performance experiment.

The refresh rule is a design implication, not a result of this experiment. The
development evidence remains immutable and must not be reused as confirmation.
