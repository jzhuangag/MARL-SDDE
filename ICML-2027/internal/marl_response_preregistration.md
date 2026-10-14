# Real-trajectory MAPPO response qualification

## Material Passport

- Origin: academic-research-suite / experiment-agent, autonomous implementation.
- Date: 2026-10-13.
- Status: design and source must be committed before development.
- Scope: real neural actor/critic response, not controller return superiority.

The fixed tasks are cooperative MPE simple_spread_v3 with discrete actions and
simple_reference_v3 with continuous actions. This introduces actual multi-agent
trajectories and both policy-head types, not a second MuJoCo performance claim.
Centralized value training and a shared decentralized local-observation actor
are explicit. q is the number of rollout streams; games have three/two agents.

Use four fresh seeds, three coupling regimes, two policy-training snapshots,
every q in {1,2,4,8}, and fixed-step versus matched-processed-row optimizer work.
The 48 stage contexts yield 384 candidate payloads. Exactly 31,200 joint
environment transitions are generated, including all sensor/truth/warmup roles.
Individual-agent transitions are recorded separately. Truth generation is
evaluator-only; sensor, candidate, and warmup work is acquisition/training cost.
No return-superiority budget comparison is made from this qualification.

Whole-trajectory shared/independent seed coupling leaves every stream's native
environment/policy marginal unchanged. Distinct roles/stages have disjoint seed
domains. Same-role duplicates in the shared regime are intentional and charged.

Keep HARL's Adam, clipped PPO, value clipping, Huber loss, ValueNorm, and five
epochs. Use its 128x128 architecture with a prospectively declared tanh choice
to make derivative diagnostics smooth. Monte Carlo truth loss and unclipped
actor surrogate are diagnostic potentials, not replacement training objectives.

Quadratic response uses exact directional differentiation, no fitted exponential
curve, target-parameter access, or evaluation-return feedback. Include the
normalizer's changing mean/scale; report a parameter-only ablation. Test old
response predictions at the later snapshot without fitting to later truth.

An isolated PettingZoo 1.24.3 directory leaves the installed runtime untouched.
The installed task entry-point SHA-256 values are
`F3880F7436E1AD61F1023EDDE6BCA040A6F2AA5203BA71F0CB333F39B93A51BE`
for `simple_spread_v3.py` and
`773DFF34955A349D0CFD70C17AEA2003CB3E1F4586A0E59A7EB25AC7BEFDEE7E`
for `simple_reference_v3.py`; the shared `simple_env.py` is
`360C9BAEC80D610F8BBB38F08926CBE462D74A1DFA67B5D64F1A636C3D8A066B`.
The qualification-only all-task smoke produced 16 finite candidate rows on
seed 64999991; an earlier one-task smoke with the same non-scientific seed is
retained separately as implementation provenance. Both smoke outputs are
excluded from scientific metrics. The final source passed 39 ICML-2027/V3
targeted tests before this freeze.
The separate qualification-only smoke seed is 64999991; smoke outputs cannot
tune any scientific constant or gate. A failed scientific payload or mandatory
gate stops this design without retries, task removal, or automatic GPU launch.
Preserve all evidence. A passed replay qualifies only the stated neural-response
interface; a true policy-return guarantee remains a separate theorem obligation.

Risk rankings whose entire spread is at most 1e-5 times max(1, base MSE)
are numerical ties; all candidate choices receive zero normalized regret in
those cases, and their count is reported. This rule is frozen before scientific
development and prevents rounding differences on identical shared data from
being interpreted as an identifiable action advantage.

Run with one CPU thread and a 30-minute hard time cap; inspect progress and
process health rather than launching overlapping copies. A replay uses the
same source/plan and a separate output directory. Runtime seconds are excluded
from byte comparisons. Every numeric gate is frozen in the adjacent JSON.
