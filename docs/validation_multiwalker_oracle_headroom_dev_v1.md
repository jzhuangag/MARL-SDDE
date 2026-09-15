## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: development-experiment validation and failure audit
- Origin Date: 2026-09-08
- Verification Status: DEVELOPMENT FAILURE RETAINED; CRITIC FITTING NOT AUTHORIZED
- Version Label: multiwalker_oracle_headroom_dev_v1_validation

# Multiwalker oracle-headroom development validation

## Frozen source and execution

The development design and runner were committed before outcome generation at
`c9e940a`. The local CPU command was

```text
D:/anaconda/envs/ust2/python.exe -m experiments.policy_dependency_sync.multiwalker_oracle_headroom_dev --output-dir tmp/policy_dependency_sync/multiwalker_oracle_headroom_dev_v1
```

The run used development seeds `95700--95707`, 352 rows, five walkers, 40
common-state launches, an eight-step counterfactual horizon, motion scales
`0.01/0.04`, budget rates `0.25/0.50`, and the frozen eleven-policy family.
It completed on one local CPU process in approximately 71 minutes. No GPU,
HPC4, or remote storage was used.

Artifact hashes are:

- `rows.json`: `AD79327D849C820C1062C88BB1502A2BB5756064D7EA23EBCFFE6203662B2D42`;
- `summary.json`: `70A76EB50725753541853B439AC8DF1EC308280DEE7B18D9ABB72B1048A1C0A8`.

## Frozen decision

The result is a mandatory **6/7 development failure**.

| Gate | Result |
|---|---:|
| D1 complete and finite | pass |
| D2 exact repeated prefix replay | pass |
| D3 prefix budget | pass |
| D4 active oracle gain positive | pass (`3.889033`) |
| D5 active recovery gap at least 10% | pass (`29.8365%`) |
| D6 active positive direction at least 75% | **fail (`11/16=68.75%`)** |
| D7 median normalized absolute effect at least 0.1% | pass (`0.5520%`) |

No rounding, threshold change, seed replacement, controller fitting, or formal
authorization is permitted from this result.

## Phase localization

The failure is not uniform. In the declared active motion phase:

- budget `0.25`: oracle beats the per-cell strong envelope in only `3/8`
  cells and has total headroom `-0.330368`;
- budget `0.50`: oracle beats it in `8/8` cells and has total headroom
  `1.490721`, with median normalized absolute headroom `1.2049%`.

At budget `0.25`, the strong winner is random in three cells, parameter gap in
three, and age in two. At budget `0.50`, it is age in four, the privileged
one-step oracle in two, random in one, and parameter gap in one.

The weak-motion controls show the same budget direction: `5/8` positive at
`0.25` and `8/8` at `0.50`. This is descriptive development evidence only.

## Methodological defect exposed by the failure

`oracle_h8` was prospectively described as a privileged oracle, but the code
performs a receding-horizon greedy choice: it maximizes the current eight-step
profile value among available one-edge actions. Under a prefix communication
budget, that action can consume a token needed by a later launch and changes
the recipient cache state. It therefore need not maximize total profile value
over the 40-launch sequence.

The fact that observable baselines beat it in five active tight-budget cells
is direct evidence that it is not an upper-bound oracle. This does not turn the
failed gate into a pass. It means the experiment cannot answer its intended
kill question: the observed `68.75%` mixes intrinsic benchmark headroom with
the suboptimal intertemporal allocation of the labelled oracle.

## Decision after failure

The original development plan authorizes no critic or efficacy experiment.
One outcome-free Amendment is scientifically warranted before abandoning
Multiwalker: replace only the receding greedy oracle by an exact finite-horizon
budget-aware oracle over the same common states, actor versions, action costs,
budgets, null-or-one-edge candidate set, rewards, and thresholds. It contains
every null-or-one-edge baseline action sequence. The complete-neighbor
baseline remains a deliberately broader comparator and can still beat it; in
that case the main action class has no certified headroom. Original rows and
hashes remain immutable.

If the corrected oracle still fails the unchanged 10% recovery, 75%
direction, or 0.1% effect thresholds, Multiwalker is stopped before critic
fitting. If it passes, independent seeds are still required; the development
outcomes cannot serve as confirmation evidence.
