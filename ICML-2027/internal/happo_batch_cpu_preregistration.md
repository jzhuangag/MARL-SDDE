# CPU decision experiment: asynchronous policy-block selection

## Material Passport

- Date: 2026-10-14.
- Authorization: user requested autonomous CPU kill validation, followed by theory and multi-environment work only if supported, or a causal redesign if it fails.
- Scope: two standard cooperative MPE tasks, independent actors and a centralized critic, pinned HARL HAPPO actor updates, actual end-to-end training and held-out return curves.
- Status: protocol is outcome-free; source/configuration hashes are locked in a separate preregistration commit before scientific execution.

## Structural correction before running

The receipt-time candidate charges `c(S)` as though rejecting an already computed packet saves its communication and rollout cost. Those costs are sunk. Moreover, choosing fewer policy blocks in one joint game does not remove the other agents' environment transitions.
The tested resource decision is therefore which actor computations to launch and communicate, under a packet-work credit, while every native joint trajectory is fully charged.
This changes the action timing and accounting, rather than changing a performance threshold.
The CPU study first asks whether a useful allocation opportunity exists. It does not presume a valid Lyapunov return certificate.

## Staged decision

Stage A trains seven fixed methods on simple_spread_v3 and simple_reference_v3, all with the same finite native rollout allowance and the same maximum actor-work/message credit.
It compares fresh sequential HAPPO, a banked full-team HAPPO schedule, concurrent fresh cyclic updates, stale cyclic updates, age-priority updates, a causal previous-packet surrogate rule, and an outcome-aware candidate-menu selector.
The selector receives extra counterfactual computation and search rollouts, reported in its ledger. Its result is an optimistic empirical opportunity diagnostic, not a deployable performance result or an exact oracle upper bound.
Independent evaluation episodes report learning curves and never feed the online methods.

Two development seeds select the strongest non-oracle comparator per task. Four fresh holdout seeds then determine the registered gates, retaining a separate comparison with cyclic HAPPO.
The practical-effect threshold is 2% AUC gain per task, with at least three of four seed gains positive and no worse than a 2% mean final-return change.
These are small-screen feasibility criteria, not significance or publication claims.
The fresh HAPPO baseline must also improve by at least 2% from initialization in each task; failure of that criterion indicates insufficient learning signal in this CPU budget and is not evidence of an intrinsic impossibility.
The outcome-aware selector is only run on the four holdout audit seeds, because its development runs do not enter baseline selection or any registered metric.
The chosen menu includes every actor and packet multiplier in the registered set; no task-specific action is removed.

Only a complete Stage-A pass permits Stage B: a separately frozen fresh-seed test of a pre-launch observable score, its post-selection confidence coverage, temporal transfer, and full estimation cost.
Only a complete Stage-B pass can support a new multi-environment performance preregistration.
If Stage A fails, first distinguish negligible allocation value, stale-packet damage, inadequate learning horizon, and noisy oracle search. The decision is limited to the tested implementation and budget.

## Verification

Run the operational smoke with the disjoint qualification seed before scientific execution.
Freeze implementation, JSON, pinned HARL source and MPE runtime hashes before running development and holdout seeds.
Require complete finite per-payload curves, unique trajectory identities, exact native transition counts, actual optimizer-row/message ledgers, the controller/evaluation separation, and final parameter hashes.
Repeat the registered held-out seed for HAPPO, the causal surrogate, and the outcome-aware selector in both tasks; require byte-identical scientific files.
At each oracle evaluation checkpoint, use paired independent episodes to compare the selected update with its pre-update policy; record this diagnostic without feeding it back to selection, to distinguish search noise from positive update value.
Run the complete analyzer twice and require byte-identical scientific outputs.
Keep runtime timings separate from deterministic scientific files.

## Sources and interpretation

HAPPO is the practical clipped optimizer from the pinned [HARL implementation](https://github.com/PKU-MARL/HARL), whose idealized sequential policy-improvement theory appears in [Zhong et al., JMLR 2024](https://jmlr.org/papers/v25/23-0488.html).
The small CPU configuration is recorded in JSON; it is not the upstream full-scale benchmark recipe.
[CoPPO](https://proceedings.neurips.cc/paper_files/paper/2021/hash/de73998802680548b916f1947ffbad76-Abstract.html) already coordinates concurrent policy updates, so passing this screen alone does not establish novelty.
MPE supplies standard cooperative navigation and communication tasks; the preserved PettingZoo 1.24.3 runtime avoids a dependency change.
AI-assisted research and implementation were used. This is an internal feasibility record.
