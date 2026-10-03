## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: independent CPU confirmation validation
- Origin Date: 2026-09-08
- Verification Status: ALL FROZEN GATES PASSED; STATISTICAL CRITIC DESIGN AUTHORIZED
- Version Label: multiwalker_oracle_confirmation_v1_validation

# Multiwalker exact dynamic-value independent confirmation

## Frozen provenance

The complete confirmation protocol, seed registry, runner, analyzer, and gates
were committed before outcome generation at `f5cb6c2`. The configuration
SHA-256 is
`AD5284984C99CB08EAD4014D64E5300CE41E867340164AD386B7F5D899D38BCB`.
The eight untouched seeds are `96100--96107`; none appears in the development
or Amendment population.

For each seed, the run regenerated all eleven baseline policies and the exact
null-or-one-edge prefix-budget oracle for both motion scales `0.01/0.04` and
budget rates `0.25/0.50`. Four isolated two-seed chunks were deterministically
merged. The primary run completed from 18:42 to 20:49 Asia/Shanghai; the clean
reproduction completed from 20:50 to 21:38. Both used the local `ust2` CPU
environment. No GPU, HPC4, or remote storage was used.

## Gate result

All nine mandatory confirmation gates pass.

| Gate | Result |
|---|---:|
| C1 exact frozen seed registry | pass |
| C2 352 baseline and 32 exact rows | pass |
| C3 baseline finite/replay/prefix validity | pass |
| C4 exact optimizer/replay/prefix validity | pass |
| C5 active exact-oracle gain positive | pass; `6.878929` |
| C6 active recovery at least 10% | pass; `54.5274%` |
| C7 active direction at least 75% | pass; `16/16=100%` |
| C8 median normalized effect at least 0.1% | pass; `1.2870%` |
| C9 byte-exact clean reproduction | pass |

The active strong-envelope gain over no refresh is `3.128031`; the exact
oracle adds `3.750898` above that envelope. The strong envelope includes the
privileged one-step oracle and complete-neighbor refresh. The original
receding-greedy `oracle_h8` diagnostic fails its full gate set again on these
new seeds; it is not used as the corrected oracle or as a strong comparator.

| Motion scale | Budget rate | Positive cells | Oracle headroom | Median normalized headroom | Oracle gain over no refresh |
|---:|---:|---:|---:|---:|---:|
| `0.01` | `0.25` | `8/8` | `0.391528` | `0.3036%` | `1.329527` |
| `0.01` | `0.50` | `8/8` | `0.541236` | `0.3631%` | `1.520729` |
| `0.04` | `0.25` | `8/8` | `1.625768` | `1.1546%` | `3.133059` |
| `0.04` | `0.50` | `8/8` | `2.125130` | `1.2870%` | `3.745871` |

## Artifact integrity

Primary and reproduction files are byte-identical:

- `baseline_rows.json`:
  `0F523DC6BA9474BE008533E1FAA90E6A8700CEF246970C6FAD9823080F313ECC`;
- `exact_rows.json`:
  `4CD2556B605FBD63659100165FD2A95C39FAE87BE551999B697A2A93046F0FF5`;
- `summary.json`:
  `912AED0AD6F9479BAF931146987CDBA9AB5C1FEEF5271D7E2FE82B2F2B381504`.

Ignored artifacts remain under
`tmp/policy_dependency_sync/multiwalker_oracle_confirmation_v1/{primary,reproduction}`.

## Scientific interpretation and authorization

The independent result closes the benchmark-value kill question: under the
registered resource and action constraints, Multiwalker has a broad,
reproducible intertemporal policy-cache scheduling gap above a strong online
envelope. This is materially different from the stopped Pistonball/Pursuit
paths, where either the local-degree contract or the learnable score bridge
failed.

The result does **not** show that the paper's observable algorithm attains the
gap. The exact oracle sees future branch values and the public evaluation
prefix remains an all-current reference trajectory. Therefore the only newly
authorized work is an outcome-free delayed-feedback critic/certificate
interface, followed by a separately preregistered causal cached-rollout CPU
controller gate. Formal claims, standard CTDE training, GPU, and HPC4 remain
unauthorized until those bridges pass.
