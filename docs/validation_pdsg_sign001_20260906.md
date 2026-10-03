## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-06
- Verification Status: VERIFIED
- Version Label: pdsg_sign001_validation_v1

# PDSG-SIGN-001 validation

## Decision

PDSG-SIGN-001 passes S1--S12.
The frozen exact signed-drift oracle has substantial, broad, reproducible
actual-learning headroom under binding message budgets and random packet
delays.
This closes the oracle-value death gate and authorizes a separately designed
CPU integration of the observable sparse critic/JVP estimator.

It does not establish an executable neural MARL controller, standard-
benchmark return improvement, or an ICML-ready result.
No formal or GPU run is authorized by this outcome.

## Provenance and execution

- preregistration commit: `7ecdff0`;
- manifest SHA-256:
  `1AC08E5407DCF653670E6C316CD46B8A6AF1D7868F661A4A20D97C5B157D0B6B`;
- runner SHA-256:
  `92AFBE9B2DF1155D3DA7C58453274CB4B5EDCE82387A3E3557FC3F48DBE6A7E8`;
- game source SHA-256:
  `AF0A1765405B2A8C1B7E21DB0166A5B75EA231ABE48ADE5A22765A3ACAF7DC36`;
- local Python: 3.11.13 in the project `.venv` backed by conda `ust2`;
- workers: four local CPU processes;
- primary runtime: 596.756 seconds;
- reproduction runtime: 594.502 seconds.

Both runs produced 9,600 endpoint rows from 24 cells, 25 policies, 16 new
seeds, and 160 launches.
All outputs were finite.

## Findings

| Metric | Frozen population | Observation | Gate | Result |
|---|---|---:|---:|---:|
| terminal-risk median improvement | 16 active cells | 65.9016% | at least 10% | pass |
| terminal-risk strict improvement | 16 active cells | 16/16 | at least 80% | pass |
| minimum terminal improvement | active cells | 40.1305% | nonnegative | pass |
| cumulative-risk AUC median improvement | active cells | 5.8298% | positive | pass |
| cumulative-risk strict improvement | active cells | 16/16 | at least 80% | pass |
| uncoupled terminal/AUC difference | eight controls | exactly 0 / 0 | at most `1e-12` | pass |
| maximum message excess | all cells | -0.064063 | at most `1/160` | pass |
| environment charge error | all endpoints | 0 | 0 | pass |
| minimum active graph-support count | active cells, averaged over seeds | 13 | at least 5 | pass |

Median terminal gains remained positive in every registered subgroup:

- coupling 0.6 / 0.9: 46.4790% / 89.1625%;
- maximum extra delay 2 / 5: 65.4904% / 66.1833%;
- message budget 0.5 / 1.0: 63.4428% / 70.3948%;
- switch probability 0.03 / 0.12: 64.5699% / 69.0651%.

The strongest AUC comparator was no refresh in nine cells, fixed offset zero
in four, and fixed offsets zero-plus-four in three.
The strongest terminal comparator was no refresh in twelve cells and fixed
offset zero in four.
Thus the result is not a comparison against one deliberately weak static
choice.

## Gate ledger

| Gate | Result |
|---|---:|
| S1 completeness, finiteness, seed separation | pass |
| S2 exact environment charging | pass |
| S3 message-budget feasibility | pass |
| S4 active terminal median effect | pass |
| S5 active terminal directionality and no negative cell | pass |
| S6 active cumulative-risk directionality | pass |
| S7 every active subgroup positive | pass |
| S8 uncoupled exact no-harm | pass |
| S9 dynamic graph support | pass |
| S10 complete resource-feasible strong envelope | pass |
| S11 frozen provenance | pass |
| S12 byte-exact reproduction | pass |

## Reproducibility

Primary and isolated reproduction are byte-identical:

- `endpoints.csv`:
  `708D6E20BF397CB700B31EA9A543A4777DA7BFD156BD2EDD0FB3C1E17E866BBC`;
- `cells.json`:
  `1CF7AA94D31281621B4536F6A0A7C7CCCC46EF404DDA9E2F6BB586FA5CE4A355`;
- `summary.json`:
  `D30A020C420E339B4418A3D9E1718AFA9899042F40892141795E445F2D1EBE57`.

The two `run_metadata.json` files differ only in runtime and process metadata
and are not scientific outputs.

## Statistical and methodological fallacy scan

Coverage: 11/11.

1. No p-value is used as an effect-size substitute.
2. Terminal improvement and cumulative-risk AUC are reported separately.
3. All active cells and all controls are retained; no favorable subgroup is
   selected out.
4. The new confirmation seeds are disjoint from development seeds.
5. Common random numbers are used for pairing and are not counted as extra
   independent samples.
6. Per-cell outcome-wise baseline selection favors the comparator and is
   disclosed.
7. Agent permutations are common across policies and remove label artifacts.
8. Zero effect in uncoupled controls is not generalized to dense or weak-
   signal games.
9. The oracle uses true model drift and is not called observable.
10. The quadratic model is not called a standard MARL benchmark.
11. Byte equality, not agreement after rerunning or retuning, determines
    reproducibility.

## Next admissible stage

Integrate the already validated expectation-level sparse signed estimator as
a predictable, past-data critic/JVP score in this asynchronous learning loop.
The implementation must preserve one-edge `O(Delta)` selection, fixed horizon
and receipt cap, full communication/environment accounting, the same strong
baseline family, and a new-seed preregistration.
It must measure oracle-headroom recovery, estimator regret, budget compliance,
terminal risk, and AUC.

Only after that observable CPU stage passes should the standard Pistonball
training experiment and its GPU/HPC4 request be frozen.
