## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-07
- Verification Status: VERIFIED
- Version Label: pdsg_obs001_validation_v1

# PDSG-OBS-001 validation

## Decision

PDSG-OBS-001 passes O1--O14.  In the registered delayed factored Markov game,
a causal translation-equivariant estimator using only completed ordinary
training packets recovers most of the exact signed graph oracle's terminal
learning headroom and substantially outperforms a 24-policy resource-feasible
envelope.

This closes the exact-model observability gate.  It authorizes outcome-free
standard-Pistonball controller integration and CPU structural/smoke tests.  It
does not establish a neural MARL return result, a general no-harm claim, or an
ICML-ready paper; it does not by itself authorize a standard-task efficacy
claim.

## Provenance and execution

- preregistration commit: `079cab2`;
- manifest SHA-256:
  `32C7148C74CF0EA086A94FD85C3676107B32422AAF9C0819AC0487E187FA95A4`;
- runner SHA-256:
  `A8172C45ECA8536340E8B671E12C156251B9EA4C1F92A0ABC5786D79856F2E31`;
- game source SHA-256:
  `0D9623C90CF16CC8BE50CA5E691C480748C736820440980893325F7362143DAF`;
- local Python: 3.11.13 in the project `.venv` backed by conda `ust2`;
- workers: four local CPU processes;
- primary/reproduction runtime: 658.78 / 629.68 seconds;
- rows: 9,600 core endpoints plus 384 oracle diagnostics;
- no GPU or HPC4 operation.

The population contains 24 cells, 16 active cells, eight exact uncoupled
controls, 16 new seeds `61001--61016`, 160 launches, one online policy, 24
strong baselines, and one diagnostic oracle.

## Primary findings

| Metric | Observation | Frozen requirement | Result |
|---|---:|---:|---:|
| active median terminal improvement | 58.9858% | at least 10% | pass |
| active terminal strict fraction | 16/16 | at least 80% | pass |
| minimum active terminal improvement | 39.7419% | nonnegative | pass |
| active median cumulative-risk improvement | 5.3956% | positive | pass |
| active AUC strict fraction | 16/16 | at least 80% | pass |
| median oracle terminal-headroom recovery | 93.6665% | at least 75% | pass |
| minimum oracle headroom recovery | 87.1139% | descriptive | -- |
| online/oracle complete trace agreement | 6.6406% | descriptive | -- |
| maximum message-budget excess | -0.076172 | at most 1/160 | pass |
| environment charge error | 0 | 0 | pass |
| uncoupled AUC/terminal difference | exactly 0 / 0 | at most 1e-12 | pass |
| minimum active mean graph supports | 14.1875 | at least 5 | pass |

Every registered subgroup retained a positive median terminal improvement:

- coupling 0.6 / 0.9: 44.7598% / 75.6623%;
- maximum extra delay 2 / 5: 58.2991% / 59.3093%;
- message budget 0.5 / 1.0: 56.4213% / 62.9114%;
- switch probability 0.03 / 0.12: 58.9858% / 58.5964%.

The strongest AUC comparator was no refresh in 12 active cells and fixed
offset four in four.  The strongest terminal comparator varied across no
refresh and fixed offsets one, three, and four.  The proposed method was
therefore not compared only with a single weak scheduler.

The online estimate did not converge exactly to the synthetic target: median
and maximum final errors were `0.311506` and `0.738259`.  Only 17 of 256
active seed-cell action traces matched the oracle completely.  Nevertheless,
the online policy recovered at least 87.11% of oracle terminal headroom in
every active cell.  The positive result is therefore consistent with a robust
signed-action margin, rather than target recovery or oracle action copying.

## Gate ledger

| Gate | Result |
|---|---:|
| O1 complete finite endpoints and seed isolation | pass |
| O2 exact trajectory charging | pass |
| O3 message-budget feasibility | pass |
| O4 median terminal effect | pass |
| O5 broad and nonnegative terminal effect | pass |
| O6 broad positive cumulative-risk effect | pass |
| O7 positive registered subgroup effects | pass |
| O8 exact uncoupled controls | pass |
| O9 dynamic graph support | pass |
| O10 complete strong envelope | pass |
| O11 frozen implementation provenance | pass |
| O12 observable oracle-headroom recovery | pass |
| O13 estimator/action diagnostics | pass |
| O14 byte-identical isolated reproduction | pass |

## Reproducibility

Primary and reproduction scientific files are byte-identical:

| Artifact | SHA-256 |
|---|---|
| `endpoints.csv` | `6C7498CC1953748C6D3091ABC72B338509EE9A9E3A9BB08E1BD17A8B258DFE08` |
| `cells.json` | `1F32F5CBF1B5C92CC270A8BCF1FC9D5D68D447298A333E07E6AFB542D2CD7E80` |
| `summary.json` | `B815F1457AB073295159540E08D5AE7D435C2D1C78C82C6F7372D96FAE633595` |

Runtime and process identifiers are stored only in `run_metadata.json` and are
not scientific equality fields.

The first post-result package regression exposed one expected-time assertion:
the frozen static test asserted that result directories did not exist, which
was true at preregistration and necessarily false after authorized execution.
The original test hash remains recorded in the preregistration document.  The post-result
test now checks that both artifact paths remain ignored; no runner, manifest,
seed, gate, result, or scientific hash was changed.  This is a test-lifecycle
erratum, not a rerun or outcome amendment.

## Statistical and methodological audit

1. No p-value is used as an effect size.
2. Terminal risk and cumulative-risk AUC are reported separately.
3. All active cells and all negative controls are retained.
4. Seeds are disjoint from development and oracle confirmation.
5. Common random numbers pair policies but are not counted as extra samples.
6. Outcome-wise baseline selection favors the comparator and is disclosed.
7. The oracle is excluded from the baseline envelope and labeled diagnostic.
8. Exact uncoupled equality is not generalized to arbitrary games.
9. Target-estimation error is not called a return guarantee.
10. The quadratic fixture is not called a standard MARL benchmark.
11. Reproduction is determined by byte equality without rerunning or tuning a
    failed gate.

## What is now closed and what remains

Closed in the exact model:

1. a dynamic signed policy-cache graph has substantial value over strong
   equal-resource static and online schedulers;
2. the value survives random launch-to-receipt delay and binding message
   budgets;
3. a causal past-packet estimator recovers most oracle value without extra
   sensing;
4. the selected graph is genuinely dynamic and the computation is
   `O(Delta)` after score statistics are available.

Still open before an ICML submission:

1. implement the centralized-critic/JVP signed score on a standard MARL task;
2. measure its actual compute and memory overhead;
3. demonstrate a return--communication improvement on Pistonball against a
   matched strong envelope, with dense controls;
4. finish the general Markov packet and score-error connection in the paired
   Lyapunov stationarity theorem;
5. decide whether a controlled-SDDE weak-limit theorem is completed for the
   appendix or omitted from the final claim.
