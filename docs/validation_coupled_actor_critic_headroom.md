# Validation: coupled actor--critic oracle headroom

Date: 2026-09-05.

## Decision

**H1--H10 pass.**  The exact-moment scan establishes sufficient oracle
headroom to continue the performance-bound gate for *Tracking the Moving
Game*.  It does not authorize sampled CPU trajectories, formal seeds, a
standard MARL benchmark, GPU or HPC4.

The distinction matters: `coupled` observes the exact first/second moment
state of a quadratic Gaussian system.  A neural actor--critic does not observe
actor error or centralized-critic tracking error.  This result proves that the
proposed causal coupling has value under a strong analytic ceiling; it does
not prove that the value can be recovered from TD residuals and policy-version
metadata.

## Frozen provenance

- theory-interface commit: `7c69dfe`;
- preregistration commit: `fc8646b`;
- frozen scenario hash:
  `528fbc597015981f8868e2d7aab567003d5b0574fadecff6b462be2c8b746ae0`;
- runner SHA-256:
  `E3ED1F574BAEEB25A0C9E8881CED91D249F49F8A86F7F24382B274868EC8735A`;
- drift module SHA-256:
  `091921C6CA2155F110DF35E7F0D884D85CC9A7F0CCED5F940AE5B04FDD97029E`;
- gate JSON SHA-256:
  `FC7440B9FD257F520FFE9484425DF39507B8D550AE2A89D95D8DA2A27D0A259A`.

No old T-083A, Two Clocks MPE, EXP-017 or other outcome was read by the
runner.  There are no random seeds or sampled trajectories in this scan.

## Results

The 128 primary scenarios and 32 controls all completed with finite values.
The primary aggregate was deliberately compared with two strong privileged
baselines.

| Quantity | Frozen threshold | Result |
|---|---:|---:|
| coupled / per-scenario best-fixed geometric AUC ratio | at most 0.90 | **0.7328231** |
| coupled / online-diagonal geometric AUC ratio | at most 0.95 | **0.9481180** |
| primary scenarios better than best-fixed | at least 70% | **128/128** |
| primary scenarios better than online-diagonal | at least 60% | **128/128** |
| jointly interior action fraction | at least 5% | **0.938599** |
| low-target median gain over diagonal | descriptive | **0.0194884** |
| high-target median gain over diagonal | strictly larger | **0.0681235** |
| zero-target maximum reduction error | at most `1e-10` | **0** |

The coupled/best-fixed AUC-ratio range was `0.45557--0.97746`; the
coupled/diagonal range was `0.80985--0.99666`.  Thus directionality is not an
aggregate artifact, although several online-diagonal differences are small.

Terminal risk needs a more conservative interpretation.  Its geometric ratio
was `0.20639` against best-fixed but `0.98264` against online-diagonal.  Most
of the coupling-specific benefit is earlier learning risk/AUC, not a broad
terminal improvement.  A later paper cannot report only the fixed-pair
comparison and hide this stronger diagonal baseline.

## Gate ledger

| Gate | Result | Evidence |
|---|---:|---|
| H1 population/finite output | pass | 128 primary, 16 zero-target and 16 zero-interaction controls |
| H2 convexity/feasibility | pass | minimum observed QP eigenvalue `0.0162564`; all actions stayed in the frozen box |
| H3 best-fixed AUC headroom | pass | geometric ratio `0.7328231` |
| H4 online-diagonal AUC headroom | pass | geometric ratio `0.9481180` |
| H5 directional breadth | pass | 128/128 against each strong baseline |
| H6 nontrivial joint action | pass | positive mean scales in every primary scenario; 93.86% jointly interior |
| H7 exact reduction | pass | target-motion zero gives zero cross curvature and byte-level equal metrics |
| H8 mechanistic direction | pass | high target-motion median gain exceeds low by `0.0486351` |
| H9 static optimizer integrity | pass | every refined value no worse than its grid start and every pair box-feasible |
| H10 clean reproduction | pass | byte-identical 317,361-byte JSON files |

The continuous optimizer reported all six refinement starts successful in 155
scenarios, five in four scenarios and four in one scenario.  H9 did not
pre-register optimizer-status unanimity; the retained result is always no
worse than the full frozen grid start.  These status differences are disclosed
rather than removed.

## Reproduction

Primary and clean reproduction SHA-256 are both:

`47446BA32CE70EB27C3FFF5CEA855D5EC6436EB86BB2E4AFBBFD22B61F4E37F1`.

The files are byte identical and each has 317,361 bytes.  Observed local CPU
wall time was approximately 25 minutes per run; almost all of it is the
privileged per-scenario fixed-pair optimization.  The deployed event action is
a constant-size two-variable box QP and must not be assigned this offline
audit cost.

## Scientific interpretation and next gate

The result rejects the simplest early kill hypothesis: the actor--critic cross
term is not numerically redundant once critic-target motion is present.  It
also shows a coherent direction: larger target sensitivity creates more value
for the coupled action, while target sensitivity zero reduces exactly to the
online diagonal rule.

The still-open kill hypothesis is more important: a practical algorithm may
be unable to estimate the required state tightly enough without paid sensing
or outcome-selected constants.  Before any sampled run, the same implemented
action must be supported by a conditional Markov-game bound that provides:

1. a critic-error envelope derived from the mandatory critic update stream;
2. an actor-progress lower bound robust to critic and packet bias;
3. off-diagonal in-flight history accounting under a valid filtration;
4. a finite-time potential/stationarity inequality with both selected actions;
5. a stated boundary for function approximation and POMDP/history critics.

The recent 2026 finite-time asynchronous MARL actor--critic and two-timescale
Markov-SA preprints make this gate essential for novelty.  A generic
two-timescale convergence proof or adaptive learning-rate heuristic is not a
publishable successor.

