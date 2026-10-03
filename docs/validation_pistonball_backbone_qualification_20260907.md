## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: experiment validation
- Origin Date: 2026-09-07
- Verification Status: VALIDATED DEVELOPMENT RESULT; NOT PILOT OR FORMAL EVIDENCE
- Version Label: pistonball_backbone_qualification_v1

# Pistonball backbone learning qualification

## Decision

The frozen backbone-learning gate passes.  The deterministic distinct-actor
centralized-critic implementation is retained for graph-controller
development.  This result does not establish a communication advantage and
does not authorize an efficacy claim: the fully fresh runs are deliberately
unconstrained learner diagnostics, the five configurations share one
development-only seed, and the no-refresh diagnostic does not match the best
fresh run's optimizer settings.

## Frozen design and provenance

- Source commit: `93a16b28e66101bd61fbaefe6d6f7369b132ecd0`.
- Slurm array: `1825626_0` through `1825626_4`.
- Development-only seed: `79003`, permanently excluded from pilot and formal
  populations.
- Remote code:
  `/scratch/jzhuangag/causal-policy-freshness-icml2027/code`.
- Raw output:
  `/scratch/jzhuangag/causal-policy-freshness-icml2027/artifacts/backbone-qualification-93a16b28e66101bd61fbaefe6d6f7369b132ecd0`.
- Summary SHA-256:
  `b0d077ccf89cbfb01c6f724d9d3a04d2983027b69f0bf30b5c729f946004830a`.
- Environment: isolated scratch venv with Torch 2.6.0+cu124,
  PettingZoo 1.26.1, Gymnasium 1.0.0, Pygame-ce 2.5.8, Pymunk 7.3.0.

Every configuration used 4,096 launches, 20 distinct actors, rollout horizon
four, 327,680 charged actor transitions, randomized Pistonball, deterministic
CUDA, zero packet delay, and identical evaluation seeds.  Four `complete_burst`
runs made every teammate cache fresh before each launch.  The frozen pass rule
required at least one fresh configuration to improve terminal decentralized
return by 2.0 from the common initialization, with finite nonzero gradient and
actor-drift diagnostics.

All five array tasks completed with exit code `0:0`.  Stderr is empty.  Each
run launched and received 4,089 gradient packets and drained the terminal
packet queue.  Slurm peak RSS was 2,578,264--2,960,980 KiB.

## Results

The common initial decentralized return was `-5.9899346271`.

| Code | Actor step | Critic step | Terminal return | Learning delta | Mean packet-gradient norm | Mean actor drift |
|---|---:|---:|---:|---:|---:|---:|
| `fresh_s1` | 0.001 | 0.0003 | -5.224508 | 0.765426 | 0.053475 | 0.006939 |
| `fresh_s3` | 0.003 | 0.0003 | -4.927264 | 1.062671 | 0.053669 | 0.022563 |
| `fresh_s10` | 0.010 | 0.0003 | **-1.163549** | **4.826385** | 0.049155 | 0.064954 |
| `fresh_c1_s3` | 0.003 | 0.0010 | -4.345800 | 1.644134 | 0.095628 | 0.031247 |
| `no_s3` | 0.003 | 0.0010 | -6.746256 | -0.756321 | 0.088011 | 0.030723 |

`fresh_s10` passes the 2.0 gate by 2.826385.  Its evaluation path was
`-5.989935, -7.686428, -6.550465, -4.833526, -1.163549` at 0, 81,920,
163,840, 245,760, and 327,680 actor transitions.  Thus the endpoint improvement
is not a consequence of a different initialization, although the transient
curve is non-monotone.

Every fresh run spent 77,501 refresh units and 6,934,479,476 optional policy
bytes.  That payload is intentionally outside the eventual communication
regime and cannot be used as a favorable method comparison.  The correct
conclusion is only that the learner can exploit a fresh joint-policy context
and that controller experiments are no longer confounded by a non-learning
backbone.

## Integrity and scope audit

- Exact transition and packet accounting: pass.
- Numerical finiteness, deterministic CUDA, and terminal drain: pass.
- Frozen pass rule: pass without alteration.
- Statistical uncertainty: not assessed; this is one development seed.
- Communication efficiency: not assessed; complete freshness is intentionally
  expensive.
- Freshness causality: not established by `fresh_s10` versus `no_s3`, because
  their optimizer settings differ.
- Generalization across seeds, delays, budgets, and tasks: not established.

The next experiment must therefore use the retained `fresh_s10` learner
settings for every scheduler, introduce delayed receipt and binding prefix
budgets, and first measure an equal-resource controller/oracle headroom.  It
must include no-refresh, complete refresh, strong age/mismatch/static
comparators, and signed-score ablations.  Only a separately frozen multi-seed
pilot may supply efficacy evidence.
