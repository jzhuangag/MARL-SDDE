## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: experiment validation
- Origin Date: 2026-09-07
- Verification Status: VALIDATED DEVELOPMENT RESULT; NOT PILOT OR FORMAL EVIDENCE
- Version Label: pistonball_gpu_development_deterministic_v1

# Deterministic Pistonball GPU development validation

## Decision

The end-to-end implementation is finite, resource matched, deterministic under
the null-action placebo, and operationally practical.  It does not yet qualify
an efficacy pilot.  The actor--critic backbone improves the terminal return by
only 0.59347 from initialization, every scheduler has the same terminal return,
and the signed-only controller selects no edge.  The next authorized work is a
separate backbone-learning qualification, not additional method seeds.

## Provenance

- Source commit: `79a1986d460c5c042068c2336b9dcb5f37f6c80d`.
- Placebo jobs: `1825574_0` and `1825574_1`.
- Development jobs: `1825576_0` through `1825576_4`.
- Environment: isolated scratch venv with Torch 2.6.0+cu124,
  PettingZoo 1.26.1, Gymnasium 1.0.0, Pygame-ce 2.5.8, and Pymunk 7.3.0.
- Raw root:
  `/scratch/jzhuangag/causal-policy-freshness-icml2027/artifacts/development-matrix-79a1986d460c5c042068c2336b9dcb5f37f6c80d`.
- Summary SHA-256:
  `7b6e641f4904c80fd845b35989b15160ce6bb93392ab4d4efa8daa6a9f1c946b`.

All five development jobs completed with exit code `0:0`.  Every run contains
327,680 charged actor transitions, 4,089 launched and received gradient
packets, an empty terminal packet queue, finite parameters and evaluations,
and at most 2,048 optional refresh units.  Peak process RSS was 2,938--3,330 MB
by Slurm.  Stderr files are empty.

## Deterministic placebo

The two 512-launch placebo runs use the same seed and differ only in whether
the signed score is computed.  Signed-only selected zero refreshes.  Its launch
trace, segment rewards, evaluation rows, cumulative training reward, and trace
SHA-256 are exactly equal to no-refresh:

```text
0F882DBDA75B1308FD428A6F58DC1DC2612A15B2516C23278E245D686BB76B46
```

This closes the GPU execution-path confound that invalidated the preceding
non-deterministic matrix.

## Development outcomes

| Scheduler | Refresh units | Optional bytes | Return AUC | Terminal return | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| signed Lyapunov | 984 | 88,044,384 | -15.943259 | -15.708343 | 696.04 |
| signed only | 0 | 0 | -15.930895 | -15.708343 | 696.07 |
| mismatch | 1,801 | 161,146,276 | -16.005078 | -15.708343 | 638.57 |
| complete burst | 2,048 | 183,246,848 | -15.943259 | -15.708343 | 635.21 |
| no refresh | 0 | 0 | -15.930895 | -15.708343 | 625.17 |

Initial return is `-16.301814` for every method.  Signed Lyapunov therefore
learns by `0.593472`, but it ties every method at the endpoint and trails the
best comparator AUC by `0.012364`.  Its cumulative exploratory training reward
is `-942.816`, compared with `-970.059` for mismatch, `-994.405` for complete
burst, and `-1030.062` for no refresh.  This trajectory-level ordering is a
diagnostic that refresh affects behavior; it is not a decentralized-execution
performance claim.

Signed-only and no-refresh are exactly equal and spend zero optional bytes.
Consequently, the observed signed Lyapunov behavior is driven by cache debt,
not by the claimed signed learning-value term.  The scheduler adds about 11.3%
runtime over no-refresh at this scale.

## Validity and fallacy scan

- Selection and optional payload accounting: pass.
- Equal actor-transition and evaluation grids: pass.
- Packet launch/receipt/drain accounting: pass.
- Numerical finiteness and hard prefix budget: pass.
- Same-seed deterministic null-action placebo: pass.
- Statistical inference: not attempted; one development seed is not a sample
  for an efficacy claim.
- Practical effect: absent on the primary endpoint and negative on AUC.
- Causal interpretation: training-return differences cannot replace final
  current-policy evaluation because behavior policies use different caches.
- Multiple comparisons, optional stopping, and post-hoc gate changes: not
  applicable; no pilot gate or hypothesis test was run.
- Generalization: not established beyond this development configuration.

## Next mandatory gate

Before revising or tuning the graph controller, qualify a strong current-policy
actor--critic backbone on Pistonball.  The qualification must expose policy
drift and update norms, use a fully fresh local-cone diagnostic in addition to
no-refresh, and show material decentralized-execution learning over
initialization.  Only after that backbone succeeds may controller coefficients
or a separately preregistered efficacy pilot be considered.
