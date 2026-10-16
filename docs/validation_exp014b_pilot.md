# EXP-014B pilot validation: hierarchical conservative neural TD

## Decision

**Honest implementation-only pilot failure.** The stationary pilot did not
pass the frozen progression gate. No shift experiment, formal
preregistration, formal seed, or standard MARL benchmark was started.
Controller tuning stopped after this decision.

The implementation is based on
`b215ff16a9d38a4ed4d2c579b4792d5c8aa38ddc`. Pilot seeds
`20270821`--`20270828` are permanently excluded from confirmation.
EXP-014A remains a pilot failure and its seeds were not reused.

## Implemented controller

At the beginning of each completed-block boundary, the controller computes

```text
N_t(q,b) = min(
  floor(B_msg_remaining / (server_overhead + q)),
  floor(B_env_remaining / b)
).
```

The certificate/stability layer maintains scalar, predictable
time-uniform bounds for Markov persistence and cross-agent collision
correlation, including a cumulative mixing-bias correction. It filters
candidate `(b, eta)` pairs using a delayed-gain and mixing screen. The
finite-horizon layer ranks feasible `(q,b,eta)` using

```text
U_t = E_t^+ exp(-mu_t^- eta N_t / kappa(D,b))
    + eta sigma_t^(2,+) [rho_t^+ + (1-rho_t^+)/q]
    + 2 delta_t^+(b)
    + confidence_penalty
    + tail_penalty.
```

This is an affine-theorem-inspired nonlinear pilot surrogate, not a
nonlinear convergence theorem. The b-step target is
`R_t^(b) + gamma^b V(S_(t+b))`; the simulator constructs a telescoping
`R_t^(b)` so the fixed teacher satisfies the identity exactly.

The no-harm switch defaults to all-agent adaptive-step. It switches only
when the certificate is sufficiently informative, the candidate passes
the stability screen, and its upper risk clears the fixed improvement
margin. The decision function does not accept true `rho`, hidden sharing
masks, teacher parameters, or validation MSE. True `rho` is used only by
the simulator, offline audit, and charged oracle.

The streaming controller uses `O(qd)` arithmetic per update and scalar
certificate state plus a bounded delay queue, `O(Dd)=O(d)` memory for the
fixed tested delays. It forms no Hessian inverse, `d x d` covariance,
preconditioner, or stored sample history.

## Design and execution

- Tasks: fixed teacher seeds `20270901`, `20270902`, `20270903`.
- Cells: `rho in {0,.5,.9}`, delay in `{0,8}`, and
  `(message, environment)` budgets `(16000,1536)` and `(32000,3072)`.
- Policies: all-agent adaptive-step, fixed `q=4`, correlation-only,
  delay-only, EXP-014A v5, hierarchical conservative, and charged oracle.
- Runs: 8 pilot seeds x 3 tasks x 3 rho x 2 delays x 2 budgets x
  7 policies = 2,016 configurations.
- Every block charges 16 probe messages and 8 probe environment steps.
  Across all policies there were 19,799 blocks, 316,784 probe messages,
  and 158,392 probe environment steps. For the hierarchical policy alone:
  2,448 blocks, 39,168 probe messages, and 19,584 probe steps.
- Hierarchical-controller time was 1.4952 s total, or 0.611 ms/block;
  hierarchical training time was 66.686 s and summed configuration wall
  time was 117.803 s. Across all policies, training time was 555.099 s
  and summed configuration wall time was 776.840 s.

CPU validation in the project environment finished with `117 passed,
14 warnings in 23.19 s`. Fourteen EXP-014B-specific tests cover the
required budget, b-step, predictability, certificate, fallback,
stability, memory, determinism, and end-to-end cases.

All eight A30 jobs completed with exit code `0:0`:

| Seed | Job/task | Node | Elapsed |
|---:|---|---|---:|
| 20270821 | `1680467_0` | `gpu06` | 00:01:47 |
| 20270822 | `1680468_1` | `gpu06` | 00:01:48 |
| 20270823 | `1680469_2` | `gpu02` | 00:01:54 |
| 20270824 | `1680470_3` | `gpu02` | 00:01:54 |
| 20270825 | `1680471_4` | `gpu06` | 00:01:45 |
| 20270826 | `1680472_5` | `gpu06` | 00:01:42 |
| 20270827 | `1680478_6` | `gpu02` | 00:01:45 |
| 20270828 | `1680479_7` | `gpu02` | 00:01:46 |

Each task requested partition `gpu-a30`, one A30, 16 CPUs, 96 GB RAM,
and a four-hour limit. The exact execution command was

```bash
/project/vincentlau/jzhuangag/MARL-SDDE/envs/exp014a-py39/bin/python \
  run_hierarchical_controller_pilot.py \
  --device cuda --seed "${SEED}" \
  --output-dir "${RUN_ROOT}/seeds/${SEED}"
```

The initial array submission met `QOSMaxSubmitJobPerUserLimit`. A locked
submitter then issued one `sbatch --array=i-i` command per newly available
slot and recorded every accepted job ID. It neither cancelled existing
jobs nor duplicated a seed.

## Frozen cell results

The hierarchical policy fell back to the all-agent policy for every
recorded block. Therefore every paired MSE and CVaR90 ratio is exactly
one.

| rho | delay | budget | geometric MSE ratio | hierarchical / all-agent CVaR90 | CVaR90 ratio | fallback |
|---:|---:|---|---:|---:|---:|---:|
| 0.0 | 0 | large | 1.000 | 0.100112 / 0.100112 | 1.000 | 1.000 |
| 0.0 | 0 | small | 1.000 | 0.114365 / 0.114365 | 1.000 | 1.000 |
| 0.0 | 8 | large | 1.000 | 0.047789 / 0.047789 | 1.000 | 1.000 |
| 0.0 | 8 | small | 1.000 | 0.070107 / 0.070107 | 1.000 | 1.000 |
| 0.5 | 0 | large | 1.000 | 0.558501 / 0.558501 | 1.000 | 1.000 |
| 0.5 | 0 | small | 1.000 | 0.710377 / 0.710377 | 1.000 | 1.000 |
| 0.5 | 8 | large | 1.000 | 0.515656 / 0.515656 | 1.000 | 1.000 |
| 0.5 | 8 | small | 1.000 | 0.318269 / 0.318269 | 1.000 | 1.000 |
| 0.9 | 0 | large | 1.000 | 0.825698 / 0.825698 | 1.000 | 1.000 |
| 0.9 | 0 | small | 1.000 | 1.530863 / 1.530863 | 1.000 | 1.000 |
| 0.9 | 8 | large | 1.000 | 0.773911 / 0.773911 | 1.000 | 1.000 |
| 0.9 | 8 | small | 1.000 | 0.688884 / 0.688884 | 1.000 | 1.000 |

Aggregate geometric MSE ratio was `1.000`. The three task-level aggregate
ratios were each `1.000`. Joint persistence/correlation certificate
coverage was `1.000` against nominal `0.990`. All endpoints were finite
and respected both budgets.

## Progression gates

| Gate | Result |
|---|---|
| All runs finite and both budgets valid | PASS |
| Certificate coverage at least nominal | PASS |
| Every cell geometric MSE ratio at most 1.05 | PASS |
| Every cell CVaR90 ratio at most 1.05 | PASS |
| At least one `rho=.9,D=8` budget has mean ratio below .70 and CVaR ratio below .80 | **FAIL** |
| Aggregate MSE ratio below .90 | **FAIL** |
| `rho=0` mainly fallback or `q=32` | PASS |
| `rho=.9,D=8` stably selects smaller `q` | **FAIL** |
| No privileged online information | PASS |
| Improvement direction consistent over all three tasks | **FAIL** |

## Structural diagnosis

This failure is attributable to the certificate/no-harm interaction, not
to heterogeneous task directions:

- In all 408 hierarchical `rho=.9,D=8` blocks, `rho_lower=0` and
  `rho_upper=1`.
- The first 336 such blocks returned `insufficient_certificate`; the
  remaining 72 returned `no_high_rho_evidence`.
- The persistence upper bound ranged from `0.91795` to `1.0`
  (mean `0.97755`). Starting from the conservative upper bound, the
  cumulative mixing-bias correction stayed large while fallback used
  `b=1`. Consequently the lower correlation certificate never became
  positive enough to permit the safer large-gap action that would itself
  reduce future mixing bias.
- This is a cold-start identification/control deadlock. The no-harm rule
  behaved safely, but the certificate design could not escape its initial
  conservatism within either frozen budget. Because every task used the
  same fallback path, the data do not support a task-heterogeneity
  explanation.

This diagnosis is post-run analysis only. No gate or controller constant
was changed after observing the pilot.

## Artifacts and provenance

- Scratch:
  `/scratch/jzhuangag/MARL-SDDE/worktrees/exp014b/experiments/nonlinear_markov_td/results/exp014b_pilot_20260731`
- Scratch logs:
  `/scratch/jzhuangag/MARL-SDDE/worktrees/exp014b/logs`
- Durable archive:
  `/project/vincentlau/jzhuangag/MARL-SDDE/artifacts/exp014b-pilot-20260731`
- Durable `SHA256SUMS` hash:
  `c27cc3ba049856c2e5f8d9ac0457e0aa2532ef0e86037b67f48b5ff594430ef9`
- `results/summary.json` hash:
  `bb763bf14e0d3c6397593b771b8f11f7c17d00308ca2e4c73650c16a438af8a1`
- `pytest.txt` hash:
  `fdb96e2ff3bf68603a7c4676b99e5a409a082dd95ac4a77e4bcf7eb869747754`

`sha256sum -c SHA256SUMS` passed after archival. Historical stderr files
have suffix `.er` because the submitted remote copy's CRLF-normalization
command removed one trailing `r`; they are empty and preserved exactly.
The committed Slurm source uses the intended `.err` suffix.

## Formal decision

**Do not enter formal confirmation.** Do not preregister formal seeds,
run the within-run shift, or design the standard MARL benchmark from this
pilot. A future phase would require a new, theoretically justified
certificate acquisition design; it must not be treated as post-hoc
tuning of EXP-014B.
