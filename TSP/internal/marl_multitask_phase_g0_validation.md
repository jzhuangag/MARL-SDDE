# Multi-task participation-phase G0 validation

## Decision

Outcome-free G0 passes and authorizes only the frozen 32-run Stage-A endpoint
headroom scan.
It does not authorize a controller, `q=2,4` completion, confirmation seeds, or
any manuscript performance claim.

## Execution

- Preregistration commit: `a51118a7f46511719ca59d372366512d5bac56eb`.
- Operational amendment commits: `a5de4eee8f18756f444475b136c372301bcc4d94`
  and `a9a1d62b5af18b84cb2290b5cd2ec2dae0ae5e0f`.
- Initial failed setup: job `1889967`, `FAILED 1:0`, no output artifact.
- Valid array: `1889976_0` and `1889976_1`, both `COMPLETED 0:0` in 16 seconds
  on `gpu04`.
- Output root:
  `/scratch/jzhuangag/MARL-SDDE-TSP-MULTITASK-PHASE-A-001/g0`.

The failed setup used the MaMuJoCo runtime for an MPE import and lacked
`supersuit`.
The amendment split G0 by task and reused the previously validated pinned
runtime for each benchmark family.
No trajectory, return, win rate, gradient, or policy update was produced by
either the failed or valid G0 execution.

## Gate results

Both tasks use clean HARL commit
`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.
All environment observations and states are finite.
Independent resets differ across at least one worker, while shared resets are
identical across all four workers.

Speaker--Listener exposes two agents, observation dimension 11, and centralized
state dimension 14.
MaMuJoCo HalfCheetah `2x3` exposes two agents, observation dimension 19, and
centralized state dimension 17.

For 20,000 Gaussian coupling draws, the maximum marginal mean error is
`0.0201898242`, the maximum marginal variance error is `0.0551656306`, and the
maximum standardized cross-worker innovation mismatch is below `9e-16`.
Thus the public innovation couples exploration without changing the registered
Normal marginals.

Every G0 gate passes for both tasks.

## Local copies and hashes

- `marl_multitask_g0_mpe.json`:
  `904b157fd14510615b38e47478a7e904d7b5daa65691cf730e15342d97eefef5`.
- `marl_multitask_g0_mamujoco.json`:
  `fc6ef08664d4ab9f015660a12040a811cd45b5317746c85d0bd2d7a36f03810f`.
