# MaMuJoCo controller G0 validation

## Material Passport

- Experiment: `TSP-MARL-MAMUJOCO-DEV-001`
- Stage: outcome-free controller-interface G0
- Source commit: `91fad2e1f6de9e85d8ab9db676a651bab4fee50b`
- Slurm job: `1892331`
- Status: pass

Both A30 array elements completed with exit code `0:0` in 18 seconds.
The smoke test instantiated the pinned HARL MAPPO runner for MaMuJoCo `HalfCheetah-v2/2x3`, collected eight fully charged critic-residual blocks from eight rollout workers, and did not record evaluation return.

The independent-coupling fingerprint has correlation estimate `-0.0393922` and one-sided upper certificate `0.0289907`.
The shared-coupling fingerprint has correlation estimate `1.0` and upper certificate `1.0`.
Both arrays have shape `[8,8]`, contain only finite values, and have strictly positive variance for every worker.
This establishes that the continuous-action coupling preserves nondegenerate worker signals while the return-free certificate distinguishes the two registered dependence regimes.

The raw JSON SHA-256 values are:

- independent: `2a00b997ee4d55ecd918040767fc3b28f9947b191b4f1c0423e829cc6552f690`
- shared: `7c9474dfea81d0982b1573ef88d680dc07ea55632c71e9beea188366c23edf02`

The frozen 40-run development experiment is therefore authorized without changing its task, methods, budgets, controller, seeds, or gates.

