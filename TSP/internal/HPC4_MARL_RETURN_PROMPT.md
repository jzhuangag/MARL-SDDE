# Prompt for the HPC4-connected Agent

Use the `$hpc4` skill before every remote action.  Work only on the MARL return
extension of `jzhuangag/MARL-SDDE`, branch `codex/joint-ms-exp007c`.  Pull and
verify the exact branch HEAD supplied by the user before running anything.

Read completely:

- `TSP/internal/marl_return_extension.md`
- `TSP/experiments/marl_return_development_grid.json`
- `TSP/experiments/run_mappo_return_bridge.py`
- `TSP/experiments/analyze_mappo_returns.py`
- `slurm/tsp_marl_dev001_a30.sbatch`
- `TSP/internal/marl_coupling_audit.json`
- the repository root `AGENTS.md` instructions, if present.

This is development-only `TSP-MARL-DEV-001`, not a formal experiment.  Do not
change its task, action grid, budgets, seed, evaluation rule, or mandatory
headroom gates.  Do not use any earlier formal result for selection.  Do not
write outputs to `/project`.

1. Audit connectivity, Slurm state, `/scratch/jzhuangag` capacity, the active
   jobs of this user, and the existing compatible Python/HARL environment.
   Stage a clean worktree under
   `/scratch/jzhuangag/MARL-SDDE-TSP-MARL-DEV-001`; keep all code and outputs
   there.  The upstream HARL checkout must be exactly
   `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2` and remain clean.
2. Re-run local unit tests for the bridge, budget identities, cyclic marginal
   preservation, and public-uniform categorical sampling.  Run the reset
   coupling audit and require `pass=true`.
3. Use `slurm/tsp_marl_dev001_a30.sbatch` to create one Slurm array task per
   cell of the frozen 16-cell grid: coupling in
   `{independent,shared}`, `q` in `{1,4,8,16}`, critic learning rate in
   `{0.00025,0.0005}`, seed `93001`.  Use one A30 GPU per job, 8 CPU cores,
   32 GB RAM, and a conservative six-hour limit.  Use the owned runner without
   modifying upstream HARL.  Pass the exact budgets and model/evaluation
   settings from the JSON.  Store stdout, stderr, HARL results, and metadata in
   the scratch worktree.

   Submit index 0 first as a GPU/environment smoke that remains the registered
   D0 cell if it completes, then submit indices 1--15 without rerunning index 0.
   Export the exact clean code HEAD as `TSP_EXPECTED_COMMIT`.  The scientific
   payload encoded in the Slurm file is:

   ```bash
   python TSP/experiments/run_mappo_return_bridge.py \
     --harl-root /scratch/jzhuangag/MARL-SDDE-TSP-MARL-DEV-001/tmp/HARL \
     --results-root /scratch/jzhuangag/MARL-SDDE-TSP-MARL-DEV-001/artifacts \
     --exp-name d0 \
     --coupling ${COUPLING} --q ${Q} --critic-lr ${CRITIC_LR} \
     --message-budget 5000000 --environment-budget 1000000 \
     --server-overhead 100 --rollout-length 25 \
     --seed 93001 --seed-registry-base 93001 --seed-registry-size 17 \
     --eval-interval 1000 --eval-episodes 32 --eval-threads 4 \
     --hidden-size 64 --activation tanh --ppo-epoch 2 --critic-epoch 2 \
     --cuda
   ```
4. Monitor to terminal state.  For failures, diagnose and retry only
   infrastructure failures with the identical scientific configuration.
   Never alter the action grid, budget, seed, coupling, or gates to obtain a
   positive result.
5. Verify every `tsp_bridge_metadata.json`: exact budget identities, expected
   upstream commit, `upstream_modified=false`, finite progress rows, and exit
   code zero.  Record commands, environment versions, job IDs, logs, run dirs,
   elapsed time, and SHA-256 hashes.
6. Run `TSP/experiments/analyze_mappo_returns.py` with the frozen config and
   coupling audit.  Its gate JSON selects the global strong fixed `(q, eta)`
   over both coupling regimes and the best fixed pair separately in each
   regime.  Report the per-regime oracle and its aggregate return-AUC headroom
   over the global strong fixed pair.  Do not use a smoothed curve for
   numerical gates; smoothing is visualization-only.
7. Apply every mandatory D0 gate verbatim.  If any fails, stop before a learned
   controller, new seeds, or a second task.  If all pass, document which fixed
   pairs create the headroom and propose—but do not yet run—the observable
   Lyapunov calibration needed to select them without the coupling label.
8. Commit only code, compact summaries, hashes, and reports.  Do not commit
   checkpoints, TensorBoard event files, or raw large artifacts.  Push the
   verified branch and update Draft PR #7 with the true development outcome.

Return a concise ledger containing the branch HEAD, job table, artifact paths,
hashes, gate table, exact AUC values, strong fixed pair, per-regime oracle
pairs, and the authorized/stopped next step.  Do not claim that D0 is formal
evidence and do not add a MARL return figure to `TSP/main.tex` unless a later
independent confirmatory preregistration passes.
