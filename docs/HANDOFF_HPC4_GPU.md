# MARL-SDDE HPC4/GPU handoff

## Material Passport

- Origin: local CPU theory and experiment program
- Handoff date: 2026-07-31
- Verification status: locally tested; remote GPU stage not yet executed
- Required source ancestor:
  `c01b900f22b404fae3e6d5c0aac26cdb5e55b480` or the abbreviated
  `c01b900`
- Local source of truth: `E:\HKUST-study\vin\SDDE`
- Git branch: `codex/joint-ms-exp007c`
- Git remote: `https://github.com/jzhuangag/MARL-SDDE.git`
- Intended remote active root: `/scratch/jzhuangag/MARL-SDDE`
- Intended durable archive root:
  `/project/vincentlau/jzhuangag/MARL-SDDE`

## Critical source-control warning

At handoff creation, the local branch is five commits ahead of the GitHub
branch because connections to `github.com:443` were repeatedly reset.  The
five local commits are:

1. `e04662d` — bounded-kernel latent-correlation certificate;
2. `dffe52f` — nonlinear variance theorem and feasibility smoke;
3. `3bb903f` — frozen EXP-013B preregistration;
4. `33d0b8a` — registration-hash record; and
5. `c01b900` — EXP-013B formal failure, audit, and selected artifacts.

The receiving agent must run

```bash
git merge-base --is-ancestor c01b900 HEAD
```

before interpreting or extending the repository.  If this returns nonzero,
the receiving copy is incomplete.  Stop and request either a successful push
of `codex/joint-ms-exp007c` or a Git bundle/patch containing the five commits.
Do not reconstruct missing code or results from this prose.

## Research objective

Develop a low-complexity convergence framework and algorithm for delayed
multi-agent learning from correlated Markov data.  The current contribution
stack is:

1. exact and minimax correlation-limited speedup
   \(q/[1+(q-1)\rho]\);
2. delayed affine Markov temporal-difference (TD) finite-time analysis under
   predictable decorrelation;
3. low-memory anytime certificates for mixing and observable or latent
   cross-agent dependence;
4. scalar, matrix-free selection of participation \(q\), decorrelation gap
   \(b\), and step size \(\eta\); and
5. an stochastic delay differential equation (SDDE) and
   Lyapunov–Krasovskii interpretation of the delay mechanism.

The recommended working title is:

> **Beyond Linear Speedup: Safe Adaptive Participation under Correlated and
> Delayed Markov Data**

The target is an ICML 2027 theory-and-experiment paper.  IEEE Transactions on
Signal Processing remains a viable fallback if the standard nonlinear
benchmarks do not support a sufficiently broad learning contribution.

## Evidence that is already defensible

### Theory

The central proof file is `docs/proof_program_joint_ms.md`.

- The aggregate covariance and random-Jacobian identities are proved.
- The minimax lower bound gives exact speedup
  \(q/[1+(q-1)\rho]\).
- A finite-time affine Markov-TD result is proved for predictable
  decorrelation and delay.
- Observable-sharing, hidden-collision, and unknown-baseline bounded-kernel
  confidence certificates are proved.
- Theorem 9 proves, for any square-integrable nonlinear stochastic update at a
  fixed parameter,

  \[
  \operatorname{Cov}(\bar G_q)
  =
  \left(\rho+\frac{1-\rho}{q}\right)\Sigma_\theta.
  \]

This theorem transfers the variance-saturation mechanism to neural TD but is
not a global neural-TD convergence theorem.

### Controlled experiments

The CPU program through EXP-012B passes its registered certificate and safety
gates.  EXP-013A establishes nonlinear implementation feasibility.

EXP-013B contains 1,024 preregistered realizable neural-TD runs.  Its correct
formal conclusion is **failure: 3/5 gates passed**:

- all 1,024 runs are finite;
- at \(\rho=0\), the geometric MSE ratio \(q=32/q=1\) is 0.0622 with
  one-sided 99% upper limit 0.1119;
- median resource-oracle participation moves from 32 at \(\rho=0\) to 4 at
  \(\rho=.9\);
- at \(\rho=.9\), fixed \(q=4\) versus \(q=32\) has ratio 0.8319 but 99%
  upper limit 1.2935, so a reliable advantage is not established; and
- the delay-specific high-correlation criterion fails at \(D=0\), with ratio
  0.9087 against the frozen threshold 0.90.

A clearly labeled post-hoc audit finds that the \(q=32/q=1\) ratio deteriorates
from 0.0622 at \(\rho=0\) to 1.1487 at \(\rho=.9\).  The interaction is
18.48-fold with a descriptive 99% lower limit of 8.51.  Thus nonlinear loss of
parallel speedup is strongly supported, while a correlation-only fixed
participation rule is not.

Read these before starting GPU work:

- `docs/theory_program_icml2027.md`;
- `docs/proof_program_joint_ms.md`;
- `docs/experiment_013b_realizable_nonlinear_confirmation.md`;
- `docs/validation_exp013b.md`; and
- `experiments/nonlinear_markov_td/results/realizable_td_confirmation/summary.json`.

## Claim boundaries that must be preserved

Do not claim any of the following:

- that EXP-013B passed;
- that correlation alone reliably determines the best neural participation;
- that Theorem 9 proves global nonlinear TD convergence;
- that the SDDE is already accompanied by a closed discrete-approximation
  theorem; or
- that synthetic realizable TD is sufficient ICML evidence.

The next algorithm must be **state- and risk-aware**.  Correlation estimates
the attainable variance reduction, while the current learning state determines
whether the run benefits more from additional server updates or additional
within-update averaging.  Seed-dependent heavy tails make tail control
essential.

## Open theory items

The following are explicitly open:

1. an unthinned joint Markov-mixing theorem;
2. a general finite-state anytime mixing estimator;
3. an unthinned affine Markov-TD finite-time bound; and
4. a rigorous SDDE-to-discrete approximation-error bound.

These are not all required before GPU experimentation.  For the ICML
mainline, retain the discrete affine theorem as the convergence guarantee and
use the SDDE as the Lyapunov–Krasovskii interpretation unless a correct
approximation theorem is completed.  Do not let the GPU stage silently broaden
the mathematical claim.

## HPC4 operating constraints

Every HPC4 action must follow the local `$hpc4` skill on the receiving
computer.

1. Use the configured SSH alias `hpc4`.
2. On Windows, invoke
   `C:\Program Files\OpenSSH-Win64\ssh.exe`, not System32 OpenSSH.
3. Diagnose DNS, TCP port 22, and HKUST VPN before changing keys.
4. Use key-only noninteractive authentication checks; never store or echo a
   password, 2FA code, or private-key body.
5. Before staging or jobs, inspect `quota`, `df`, bounded `du`, `squeue`, and
   `sinfo`.
6. Do not place environments, datasets, caches, checkpoints, or run archives
   in `/home/jzhuangag`.
7. Use `/scratch/jzhuangag/MARL-SDDE` for active reproducible work.
8. Use `/project/vincentlau/jzhuangag/MARL-SDDE` for completed valuable
   artifacts and archives.
9. Direct Hugging Face and dataset caches to `/project`, for example:

   ```bash
   export HF_HOME=/project/vincentlau/jzhuangag/hf_home
   export HF_DATASETS_CACHE=/project/vincentlau/jzhuangag/hf_datasets_cache
   export TORCH_HOME=/project/vincentlau/jzhuangag/torch_cache
   ```

10. Unset `ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` for CUDA jobs.
11. Never use destructive mirroring or broad cleanup.  Verify durable copies
    before any authorized deletion.

## Required GPU-stage sequence

### Phase 0 — provenance and environment

- Verify the required Git ancestor and record `git status`, `git log`, and
  any diff.
- Perform HPC4 read-only preflight.
- Inspect actual GPU partitions and limits with `sinfo`; do not invent a
  partition name.
- Create a project-pinned environment outside home and record Python, PyTorch,
  CUDA, driver, GPU, package lock, and Slurm details.
- Run `python -m pytest -q` before modifying the research code.

### Phase 1 — state/risk-aware controller smoke

Implement an online, predictable controller over a finite candidate set for
\((q,b,\eta)\).  Its risk surrogate must contain:

- a transient/progress term using information available before the next
  block;
- a variance term using the certified effective participation
  \(q_{\mathrm{eff}}=q/[1+(q-1)\rho^+]\);
- a delay/stability constraint;
- message or wall-clock cost; and
- an upper-tail or uncertainty penalty.

Do not use inverse Hessians, covariance matrices, or a preconditioner.  The
intended complexity is \(O(qd)\) arithmetic and \(O(d)\) memory, matching
gradient aggregation.

Compare at least:

1. all-agent adaptive-step control;
2. fixed small participation;
3. correlation-only participation;
4. delay-only participation;
5. the state/risk-aware controller; and
6. a charged information oracle reported only as an upper benchmark.

Use implementation-only pilot seeds first.  Record full learning trajectories,
participation, effective participation, messages, wall time, gradient-noise
statistics, stability events, mean performance, and a tail metric such as
90% conditional value at risk.  Do not reuse pilot seeds for confirmation.

### Phase 2 — preregistered controlled confirmation

Only after the smoke establishes numerical stability and identifiability:

- write `docs/experiment_014a_*.md`;
- freeze seeds, budgets, controller hyperparameters, baselines, endpoints,
  bootstrap unit, multiplicity handling, and gates in a Git commit;
- record that commit hash before running;
- execute fresh paired seeds through Slurm;
- do not change gates or exclude seeds after observing outcomes; and
- produce `docs/validation_exp014a.md` regardless of pass or failure.

The primary question should be whether state/risk awareness reduces
resource-matched error and high-tail risk relative to both all-agent and
correlation-only controls.  A controller that only wins against a deliberately
weak baseline is insufficient.

### Phase 3 — standard multi-agent Markov benchmarks

The method is not required to be actor–critic.  Prefer a shared neural TD,
Q-learning, or value-evaluation implementation that makes per-agent stochastic
updates and delayed aggregation explicit.  Select maintained environments
available on HPC4 after a dependency smoke test.  Suitable candidate families
include PettingZoo multi-agent particle or SISL tasks; record any substitution
and its reason.

Use at least three tasks or environment regimes if feasible.  The formal
comparison must use identical environment steps, message budgets, evaluation
episodes, and paired seeds.  Report:

- return or task score;
- Bellman/value error when a reference is available;
- wall-clock and environment steps;
- messages/bytes;
- stability or divergence rate;
- selected \(q,b,\eta\);
- estimated versus empirical effective participation; and
- mean plus tail-risk statistics.

Include ablations for state information, correlation certification, mixing
control, delay control, and the tail penalty.  Natural environment correlation
and deliberately injected common-factor correlation must be distinguished;
the latter is a causal stress test, not “natural MARL performance.”

### Phase 4 — validation and archival

- Run deterministic analysis reproduction and compare hashes where applicable.
- For stochastic training, use fresh reruns and compare prespecified
  distributions, not wall-clock equality.
- Run the complete test suite.
- Save commands, Slurm scripts, job IDs, logs, metrics, plots, checkpoints
  needed for verification, and a machine-readable environment lock.
- Copy completed valuable artifacts to
  `/project/vincentlau/jzhuangag/MARL-SDDE`.
- Verify counts or hashes after copying.
- Do not delete scratch material without explicit user authorization.

## Required run ledger

Maintain `docs/hpc4_run_ledger.md` with one row per job:

| Field | Required content |
|---|---|
| Timestamp | Asia/Shanghai and cluster time |
| Local/source commit | full SHA and dirty-diff status |
| Remote worktree | absolute path |
| Environment | name/path and lockfile |
| Command | exact `sbatch` and program arguments |
| Job | Slurm job ID, partition, GPU, CPU, memory, time limit |
| Logs | stdout/stderr absolute paths |
| Outputs | metrics/checkpoints/figures absolute paths |
| Status | queued/running/completed/failed/cancelled |
| Validation | exit code, tests, row counts, finite checks, hashes |
| Archive | durable destination and verification |

## Stop and escalation conditions

Stop and report rather than improvising when:

- `c01b900` is absent;
- HPC4 TCP/key authentication fails after VPN-aware diagnosis;
- required storage is unavailable;
- the desired environment would require changing a project-pinned shared
  stack;
- a job crashes or produces non-finite primary metrics;
- a formal gate fails;
- a benchmark changes the meaning of “participating agent” so that it no
  longer matches the theorem; or
- a new theoretical claim would be required to interpret the result.

Queued Slurm state is not a failure.  Check `squeue`, `sacct`, and bounded log
tails before resubmitting.  Never submit duplicate jobs merely because output
is temporarily unchanged.

## Completion definition for the receiving agent

The handoff is complete only when the agent returns:

1. provenance and HPC4 preflight records;
2. the state/risk-aware method and tests;
3. a preregistered controlled GPU confirmation with an honest pass/fail
   validation;
4. at least a standard-benchmark smoke result, or a precise dependency/data
   blocker;
5. the Slurm run ledger and artifact locations;
6. a concise research decision on ICML 2027 viability; and
7. Git commit/push status without claiming an unverified remote update.
