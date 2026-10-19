# Independent V3 actual-risk gate validation

This package executes the human-approved **separate8CPU-hour** study. It is not
the140h MARL study and it does not confer manuscript admission. Original V3,
revised manuscript, historical results, gates and native140h source are untouched.

## Immutable scientific extent

`protocol.json` is the approved science and is unchanged. There are16 public
families,4budgets,2hypotheses,2learners,6policies and128new seeds:196,608 actual
policy trajectories, followed by **all196,608** semantic replays and two separate
analyses. The six policies include the corrected gate, its no-carryover control,
information-only switching, all-agent fallback, fixed sampling, and the explicitly
evaluation-only known-hypothesis oracle. Every cell, negative result, infeasibility,
cost, partial file and failure is retained. No shrinkage or scientific retry.

The world uses stationary Gaussian common noise and prefix-coupled private noise.
Full SHA256 words, not a truncated32-bit subkey, initialize the deterministic
random stream for each seed/family/hypothesis/learner. Potential observations are
shared over budgets and policies. Public fixed-program selection never receives
the true hypothesis or observed risk. Only the environment and labeled evaluation
oracle use that truth. Both direct learners apply the paid FIFO queue; exact
weights are independently tested against its recurrence.

## Preparation, registration and budget

`prepare.py --test` records an immutable report for **each** deterministic toy-test
attempt and its exact Windows child-process CPU. These fixtures do not generate
formal observations or use formal experiment seeds. `prepare.py --freeze` requires
a current successful test snapshot and pushed source, screens historical seed
metadata without reading outcomes, then creates immutable `lock.json` and
`admission.json`. Those registration files must also be committed and pushed before
data; the launch guard verifies their bytes against Git and the runtime fingerprint.

Total cap is28,800CPU seconds. All toy-test attempts, worker imports/science/replay/
analysis, supervisor CPU, exact child CPU, partial/failed work and audit are inside
it. A conservative60CPU-second registration/auxiliary charge is made upfront for
preparation-parent work, metadata-listing subprocesses, registration/preflight and
later agent read-back/plot checks. This is **charged allowance**, not a claim of
measured native time; measured and charged figures are kept distinct. Preparation
parent snapshots are retained. Avoid repeatedly running readiness checks. The
supervisor additionally measures Git-verification child CPU and the entire worker
CPU; it cannot silently obtain another8h by starting a fresh process.

After the audited handoff only, the first complete seed is retained as the formal
science prefix and its cost is profiled. It is not a second pilot or an extra set
of trajectories. Prefix plus setup must fit900CPU seconds. A cost-only projection
uses1.25 times its cost for the remaining255 seed parts plus2700s analysis/audit
reserve. Projected overrun stops with that prefix preserved: no cheaper suite,
return inspection, tuning or restart. The full28,800s ceiling remains decisive.

## Automatic audited handoff — no additional confirmation

The existing hourly heartbeat, not another watcher, owns orchestration. It first
finishes the140h workflow and **independent final audit**, retaining its scientific
or resource decision. A genuine audit receipt is then saved at:

`ICML-2027/internal/sealed_bridge_learning_140h_audited_terminal.json`

Receipt schema (create only from actual completed audit, never as a stub):

- `audit_complete`, `all_preserved_evidence_reviewed`,
  `terminal_decision_recorded`: alltrue;
- `terminal_decision`: the actual nonempty scientific/resource/integrity decision;
- `native_root`: absolute140h output-root path;
- `driver_pid`: actual native owner PID, which must have exited;
- `terminal_marker`: `workflow_complete.json` or `failure.json`;
- `terminal_marker_sha256`: exact digest of that preserved marker;
- `audit_report`: workspace-relative final independent audit report path;
- `audit_report_sha256`: its exact digest.

A failure is audited, not retried; the receipt does not turn failure into success.
Missing receipt, live predecessor, missing marker/report or changed digest blocks
launch before any new scientific output root is created. The new supervisor also
uses the same cross-process native-owner mutex as140h. The sole native worker
requires its admitted parent. A preexisting output root blocks overwrite/resume.

Once the verified140h audit receipt exists, the authorized command is:

```powershell
& D:/anaconda/envs/ust2/python.exe E:/HKUST-study/vin/SDDE/experiments/v3_risk_gate8h/launch.py --run
```

For asynchronous launch by the existing heartbeat, use `Start-Process` with
`-WindowStyle Hidden`, this exact Python/script, `--run`, the package working
directory, and separate outer stdout/stderr logs. Inspect processes before launching
and never duplicate it. All native scientific artifacts are under the one fixed
`results/formal_20261019` root. No worker is started during preparation.

## Terminal audit and reporting

Science and replay each have128 seed parts of1,536rows. All payload hashes must
match, all196,608 science rows must have finite outputs and exact paid ledgers,
and all seven analysis artifacts—not merely the minimum three—must be byte equal.
All1,536 cell and96 macro comparisons are retained; bootstrap units are the128
independent paired seeds with families macro-averaged within seed first. Probe
directional errors are deduplicated over shared budget/policy probes. Zero-slack
stress reuses saved branches and adds no scientific trajectories. Model upper
bounds versus sample means are descriptive calibration evidence, not finite-sample
guarantees or a new empirical confidence theorem.

The supervisor performs a mechanical read-back **after worker exit**, saves exact
child CPU and a terminal receipt, and marks visual plot review pending. The agent
then inspects all four plots and records its independent scientific assessment;
an automated marker alone is not a TSP-readiness decision. Report numeric stage
counts, live used/remaining separate CPU budget, failures and materially changed
ETA; preserve positive, negative and inconclusive findings equally.
