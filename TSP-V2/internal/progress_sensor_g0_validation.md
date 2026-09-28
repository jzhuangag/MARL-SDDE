# TSP-V2 learning-progress sensor G0 validation

## Material Passport

- Experiment: `TSP-V2-MARL-PROGRESS-G0`
- Frozen source: `2eaa373f3283d70bf4475173ab1365a2e617e2cc`
- Slurm array: `1914391`
- Decision: **stop before scientific execution**
- Scientific result: none

## Execution record

All eight array elements terminated at elapsed time `00:00:00` with Slurm exit
code `0:53`; their batch steps were cancelled with the same code.  The remote
source checkout remained present and clean, but the parent `artifacts`, `logs`,
and `tmp` directories had not been created because the preceding clone command
timed out after the repository checkout and before its final directory-creation
step.  The submission command supplied output paths below the absent `logs`
directory and did not independently create that directory.  Slurm therefore
failed before invoking the batch payload.

The post-failure audit found:

- no stdout or stderr file;
- no artifact directory;
- no `progress_sensor_g0.json`;
- no trajectory, reward trace, gradient, model update, or evaluation return;
- no write to `/project` or `/home`.

This is an infrastructure failure, not evidence for or against the
learning-progress sensor.  The preregistered no-retry rule is retained: job
`1914391` is not resubmitted, and no gate, task, seed, or threshold is changed.
A future execution requires a separately identified operational amendment and
new authorization; it must first make output-directory creation an atomic
precondition of submission.

