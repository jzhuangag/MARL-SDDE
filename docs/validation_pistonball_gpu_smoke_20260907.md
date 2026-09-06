## Material Passport

- Origin Skills: hpc4; academic-research-suite
- Origin Mode: GPU systems validation
- Origin Date: 2026-09-07
- Verification Status: VERIFIED SYSTEMS SMOKE; NO LEARNING CLAIM
- Version Label: pistonball_gpu_smoke_validation_v1

# Pistonball policy-freshness GPU smoke validation

## Decision

The 20-agent CUDA systems smoke passed every predeclared plumbing condition.
It authorizes the separately documented GPU development matrix, not a pilot or
formal experiment.  The two-point return curve is unchanged from
initialization and is not used as evidence.

## Provenance and resources

- source commit: `51bb8091aeed47ce8469b18e23e3b33d3cc09f75`;
- Slurm job: `1825368`, account `vincentlau`, `gpu-a30`, one NVIDIA A30,
  8 CPUs, 32 GB requested, 30-minute limit;
- state: `COMPLETED`, exit `0:0`, elapsed 49 seconds;
- batch maximum RSS: 1,172,196 KiB;
- Python environment:
  `/scratch/jzhuangag/causal-policy-freshness-icml2027/envs/pistonball-py310`;
- code:
  `/scratch/jzhuangag/causal-policy-freshness-icml2027/code`;
- artifact:
  `/scratch/jzhuangag/causal-policy-freshness-icml2027/artifacts/gpu-smoke-51bb809/signed.json`;
- stdout/stderr:
  `/scratch/jzhuangag/causal-policy-freshness-icml2027/logs/gpu-smoke-51bb809/1825368.{out,err}`.

The initial submission attempt produced no job because the cluster required an
explicit PI account.  The queue was checked before resubmission, so there is no
duplicate smoke job.

## Results

| Invariant | Result |
|---|---:|
| CUDA visible | NVIDIA A30, PyTorch 2.6.0+cu124 |
| distinct-agent transitions | 5,120 |
| delayed owner packets received | 57 |
| packets after terminal drain | 0 |
| allowed refresh units | 32 |
| used refresh units | 13 |
| optional policy bytes | 1,163,188 |
| finite parameters and returns | pass |
| prefix budget | pass |
| training-loop runtime field | 23.726 s |
| Slurm exit | `0:0` |

The action trace SHA-256 is
`D1AFA4E3F56056C2B36B27EC4311F277ECFD24556644E7EFC42514E2DC2FBE5F`.
The controller therefore neither collapsed to the null action nor exhausted
the available budget during this systems run.

The initial and terminal evaluation return are both `-14.630681818181783`.
With only 64 launches, this is an expected no-learning smoke result and is not
a gate.

## Integrity

```text
signed.json
  d4c0ca4f8ad9dada63713c1013330a188103b1b3c9edd133791610da82cbc112
environment-freeze.txt
  102641378fbb03e0f22ba102face588b2d1a82df46605facc01652a30da6d9c0
stdout
  375cbf852705baf50496c6a59a6c82984d7ef2c80f96ef2b864c11c333a5e003
stderr (empty)
  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

At validation time, `/scratch` was 57% used with 218 GB available.  The new
root used about 293 MB for the clone and 344 MB for its isolated environment;
artifacts and logs were below 1 MB.  Nothing was written to `/project` or
cleaned.  Existing L20 jobs 1825300 and 1825301 were not modified.

## Next admissible action

Run the explicitly non-confirmatory development matrix on seed 79001.  Its
results may select a pilot design but may never enter confirmatory statistics.
If learning does not rise above initialization, stop and debug the learner
before spending new pilot seeds.
