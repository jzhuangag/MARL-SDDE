# Fully charged sample-split strategic-drift audit

## Decision

**Positive development evidence, with an unresolved statistical guarantee.**
Replacing the oracle directional derivative by an independent half-packet
estimate preserves a large advantage over the pathwise constant certificate
and the fully utilized barrier.  It no longer exactly matches raw async and it
does not provide a high-probability no-harm certificate.  The result supports
standard-benchmark implementation, not a formal algorithm claim.

## Executable observation interface

Every 16-trajectory Markov packet is split before its proposal is formed:

- eight trajectories generate the policy-block direction;
- eight independent trajectories estimate the directional value;
- both halves are charged, and together they cost exactly the same 256 Markov
  transitions as a comparator packet;
- the arriving server computes teammate drift from the packet's stored birth
  snapshot and the current joint policy;
- the scalar update scale is solved in closed form.

Conditionally on the proposal half, the validation inner product is an unbiased
estimate of the birth-policy directional value.  Because the chosen scale is a
nonlinear function of that noisy estimate, this alone is not a simultaneous
confidence guarantee.  The experiment therefore does not use the phrase
"certified no-harm."

## Frozen development grid

- namespace: `sample-split-strategic-drift-development-v1`;
- eight development seeds;
- 16 coupling-by-service cells;
- five policies and 640 fully executed jobs;
- controller parameters inherited unchanged from the oracle audit:
  `V=10`, risk budget `.001`;
- horizon 16, total packet batch 16, maximum time 180, target gap `.3`.

## Results

| Population | Comparator | Time ratio | Charged-work ratio | Final-gap ratio | Faster cells |
|---|---|---:|---:|---:|---:|
| all 16 cells | oracle debt | 1.05437 | 1.05822 | 1.02660 | 0/16 |
| all 16 cells | pathwise constant | **0.72683** | **0.72823** | **0.66755** | 11/16 |
| all 16 cells | raw async | 1.06024 | 1.06382 | 1.18457 | 0/16 |
| all 16 cells | shadow barrier | **0.39910** | **0.39714** | **0.65617** | 14/16 |
| 12 heterogeneous cells | oracle debt | 1.06069 | 1.06520 | 1.28310 | 0/12 |
| 12 heterogeneous cells | pathwise constant | **0.69154** | **0.69305** | **0.77502** | 8/12 |
| 12 heterogeneous cells | raw async | 1.06857 | 1.07272 | 1.52737 | 0/12 |
| 12 heterogeneous cells | shadow barrier | **0.29094** | **0.28993** | **0.68500** | 12/12 |

The observable estimator costs `6.86%` wall-clock relative to raw async on the
heterogeneous population.  Its mean scale ranges from `.4088` to `.5543`, and
its mean rejection rate from `.3085` to `.4491`.  Mean terminal debt is
`13.24--14.93`.  The final-gap loss against raw async is concentrated in
high-service-ratio cells and shows that fixed `V` and budget do not yet deliver
a universal end-of-training advantage.

The correct positive statement is: under fully equal packet work, observable
sample splitting recovers about 31% of the conservative pathwise learner's
time and work while retaining a 71% wall-clock reduction against the barrier
in heterogeneous cells.  The negative statement is equally important: raw
async remains the strongest speed comparator, and the current noisy controller
does not dominate it in terminal accuracy.

## Provenance

Ignored artifact:

`experiments/clocked_async_mpg/results/sample_split_strategic_drift_development_v1/summary.json`

Artifact SHA-256:

`9115e718b34364131b130dc60d59f847a475f52d6504883cb1884787d19e7ab1`

Source SHA-256:

- simulator: `e650a42ca3de80ec6f80f247acefc8dcc9816e6a1e8f7457699119a2e10f1c89`;
- runner: `74d364016daf53ebb7ccc6d69878c135c840828f5043e8bfc0a8eedb9eca9130`.

Command:

```text
.venv/Scripts/python.exe -m experiments.clocked_async_mpg.run_sampled_strategic_drift_development --output experiments/clocked_async_mpg/results/sample_split_strategic_drift_development_v1/summary.json --seeds 8 --workers 4 --namespace sample-split-strategic-drift-development-v1
```

## Consequence for the paper

The main theoretical algorithm remains the pathwise single-flight policy
gradient with a Lyapunov--Krasovskii finite-time guarantee.  The debt-scaled
sample-split rule is the practical online extension.  It may be promoted to a
main theorem only after a predictable-noise analysis bounds the selection term
introduced by the validation estimate.  Otherwise it must be presented as a
principled practical controller with an oracle-headroom audit and full
ablations, not as a high-probability safety theorem.
