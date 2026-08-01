# T-019 absorbing-state and fixed-q phase-diagram audit

## Status and scope

T-019 is a read-only, descriptive post-pilot audit of the frozen EXP-017A
negative result. It submitted no Slurm or GPU job, generated no new scientific
trajectory, and changed no EXP-017A runner, seed, gate, result, or artifact.
The only outcome input was the existing combined endpoint table:

- source: `/scratch/jzhuangag/exp017a-pilot-17a4c32/endpoints.csv`;
- SHA-256: `bc241c772d20b76c5f42f72bd8a5523bda2ba225e113811e695dd840007191f0`;
- full input population: 1,584 endpoints;
- audit population: 864 endpoints from the six requested arms and both pilot
  seeds;
- evidence status: **descriptive post-pilot audit, not formal evidence**.

The requested `academic-research-suite` skill was not installed in this Codex
workspace. The mechanism audit is source-code and artifact based. Scientific
boundaries continue to use the primary sources already recorded in
`exp017a_nonlinear_audit.md`, including DASA (arXiv:2403.17247), AsyncMATD
(arXiv:2407.20441), neural TD (arXiv:2312.05397), and official Gymnasium
documentation. No secondary-source claim was added.

## The q=1 absorbing-state theorem

### Proposition

Consider an EXP-017A uncertainty-driven controller call whose state has
`collision_trials=0`. For every registered task, mixing profile, delay trace,
budget regime, and every remaining-budget state, the frozen score selector
chooses `q=1` whenever it chooses a feasible action. After that action it adds
zero pairwise collision trials. Consequently `collision_trials=0` and `q=1`
form an absorbing state by induction.

This proposition applies to `learning_aware`, `information_only`,
`no_delay_ablation`, and `mixing_blind_ablation`, which retain the
`rho_upper` planning input. It does not claim that fixed-q policies, the
true-rho oracle, or `correlation_blind_ablation` use the same branch.

### Proof

When `collision_trials=0`, `correlation_bounds` returns

\[
(\rho_L,\rho_U,\widehat\rho)=(0,1,1).
\]

The affected controllers set their planning correlation to
`rho_upper=1`. For action `(q,b)`, parameter count `p`, remaining message
budget `M`, and remaining environment budget `E`, define

\[
H(q,b)=\max\!\left\{0,\min\!\left(
\left\lfloor\frac{M}{c_0+4pq}\right\rfloor,
\left\lfloor\frac{E}{b}\right\rfloor\right)\right\}.
\]

For fixed `b`, `H(q,b)` is nonincreasing in `q`. If `(q,b)` is feasible,
then `(1,b)` is feasible and `H(1,b) >= H(q,b)`. The frozen score is

\[
S(q,b)=L\exp\!\left[-\frac{aH(q,b)}{1+d_{90}/b}\right]
+0.2\eta G\left(\rho_U+\frac{1-\rho_U}{q}\right)
+0.1\lambda^b+\frac{0.1}{\sqrt{H(q,b)}}.
\]

Here `L>0`, `G>0`, and the implementation clamps `a` to a positive value.
At `rho_U=1`, the variance factor is exactly one for every `q`. The mixing
term is also independent of `q`. Because both the transient term and horizon
penalty weakly decrease as `H` increases,

\[
S(1,b)\leq S(q,b)\quad\text{for every registered }q,b.
\]

Thus every candidate with `q>1` is weakly dominated by the candidate with the
same `b` and `q=1`. Exact ties are resolved by `(score,q,b)`, so the smaller
`q` is selected. If no action is feasible, all scores are infinite, the same
tie-break returns `(1,1)`, and the runner exits before collecting evidence.

For `q=1`, `_collision_observations` receives fewer than two states and returns
`(collisions,trials)=(0,0)`. The update
`collision_trials += trials` therefore leaves the count at zero. Reapplying
the argument proves the absorbing state by induction over controller blocks.

The proof uses only positivity of the model size and the resource costs. Task
dimension changes `p`; mixing changes only `lambda`; delay changes only
`d90`; and the two budget regimes change `M,E`. None reverses the monotonicity.
The CPU grid audit confirms `q=1` selection and same-`b` weak domination in
all 24 registered task × mixing × delay × budget combinations.

## Fixed-q phase diagram

The fixed envelope treats `single_agent` as q=1 and compares it with
`fixed_q4`, `fixed_q16`, and `fixed_q32`. `always_all` is retained as a
registered duplicate check and exactly matches `fixed_q32` terminal errors.
The oracle is reported but is not eligible to define the fixed envelope.

For every one of the 72 task × mixing × rho × delay × budget cells,
`t019_fixed_q_phase_diagram.csv` reports each requested arm's geometric
terminal prediction error, ratio to the cellwise best fixed arm, two-seed
CVaR90, mean message use, mean environment use, agent transitions, and wall
time. With only two seeds, CVaR90 is exactly the larger seed error and must not
be read as a stable tail estimate. `t019_fixed_q_best_cells.csv` is the
72-row best-arm projection.

### Best-q distribution

| Slice | q=1 | q=4 | q=16 | q=32 |
|---|---:|---:|---:|---:|
| all 72 cells | 23 | 20 | 11 | 18 |
| message-binding | 22 | 14 | 0 | 0 |
| environment-binding | 1 | 6 | 11 | 18 |
| rho=0.0 | 4 | 11 | 3 | 6 |
| rho=0.5 | 9 | 5 | 4 | 6 |
| rho=0.9 | 10 | 4 | 4 | 6 |
| zero delay | 8 | 7 | 3 | 6 |
| edge jitter | 7 | 8 | 3 | 6 |
| WAN bursty | 8 | 5 | 5 | 6 |

The strongest transition is resource-driven: in all 36 matched pairs the
environment-binding optimum is at least the message-binding optimum, and it
is strictly larger in 34/36. Higher correlation gives a weakly nonincreasing
best-q path in 22/24 matched paths. Increasing delay severity gives a weakly
nonincreasing path in 21/24. These directions agree descriptively with lower
participation value under stronger dependence/delay and higher participation
value when environment rather than communication is binding.

The direction is systematic, but the magnitude is modest. The global best
fixed policy is q=1 with geometric error `33.66919`; the cellwise fixed
envelope is `33.11457`, a ratio of `0.983527` or 1.6473% descriptive
improvement. The phase diagram therefore shows adaptation value sufficient to
motivate a new independently preregistered pilot design, not a positive or
formal adaptation claim.

## Three distinct failure mechanisms

1. **Information starvation / absorbing state.** The theorem above is a
   deterministic implementation failure. With no initial pairwise evidence,
   the upper confidence bound forces q=1; q=1 cannot create pairwise evidence.
   This alone explains the observed median q=1 at both low and high rho.

2. **Controller-surrogate misspecification.** Removing information starvation
   is not sufficient. The registered oracle receives true rho but uses the
   same surrogate. It beats the cellwise best fixed envelope in only 5/72
   cells and has an aggregate geometric error ratio `5.0292` relative to that
   envelope. Thus the surrogate's horizon/variance/mixing trade-off is badly
   calibrated even when correlation is known.

3. **Python/GPU controller overhead.** EXP-017A recorded a controller wall
   fraction of `0.50137`, failing the 10% gate. Static inspection finds that
   each decision rebuilds and scores the 12 `(q,b)` candidates in Python and,
   more importantly, recomputes the 131,072-value delay summary and sort via
   `trace_summary` inside controller calls. This is host-side orchestration
   overhead, not useful A30 computation. Candidate features and delay
   summaries must be cached/vectorized before another GPU pilot.

## Reproducibility and decision

The copied endpoint input matched the HPC4 SHA-256 before analysis. The T-019
CPU suite passed 14/14 tests: frozen-runner hash/control-flow identity, q=1
domination, all-cell universality,
arbitrary positive learning summaries, phase population/accounting, two-seed
CVaR semantics, q32 duplicate equivalence, independently identifying probes,
charged probe resources, non-q1 fallback, and cached vectorized features.

Generated artifact SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `t019_fixed_q_best_cells.csv` | `a59685b13842bbbf313f90f7cdd00e2faa8ef3d7f39c5f9eb08b5222e5115df8` |
| `t019_fixed_q_phase_diagram.csv` | `7f1afc30ba36552a42e809b56fe3d82240b211e3716b8511df2d378bb8c7969d` |
| `t019_fixed_q_phase_summary.json` | `15048626b222c5173d314708b28f1f7cd26aa0571019603dbe9fda2fc00cb451` |

Code/test hashes and the read-only HPC4 provenance are recorded in
`t019_reproduction_audit.json`.

The two conditions for designing a future pilot are met descriptively: the
absorbing state is excluded by a static independently charged probe design and
CPU tests, and the fixed-q diagram has systematic adaptation structure.
Accordingly an EXP-017B protocol may be designed. GPU execution remains
unauthorized until a separate outcome-free preregistration commit freezes the
complete runner, analyzer, new seeds, and gates. No EXP-017B seed or job was
created in T-019.
