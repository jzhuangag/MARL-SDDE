## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-frozen CPU development design
- Origin Date: 2026-09-09
- Verification Status: PREREGISTERED BEFORE NEW POTENTIAL-POLICY OUTCOMES
- Version Label: multiwalker_paired_potential_dev_v1

# MW-PF-DEV-001: composite cache-potential bridge

## Question and limits

This is a development-only kill gate for the post-alignment paired-freshness
reformulation. It asks whether the exact reset term in

\[
H_t=\frac12\sum_e\beta_e\|\theta_{j(e),t}-\chi_{e,t}\|^2
\]

can recover a nontrivial share of the independently confirmed dynamic
cached-profile return headroom. It does not fit a critic, apply gradient
packets, claim confirmation, or run cache-dependent environment prefixes.

The eight seeds `96100--96107` and their authenticated baseline/exact-oracle
tables were already inspected in the prior confirmation. They are now openly
reclassified as post-outcome **design data**. They may never be used to confirm
the resulting controller. Passing this gate authorizes only a fresh-seed,
cache-dependent rollout preregistration.

## Frozen controller

The public Multiwalker prefix, actor drift, five walkers, 40 launches, `H=8`
and prefix budgets `0.25/0.50` are unchanged. Candidate actions are null or one
chain-neighbor refresh. For each action the privileged development evaluator
computes the exact current `H`-step return `U_p(a)`. The launch rule is

\[
a_p\in\arg\max_a\{U_p(a)+B_p(a)-Q_pc_p(a)\},\qquad
Q_{p+1}=[Q_p+\nu(c_p(a_p)-\bar c)]^+,
\]

with the hard prefix cap applied first. The reset is exactly

\[
B_p(j\to i)=\frac{\beta}{2}\|\theta_j-\chi_{j\to i}\|^2,
\qquad \beta=\frac{2b_{\max}}{R_{\rm cache}^2}.
\]

The outcome-free radius is `R_cache=16*0.04=0.64`: each donor receives at most
eight registered four-coordinate increments and each coordinate increment has
magnitude at most `0.04`. The fixed catalogue is

\[
b_{\max}\in\{0,0.025,0.05,0.1,0.2\},\qquad
\nu\in\{0.0025,0.01\}.
\]

For each held-out seed, one global pair is selected by total `H`-return on the
other seven seeds across both drifts and budgets. The held-out outcomes are not
read during selection. This leave-one-seed-out policy is the only reported
candidate; per-cell or per-outcome oracle selection is prohibited.

## Strengthened comparators

The strong per-cell envelope includes every prior online policy except
`no_refresh`, importantly including the privileged receding `oracle_h8` rule.
The exact prefix-budget DP/MILP remains the dynamic ceiling. This is stricter
than the earlier confirmation envelope and prevents the potential controller
from receiving credit merely for access to the `H`-step branch value.

## Frozen gates

There are 320 rows: 8 seeds x 2 drifts x 2 budgets x 5 reset caps x 2 queue
steps. Every gate is mandatory.

1. P1: the grid and keys are complete and unique.
2. P2: every output is finite; replay errors and prefix excess are zero; the
   analytic cache radius and reset cap hold pathwise.
3. P3: every held-out choice belongs to the frozen catalogue and uses only the
   other seven seeds.
4. P4: the active exact oracle retains positive headroom over the strengthened
   envelope that includes `oracle_h8`.
5. P5: on drift `0.04`, cross-fitted candidate headroom recovers at least 10%
   of exact-oracle headroom above that envelope.
6. P6: candidate strictly exceeds the strengthened envelope in at least 75%
   of the 16 active seed-budget cells.
7. P7: active median normalized headroom is at least `0.0005` (0.05%).
8. P8: every fold selects a nonzero reset cap.
9. P9: the queue price changes at least one active decision relative to the
   same unpriced potential rule.
10. P10: across all 32 cross-fitted cells, aggregate candidate gain over the
    authenticated no-refresh policy on the same public prefix is positive.

Any failure permanently stops this exact potential bridge. Thresholds,
catalogue, seeds, source hashes and strong policies must not be changed after
outcome generation. No partial pass authorizes a learned critic, confirmation,
GPU, HPC4, or formal claim.

## Execution and provenance

Frozen pre-run identities are:

- config SHA-256:
  `8EC001E4F394918673CA09BC3F6749869592EAEFEF5B474039D149E8B86BAC8C`;
- runner/analyzer SHA-256:
  `9E50453A798F1B364A4E7BCAAE29FE5BC7D9A0536F9FDB87F555F8810A2D7227`.

The run is local CPU only, preferably eight isolated one-seed processes,
followed by hash-checked deterministic merge and a clean reproduction. Raw
rows stay ignored under `tmp/policy_dependency_sync/multiwalker_paired_potential_dev_v1`.
The source commit, config/runner hashes, commands, timestamps, row/summary
hashes, all gates and stop decision must be recorded before any subsequent
experiment.
