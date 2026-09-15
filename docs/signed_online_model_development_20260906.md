## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-06
- Verification Status: VERIFIED
- Version Label: signed_online_model_development_v1

# Observable signed-model controller: development decision

## Decision

The translation-equivariant online model estimator retains enough of the
PDSG-SIGN-001 oracle value to justify a separately preregistered CPU
confirmation.  It is not formal paper evidence and does not authorize a GPU or
standard-MARL run.

The earlier zero-initialized estimator result is rejected from scientific use.
Parameter coordinates have no privileged origin, and zero initialization could
exploit the synthetic fixture's sign pattern.  The retained implementation
initializes the latent reference estimate at the observable current iterate and
has a tested translation-equivariant decision rule.

## Executable estimator

For a completed packet with public conditional row `r`, launch cache `chi`, and
observed noisy gradient `G`, the learner observes

\[
y=r^\top\chi-G=r^\top\theta^\star-\xi.
\]

It updates the previous estimate using normalized LMS,

\[
\widehat\theta^\star_+
=\widehat\theta^\star
-\gamma\,r\,
\frac{r^\top\widehat\theta^\star-y}{\max\{\|r\|^2,10^{-12}\}},
\qquad \gamma=0.25.
\]

Only completed ordinary training packets are used.  There is no probe, extra
trajectory, future packet noise, or true-target input to the online graph
scorer.  The next null-or-one-edge action minimizes the same signed Lyapunov
index as the oracle after replacing the unknown target by this predictable
estimate.  Selection remains an exact `O(Delta)` scan.

The estimator is an exact-model realization of observability, not yet the
neural critic/JVP estimator required by the eventual MARL algorithm.

## Development population and result

- grid: the unchanged 24-cell binding-budget grid;
- active cells: 16 with coupling in `{0.6,0.9}`;
- negative controls: eight coupling-zero cells;
- seeds: `43005--43012`, disjoint from PDSG-SIGN-001;
- launches: 80;
- comparator: the outcome-wise strongest feasible member of the unchanged
  24-policy static/online baseline family;
- status: development only.

Across the 16 active cells:

| Quantity | Observation |
|---|---:|
| median terminal-risk improvement | 10.8330% |
| terminal-risk strict-improvement fraction | 16/16 |
| minimum terminal-risk improvement | 5.7379% |
| median cumulative-risk AUC improvement | 1.4261% |
| AUC strict-improvement fraction | 16/16 |
| median oracle terminal-headroom recovery | 98.2146% |
| minimum mean dynamic graph supports | 11.875 |

The maximum message-budget excess was `-0.0765625`; environment charging was
exactly four transitions per launch.  Both terminal and cumulative-risk
differences were exactly zero in all coupling-zero controls.

## Action-level audit

On all 128 active seed-cell comparisons, online and oracle graph-action traces
agreed in 36.71875% of cases.  Their maximum absolute cumulative-risk,
terminal-risk, and message-rate differences were respectively `0.0402761`,
`0.0405955`, and `0.0375`.  The median and maximum final target-estimation
errors were `0.316435` and `0.828094`.

Thus the retained result is not caused by identical oracle actions or a
vanishing estimation error.  It says that the signed action ranking has enough
margin in this development population for a noisy, causal system-identification
step to recover most oracle value.

## Provenance

- full-envelope development output SHA-256:
  `9B8FA8A3353CA9DB1C41682DBF45A23A0332663DB434BEDDC32E7E945DFC455B`;
- action audit output SHA-256:
  `B04D4956A4080EE3E7B315B10C3B9D6DB4E6033620FB3E525D622A2ED493221A`;
- game source SHA-256:
  `0D9623C90CF16CC8BE50CA5E691C480748C736820440980893325F7362143DAF`;
- development runner SHA-256:
  `9191FA4A6775AF7826FD47AE6B86918B462A2037A5997906A143FD983501531E`;
- action-audit runner SHA-256:
  `629C693F1E0B12511345B807BFBB2449054F96CF3753AA976494AEC298488A53`.

Raw outputs remain ignored under `tmp/policy_dependency_sync/`.

## Next admissible stage

Freeze a new CPU confirmation before using new outcomes.  It must use new
seeds, retain the 24-cell population and all 24 strong baselines, include the
oracle only as a diagnostic upper bound, require positive terminal and AUC
effects, audit oracle-headroom recovery, enforce exact resource charging and
uncoupled controls, and reproduce byte-for-byte.  Failure stops this estimator
without changing thresholds or recycling PDSG-SIGN-001 outcomes.

