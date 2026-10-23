## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: implementation and theorem-interface audit
- Origin Date: 2026-09-07
- Verification Status: VALIDATED NEGATIVE DEVELOPMENT RESULT
- Version Label: pistonball_score_scale_relu_v1

# ReLU Pistonball signed-score scale audit

## Decision

The frozen score-scale gate fails and the equal-resource controller-headroom
matrix is not authorized on the current ReLU critic. Training completed and
all accounting invariants passed, but the signed learning component was
identically zero in every one of 980 scored launches. The cache and queue terms
were both active, so the failure isolates the differentiable learning-value
interface rather than the packet system or Lyapunov queue.

## Provenance

- Frozen source commit: `738eacf8ebe689a2fea9adb6fffdb57d403eb322`.
- Analyzer fail-closed amendment: `fba2827499da8c3bc312e8c362be52edb386cc91`.
- Slurm job: `1825676`, A30 x1, CPU 8, 16 GB.
- Raw output:
  `/scratch/jzhuangag/causal-policy-freshness-icml2027/artifacts/score-scale-738eacf8ebe689a2fea9adb6fffdb57d403eb322/raw.json`.
- Raw SHA-256:
  `cd471f14d3e06779ec225889b61a58672a12cd9142ee157749f52897ed4f4dce`.
- Patched analyzer summary SHA-256:
  `738d0b42f06e4928e5ca0d0b8afbf51570ef439e928cd58f445c82403205f953`.

The Slurm batch ended `FAILED 1:0` after 5:48 because the originally committed
analyzer raised on an identically zero component. The learner had already
completed and wrote a valid 1.1 MB raw JSON. The fail-closed amendment changed
only the analyzer behavior: it records the failed gate and a null recommended
weight instead of raising. No trajectory was rerun or overwritten.

## Frozen diagnostics

- scored launches: 980;
- signed-favorable fraction: `0.0`;
- raw signed learning-drift delta: minimum = median = maximum = `0.0`;
- cache-reset energy median: `1.83218e-5`, maximum `0.00513702`;
- queue active fraction: `0.784694`, median price `1.0`, maximum `4.0`;
- legacy composite selected fraction: `0.745918`;
- refresh units: 731 of 1,024 allowed;
- optional policy bytes: 65,406,956;
- learner runtime: 337.08 seconds;
- numerical, drain, and finite total-budget invariants: pass.

Returns are tainted development diagnostics and did not enter the gate or
calibration. The terminal return decreased in this short run, so it supplies
no efficacy evidence independently of the score failure.

## Why the signed term is exactly zero

The critic action head is a feed-forward ReLU network. Away from activation
kinks it is affine in the concatenated joint action. Therefore, for distinct
owner and donor actions,

\[
 \frac{\partial^2 Q_\phi(s,a)}
 {\partial a_i\,\partial a_j}=0
 \quad\text{almost everywhere}.
\]

Within one activation region, the owner actor Jacobian is independent of the
donor parameters. Hence the cross-policy VJP used by the estimator satisfies

\[
 \nabla_{\theta_j}
 \langle g_i^{\rm current},g_i^{\rm cache}\rangle=0
 \quad\text{almost everywhere}.
\]

This explains both the present result and the earlier observation that the
signed-only scheduler selected zero edges. A first-order cross-policy VJP
cannot reveal strategic freshness when the critic has zero mixed curvature.

## Authorized repair

The main problem is not stopped, but the ReLU scoring interface is. The next
development amendment must use a smooth critic action head, preferably
Softplus with fixed beta, so that mixed action curvature exists and has an
explicit derivative bound. The architecture change must pass both a mixed-VJP
unit test and a fresh-cache learner qualification before any equal-resource
controller comparison. Finite differences over every edge are not adopted
because they would replace the claimed one-VJP `O(Delta)` interface with
`O(Delta)` separate backward evaluations.
