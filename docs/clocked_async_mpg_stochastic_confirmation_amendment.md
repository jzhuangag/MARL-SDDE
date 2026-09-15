# Stochastic confirmation Amendment 1: pathwise renewal certificate

Status: outcome-free correction frozen before a replacement run.

## Why version 1 was stopped

After preregistration commit `40cf489`, version 1 was launched locally with
four CPU workers.  Before the runner wrote an output directory or summary, a
theory--execution audit found that its label “certified” was too strong.  The
implemented single-flight common-step bound used fixed event probabilities
`p_i=lambda_i/sum_j lambda_j` in a one-event conditional drift.  This is exact
for iid marked-Poisson completions.  The simulator instead uses independent
bounded renewal durations.  Their long-run frequencies agree with `p_i`, but
successive completion marks are conditionally dependent, so stationary
frequencies cannot silently replace predictable one-event probabilities.

The process was interrupted before any endpoint or scientific summary was
written.  No outcome was inspected.  Nevertheless, all 64 version-1 seeds are
treated as touched and permanently excluded.  Version 1 is not resumed and
its committed config remains unchanged as provenance.

## Pathwise correction

For `L_off` equal to `L` with zero diagonal, define

\[
\ell_i^{\rm off}=\sum_{j\ne i}L_{ij},
\qquad
u_j=\max_i\ell_i^{\rm off}L_{ij}^{\rm off}.
\]

For a common step `alpha`, the pathwise history weight is
`w_j=alpha u_j`.  For every activation sequence with event delay at most `D`,
the single-flight Lyapunov drift closes when

\[
L_{jj}\alpha+(1+\delta)D^2u_j\alpha^2\le1
\quad\text{for every }j.
\]

Unlike the previous average-mark weight, the maximum over possible arriving
blocks makes the inequality valid without an iid mark assumption.  Version 2
fixes `delta=1` (history inflation two) so the same certificate includes the
fixed-horizon packet-bias split.  Random compatible-history tests enumerate
every possible activated block with nonzero bias and centered packet noise.

The online apply cost remains `O(d_i)` per packet.  Forming the static
certificate is `O(n^2)` followed by `n` scalar quadratic roots; no Hessian
inverse, covariance matrix, QP or candidate scan is used.

## Outcome-free development feasibility

A separate four-seed development namespace was used after the pathwise proof
and before this amendment.  On the unchanged 16-cell grid, the pathwise method
reached the target in every cell.  Over the 12 heterogeneous primary cells,
its geometric time/work ratios to the fully utilized barrier were
`0.44582/0.44502`, and all 12 cell medians favored it.  Its time/work cost
relative to raw asynchronous common-step learning was `1.53846/1.53323`.

These values only establish headroom.  They motivate a registered certificate-
cost ceiling of `2.0`; they are not confirmation evidence.  The raw method's
advantage remains a required disclosure, not a failed superiority claim.
Development artifact SHA-256:
`E383A89F696186745E63149CD1EA74CB71FAE20A0C17373C21131E059AE7F28C`.

## Version-2 changes

- new config version and untouched seed namespace;
- primary policy becomes `single_flight_pathwise_constant`;
- stationary-average certificate remains as an ablation;
- fixed history inflation is two;
- `S7` certificate-cost ceiling becomes `2.0` for both time and work;
- all other population, packet, target, accounting and primary benefit gates
  remain unchanged;
- row count becomes 6,144 because there are six registered policies.

All 12 gates remain mandatory, including byte-exact reproduction.  No GPU,
HPC4 or standard-MARL run is authorized by this amendment.
