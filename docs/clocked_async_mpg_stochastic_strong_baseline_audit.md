# Stochastic strong-asynchronous-baseline audit

Status: development-only decision audit; not preregistered evidence.

## Decision

The current `alpha_i proportional to 1/L_ii` and rate-balanced allocations
must not be presented as performance improvements over a strong asynchronous
common-step learner.  Across the full 16-cell development grid, separately
tuned `single_flight_local` was slower than `common_global` in 15/16 cells at
fractions in `{0.1,0.2,0.4,0.8}`.  Its geometric time and charged-transition
ratios were `1.43493` and `1.43511`.  Extending the development scale did not
reverse that conclusion.

This finding changes the paper-facing claim but does not kill the event-driven
architecture.  A newly implemented **certified common-step** rule uses the
single-flight Lyapunov condition with only off-diagonal teammate sensitivity
in the history term.  At its maximal certified scale it was faster than the
fully utilized frozen-policy shadow barrier in all 16 cells:

| comparison (certified / comparator) | geometric time | geometric transition work | final gap | cells faster |
|---|---:|---:|---:|---:|
| common-step raw asynchronous | 1.11977 | 1.12203 | 1.12928 | 2/16 |
| fully utilized shadow barrier | 0.41318 | 0.41142 | 0.74501 | 16/16 |
| generic rate-balanced certificate | 0.21587 | 0.21377 | 0.35062 | 16/16 |
| local-curvature certificate | 0.79321 | 0.79191 | 0.82615 | 10/16 |

Thus the defensible algorithmic picture is a three-way frontier:

1. a tuned raw asynchronous common-step learner is the empirical speed
   reference but is not covered by the current worst-case delay certificate;
2. the certified asynchronous common-step learner pays about 12% in this
   development population while retaining roughly 59% less elapsed time and
   transition work than the barrier;
3. the fully utilized barrier uses fresh policies but remains straggler-bound.

The future claim is not that a novel step heuristic beats raw asynchrony.  It
is that single-flight barrier-free independent policy learning admits an
interaction-sensitive Lyapunov certificate and a wall-clock phase in which a
certified learner preserves a substantial part of the asynchronous gain.
Whether this supports an ICML paper still depends on independent stochastic
confirmation, a matching wall-clock theorem/lower boundary and standard MARL
benchmarks.

## New algebra closed by this audit

For common `alpha`, let `L_off` be `L` with zero diagonal and

\[
u_j=\sum_i p_i\ell_i^{\rm off}L_{ij}^{\rm off},
\qquad
\ell_i^{\rm off}=\sum_{j\ne i}L_{ij}.
\]

The largest common step in the existing bounded-event-delay proof is the
minimum positive block root of

\[
L_{jj}\alpha+(1+\delta)D^2u_j\alpha^2\le 1.
\]

This removes diagonal curvature only from the stale-history price; it remains
in ordinary block smoothness.  The implementation also closes the compatible-
history enumeration with conditional packet bias and centered packet noise:

\[
\begin{aligned}
\mathbb E[V_{k+1}\mid\mathcal G_k^-]
\le{}&V_k-\frac12\sum_i p_i\alpha_i\|g_i(\theta^k)\|^2\\
&+\frac{1+\delta^{-1}}2\sum_i p_i\alpha_i\beta_i^2\\
&+\frac12\sum_i p_i\left(
L_{ii}\alpha_i^2+(1+\delta)D^2\widetilde w_i\alpha_i^2
\right)\nu_i^2.
\end{aligned}
\]

This is an algebraic finite-time interface, not yet the renewal-process
wall-clock theorem.

## Development protocol and taint boundary

All runs used local CPU, fixed-horizon `H=16`, batch size `16`, maximum time
`180`, target normalized potential gap `0.3`, couplings
`{0,.08,.16,.24}`, service ratios `{1,2,4,8}` and four development seeds per
cell.  Service-duration and trajectory streams are separated; matching
policies use the same named CRN stream.  Every completed packet, barrier-
cancelled partial packet and horizon-partial packet is charged in transition
work.

The files under `tmp/` are ignored development artifacts.  Their SHA-256
digests are:

- `stochastic_strong_baseline_development_v1.json`:
  `40463F014A64639BA46EC1844432D12190105F0B0D4FB3DC778EBF29DE97070B`;
- `stochastic_strong_baseline_development_v2.json`:
  `0EA7837A0B8C3539EEDCED4B3FACA019B3B27DA4CE8361C56092CC10A94371BB`;
- `stochastic_strong_baseline_development_v3.json`:
  `2CC09ED54518440B380CA4CDEA062498FC7DF461FF7375FA428FB1B21A72DD25`;
- `stochastic_certified_constant_development_v1.json`:
  `DE8F568BA9F9AB5B20622A3389E6CC8EB6A4795B87364727450E3DA57A67DAC8`.

The first three scans expanded the step-fraction catalogue after each boundary
was selected and are explicitly tuning-tainted.  The fourth used the natural
maximal certified fraction one and is still development data.  None may be
reported as independent confirmation or used for a p-value.

An additional one-seed mutation of local weights and initialization was used
only to test whether making the slow block essential rescued rate balancing;
it did not.  That mutation was never written to source or a scientific result
directory.

## Stop/go rule

Proceed only with a separately committed stochastic confirmation that fixes
the maximal certified common step, strong raw asynchronous common step,
fully-utilized shadow barrier and the two allocation ablations.  Use a fresh
seed namespace and unchanged Markov game grid.  The primary comparison is
certified asynchronous versus the fully utilized barrier; raw asynchronous is
a mandatory strong reference and its advantage must be disclosed.  Failure of
the time, work, endpoint, delay, accounting or reproducibility gates stops the
standard-MARL stage.
