# EXP-012A: latent correlation from observable Markov collisions

## Material Passport

- Artifact: preregistration
- Role: CPU validation of Theorem 7 and hidden-sharing identification
- Formal seeds: 128 per scenario
- Formal base seed: 20270201
- Pilot exclusion: four seeds with base 20270111 were inspected to debug the
  implementation and set the fixed thresholds below; they are excluded from
  formal evidence

## Question

Can the controller obtain a safe and useful cross-agent correlation
certificate without observing the hidden sharing masks used in EXP-011B?

Each probe reveals only whether two agent Markov samples are equal.  For the
registered symmetric two-state marginal,

\[
c_\pi=\Pr(E_1=E_2)=\tfrac12,
\qquad
\Pr(Y_1=Y_2)=\tfrac12+\tfrac12\rho.
\]

The estimator receives the collision bit, not either hidden mask.

## Frozen confidence design

The persistence stream retains the beta-binomial mixture confidence sequence
with \(\alpha_p=.005\).  Before every collision block, its upper endpoint is
rounded upward to a .002 grid and selects the smallest gap satisfying

\[
\delta_j^+
=
\min\{1,\tfrac32(2p_j^+-1)^{b_j}\}
\le .01.
\]

For \(n\) collision observations, Theorem 7 uses

\[
\vartheta_n^+
=
\min\left\{
1,\bar X_n+
\sqrt{\frac{\log\{\pi^2n^2/(6\alpha_\rho)\}}{2n}}
+\frac1n\sum_{j=1}^n\delta_j^+
\right\},
\quad \alpha_\rho=.005,
\]

and \(\rho_n^+=2\vartheta_n^+-1\), clipped to \([0,1]\).  The two sequences
have joint time-uniform coverage at least .99 under arbitrary predictable
probe counts.

## Frozen resource protocol

- Total resource: 20,000.
- Initial persistence-only transitions: 128.
- Decision block: 2,000.
- Collision-probe cost: \(b+8+2\).
- If no collision probe fits in a block, the full block refines persistence.
- Otherwise the maximum number of probes fitting in the block is executed;
  leftover resource also refines persistence.

The final rounded \((p^+,\rho^+)\) is passed to the unchanged scalar
\((q,b,\eta)\) controller for delays \(D\in\{0,2\}\).  Exact covariance
propagation at the true parameters evaluates conditional safety.

## Frozen scenarios

\[
p\in\{.5,.9,.98\},\qquad
\rho\in\{0,.5,.9\}.
\]

Each of the nine cells receives 128 fresh seeds.

## Preregistered decision

Both validity gates and at least three of four scientific gates must pass.

### Validity gates

1. Joint simultaneous coverage of the true \(p\) and latent \(\rho\) is at
   least 97.5%.
2. Every updating final action on a simultaneously covered run has exact
   covariance radius strictly below one.

### Scientific gates

1. Median final \(\rho^+\) is non-decreasing in true \(\rho\) at every
   persistence.
2. At \(p=.5\), median \(\rho^+\) is at most .25 for \(\rho=0\), lies in
   \([.5,.85]\) for \(\rho=.5\), and covers .9 for \(\rho=.9\).
3. At \(p=.98\), every correlation cell has at least 50 median collision
   probes.
4. For each delay, median selected \(q\) at \(\rho=.9\) is no larger than at
   \(\rho=0\) for all three persistences, with at least one strict response
   across the two delays.

## Interpretation boundary

A pass replaces direct hidden-sharing labels by observable collisions in the
registered pair-sharing family.  It does not identify arbitrary Gaussian
common factors, for which exact sample equality has probability zero, or solve
unknown baseline collision probability in a general state space.  The
bounded-kernel extension in Proposition 4 supplies the next generalization.
