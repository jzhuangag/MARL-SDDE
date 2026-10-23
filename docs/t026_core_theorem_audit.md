# T-026 core theorem audit

## Verdict

No proof-breaking error was found in Theorems 3, 4, 5, or 9 after checking
their displayed recursions against their stated assumptions. Theorems 3 and 4
are valid sufficient bounds for the predictably decorrelated algorithm;
Theorem 5 is an exact Gaussian location result with one additive resource
cost; and Theorem 9 is an exact covariance identity for the registered
hidden-sharing construction.

The audit did find one material over-attribution: earlier claim maps described
Theorem 5 as a dual-budget/delay theorem. It is not. That wording is corrected
in the proof program and T-025 ledger. The dual-budget statement comes from
AC-7, the controlled-belief occupation lower bound, the learning-value
separation theorem, and EXP-016B.

## Theorem-by-theorem checks

### Theorem 3: PASS within the fixed-delay, decorrelated scope

- The conditional total-variation argument is valid because the current
  error vector is measurable before the fresh block and the sample Jacobian is
  bounded.
- The delayed difference uses indices no earlier than `k-2D`, matching the
  sliding envelope.
- Two Jensen bounds give the registered
  `eta^2 L^4 tau_rms^2 R_k` remainder.
- The cross term and squared remainder yield exactly the displayed
  contraction coefficient.
- After `2D+1` updates, the old envelope has been fully replaced.

Required paper wording: delays in the theorem are fixed bounded integers and
the joint sample is obtained after a predictable decorrelation gap. The result
does not cover the unthinned recursion or arbitrary time-varying delays.

### Theorem 4: PASS within the bounded affine scope

- Total-variation duality controls the conditional innovation mean without
  assuming conditional centering.
- The stationary squared affine update is bounded by the displayed curvature
  and innovation second moments; the TV remainder follows from the pathwise
  `L ||e|| + G` bound.
- Young's inequality produces `a_delta` and `beta_delta` with the displayed
  constants.
- The affine delay telescoping includes the innovation, producing both the
  state and constant delay remainders.
- Optimizing the scalar Young parameter gives the displayed square form and
  the block residual recursion.

Required paper wording: bounded innovations and a stationary zero-mean
aggregate innovation are assumptions. The theorem is not a general unbounded
neural-TD result.

### Theorem 5: PASS, but only for one additive cost

- Sherman--Morrison gives the exact per-round Fisher information.
- Generalized least squares attains the fixed-q constant risk and a diffuse
  Gaussian prior supplies the matching minimax lower bound.
- For predictable adaptive actions, action kernels add no
  parameter-dependent likelihood term, and the pathwise budget bounds total
  precision.
- The continuous optimum follows from a convex scalar objective.

Correction: the constraint is `sum_t(h+q_t) <= B`. It is not a two-resource
message/environment theorem, and q-dependent gaps or heterogeneous delay
cannot simply be hidden in a constant `h`.

### Theorem 9: PASS for the explicit hidden-sharing model

- Every agent update has the original marginal distribution.
- A pair shares the common draw with probability
  `sqrt(rho)^2 = rho`, giving off-diagonal covariance `rho Sigma_theta`.
- Summing diagonal and off-diagonal blocks gives
  `rho + (1-rho)/q` exactly.

Required paper wording: the identity is induced by the explicit Bernoulli
hidden-sharing construction. A scalar pairwise correlation label alone does
not imply proportional cross-covariance for arbitrary multivariate gradients.
EXP-018B prospectively validates this construction at frozen neural TD
parameters; it does not prove nonlinear learning convergence.

## Remaining proof boundaries

The following are still open and must not enter the theorem statement by
implication:

- the unthinned finite-time Markov-TD recursion;
- an SDDE-to-discrete approximation theorem;
- nonlinear optimization/convergence under parameter drift;
- uniform unknown-mixing adaptation as mixing approaches one;
- uniform matching to the controlled-belief occupation optimum.

None is required for the current narrow fixed-q mainline if the paper labels
SDDE as interpretation and treats nonlinear evidence as a fixed-gradient
bridge. They would become mandatory if the title or abstract promises an
online universal controller or general nonlinear MARL convergence.

## Computational cross-check

The four closest existing test modules for the Gaussian speedup lower bound,
affine Markov certificate, adaptive change of measure, and T-017 theory audit
pass together: `30 passed in 43.25s`. These checks support the algebra and
high-precision likelihood formulas; they are not substitutes for the proof
scope audit above.
