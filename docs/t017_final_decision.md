# T-017 final decision

## Decision: A + D

**A:** A nontrivial finite-budget adaptation-worthwhile threshold sandwich is
proved on an explicit compact separated class. It differs from pure
controlled sensing because the sufficient threshold must amortize
identification against subsequent correlation-limited learning risk, both
resource costs, delayed usability, wrong-oracle regret, and a
baseline-relative safety deficit.

**D:** The unrestricted fully unknown-mixing problem is closed negatively.
Every positive algorithm or theorem must assume public
\(\lambda\leq1-\gamma\) or consume an independent valid mixing certificate.

This is not option B: the retained theorem is not merely an information per
control-cost specialization. It is not option C either: a separated-class
finite-budget threshold upper/lower comparison is available. However, A does
not upgrade to global matching of the full adaptive controlled-belief
occupation optimum; that stronger AC-9 component remains open.

## Formula audit

The documentation's two missing addition signs were genuine transcription
errors. The corrected stationary model and prediction step are

\[
C_{S_t}=\lambda^{b_t}C_{S_{t-1}}+
\sqrt{1-\lambda^{2b_t}}\xi_{j,t},
\]

\[
P^-_{j,t}=\lambda^{2b_t}P^+_{j,t-1}+
(1-\lambda^{2b_t})\theta_j.
\]

The Python implementation already used these additive formulas. Tests now
lock stationary initialization and the documentation signs. The audit also
confirmed irregular stride, two hypothesis-specific filters, both KL
directions, predictable action-kernel cancellation, bounded stopping, the
extra uniform-integrability requirements for an unbounded extension, dual
budgets, and that delay affects resources/usable learning horizon rather than
the observation covariance.

## Novelty gate

The gate **passes only for the narrowed joint threshold/impossibility claim**.
The following are explicitly inherited: general controlled-Markov likelihood
factorization, action-kernel cancellation, information per control cost,
best-arm allocation change of measure, covariance-adaptive simultaneous
subset identification, and factored multi-agent BAI matching.

The defensible contribution candidate is:

> finite-budget safe adaptation thresholds coupling unknown correlation,
> participation-dependent information and terminal learning risk,
> communication/environment costs and delayed usability, plus a formal
> impossibility theorem for unrestricted unknown mixing.

No SDDE convergence claim is made. AC-12 remains open and out of scope.

## Frozen experiment state and authorization

- EXP-015A remains exactly a 7/8 pilot failure.
- The gate remains `0.80`; the observed fallback remains `0.777778`.
- No gate, seed, result, formal seed, HPC4 job, GPU job, or experiment was
  created or changed.
- T-017 permits the *next step to design* CPU EXP-016A, but **does not permit
  starting it and does not create its preregistration**. Any such work still
  requires a separate preregistration commit.

The machine-readable statuses and citation checks are in
`theorem_dependencies_t017.json` and `citation_verification_t017.json`.

## Verification

The complete repository suite passes under the existing `ust2` environment:
`156 passed in 7.05 s`. Six new T-017 algebraic tests cover the exact
Gaussian-scale TV/Le Cam floor, covariance KL, finite irregular-law
convergence to \(\lambda=1\), the one-latent-draw KL ceiling, terminal-risk
asymptotics, and oracle-gap threshold divergence. Two additional tests lock
the additive state and prediction formulas. All documentation JSON files
parse successfully.
