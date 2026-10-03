# Information-only taint audit

## API boundary

The registered information-only score is
`run_exp016a.information_only_score(theta0, theta1, mixing, q, b, overhead,
ray)`.

The function signature does not admit downstream risk, wrong-commit loss,
`epsilon_safe`, oracle action, hidden regime, true theta, latent state, or
policy outcome data. It scores identification information per charged probe
cost only.

## T-018 interpretation

The EXP-016A Amendment 1 active subset was empty because, on the original
above-`B_S` cells and active-subset definition, the frozen information-only
and learning-aware intended paths often coincided. T-018 does not reinterpret
that as an EXP-016A result. Instead, it freezes a broader outcome-free grid
to test whether a genuine separation zone exists before any new experiment is
designed.

No scientific outcome is present in this audit.
