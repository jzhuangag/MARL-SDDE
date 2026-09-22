# Safety metric alignment

## Two safety quantities

Let `X = L_policy - L_all`. The two natural safety quantities are

```text
S_mean = [E(L_policy) - E(L_all)]_+ / E(L_all)
S_path = E[(L_policy - L_all)_+] / E(L_all).
```

They are not equal in general. Convexity gives

```text
[E(X)]_+ <= E[X_+].
```

Common random numbers and pairing reduce variance for comparisons, but they
do not turn the positive part of an expectation into the expectation of a
positive part.

## Theorem-facing metric

`theorem_derived_fallback.md` controls expected finite-budget risks. Its
equation (7) is therefore a bound on `S_mean`. The declared `epsilon_safe`
must be read as a theorem-facing `S_mean` budget.

`S_path` is a stronger descriptive tail metric. It may be reported in future
experiments, but T-018 does not claim that `epsilon_safe` controls it.

## Consequence for EXP-016A Amendment 1

Amendment 1 reported a prospective G5 minimum margin of approximately
`-0.015104`. That value is a static prospective `S_mean` calculation under
the Amendment 1 path model, not evidence that `epsilon_safe` controlled
`S_path`.

If a future implementation violates `S_mean`, that is a theorem/code
alignment problem and must be fixed before any new experiment design. If a
future project wants `S_path` safety, it needs a separate Gaussian
quadratic-loss tail or moment bound, additional slack, and explicit sample
complexity accounting.

No scientific outcome is present in this document.
