# Learning-value separation theorem

## Proposition

Consider the compact separated Gaussian common-factor class with finite
action catalogue, public `lambda <= 0.94`, positive message/environment costs,
bounded delay, and a declared theorem-facing safety budget. Let `B_id` be the
smallest dual-budget scale at which a statistically reliable fixed
identification probe is affordable. Let `B_value` be the smallest scale at
which the same family of probes is also downstream worthwhile and satisfies
the `S_mean` safety rule.

If `B_id < B_value`, then the separation zone

```text
Z = {B : B_id <= B < B_value}
```

is nonempty. For every `B` in `Z`, the information-only policy may probe
because identification is reliable, while the learning-aware policy falls
back because the probe is not yet amortized by downstream learning value or
the safety constraint.

For any true instance direction in which the post-probe expected risk
`R_info(B)` exceeds the fallback risk `R_all(B)`, the opportunity loss is

```text
Loss(B) = R_info(B) - R_all(B) > 0.
```

The relative loss is `Loss(B) / R_all(B)`. T-018 freezes `0.03` as the
practical effect threshold for the scan.

## Explicit decomposition

The sufficient probe cost is

```text
C_id(q,b,n,D) =
max(n(h+q)/beta_message, (nb + D)/beta_environment).
```

The downstream gain after a correct high-regime commit is

```text
G(B,D) = R_all(B_after_probe,D) - R_star(B_after_probe,D).
```

The learning-aware rule probes only when the identification cost, latency
cost, wrong-commit loss, and safety slack are dominated by this downstream
gain. Therefore `B_value` weakly increases with larger probe cost, delay,
overhead, or more imbalanced binding resources, all else equal. It weakly
decreases as the oracle gain grows, subject to finite-budget rounding.

## What is not claimed

The theorem does not claim `Z` is universally nonempty. If the reliable
identification threshold and the downstream-worthwhile threshold coincide,
the two policies are equivalent on the registered grid. If `Z` appears only
in a few extreme handcrafted cells, the adaptation-cost route fails the
novelty gate.

The theorem also does not hide gray-zone limitations. It separates
identification feasible, identification statistically reliable, adaptation
downstream worthwhile, and adaptation safe.

## Relation to EXP-016A thresholds

`B_value` is the T-018 analogue of the sufficient learning-aware threshold
used by EXP-016A, but T-018 does not modify EXP-016A. `B_oracle` remains a
known-instance amortization quantity from EXP-016A provenance and is not used
to revive the stopped EXP-016A pilot.
