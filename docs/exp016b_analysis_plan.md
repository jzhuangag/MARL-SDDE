# EXP-016B analysis plan

The primary estimand is the CRN-paired mean terminal-risk difference
`risk(information_only) - risk(learning_aware)` in the registered Layer-A
practical-effect finite-Z population. Report its relative form against
always-all risk, oracle regret, identification error, probe/commit/fallback,
selected `(q,b)`, message/environment use, usable post-delay updates, and
crossover location. Layer B additionally reports TD parameter error and
Bellman/teacher error.

All continuous primary families use paired seed-block means and simultaneous
one-sided Bonferroni bounds at familywise alpha
`0.01`. Statistical direction (lower bound
above zero) and practical magnitude (point estimate at least 3%) are separate.
Neutral-Z cells remain in their frozen descriptive table and cannot rescue
P4. P5 uses the frozen scenario-level coverage denominator and threshold
`60%`. `S_mean` alone is compared with
`epsilon_safe`; `S_path` and CVaR90 are descriptive.

Identification certification is a deterministic theorem/runtime compliance
gate. Empirical identification error is calibrated only over registered
aggregate families; per-cell rare errors are descriptive implementation
anomaly checks, never the infeasible EXP-016A 64-seed rare-event gate.

If Layer A passes but Layer B lacks aggregate directional consistency, no
multi-agent Markov-learning transfer claim is allowed. Any mandatory P1--P12
failure forbids a formal run. A clean same-seed rerun must reproduce core
CSV/JSON byte for byte before formal authorization.
