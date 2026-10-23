# Human approval addendum, 2026-10-19

After reviewing the separate budget question, the human explicitly answered:
“批准独立8 CPU小时”. This approves the proposed V3 gate validation's maximum
28,800 CPU seconds, including full replay and analysis, independently of the140h task.

The earlier release_check.json and proposal record the state BEFORE this reply;
their “unapproved” wording is historical, not a current blocker. Fresh direct
validation itself is still pending. No new scientific driver has started.

The approved design is materialized in experiments/v3_risk_gate8h/protocol.json.
Implementation/runtime/source admission must still be completed, verified and
committed/pushed before any new data. Default launch remains after the sole140h
native workflow reaches its terminal decision and exits, so no competing worker.
Do not combine budgets, relax cells/gates, reduce replay, extend the cap or retry.
No further budget approval is needed for the exact approved design; new scope or
an overrun still needs human direction. Existing known-grid evidence is not reset.
