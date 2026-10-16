# TSP-V3-CERT-SHIELD-002 preregistration

## Design change motivated before new outcomes

TSP-V3-CERT-SENS-001 showed that a single worst-case score can preserve
coverage yet over-penalize learning.  The successor separates two roles that
the theory permits:

1. upper certificates define a hard robust-feasibility shield and a valid
   risk envelope;
2. the nominal finite-horizon Lyapunov score ranks only actions that pass that
   shield.

This is a substantive controller change, not a rerun of the failed policy.
The executed \((q,b,\eta)\) is never altered after selection.  If the nominal
minimizer is not robustly feasible, the controller moves to the next nominally
ranked certified action.

## Frozen design and gates

The model, 12 cells, resource budget, checkpoints, candidate catalogue,
moderate/strong stress levels, myopic ablation, and correlation-only ablation
are identical to the previous study.  All seeds are new and disjoint:

- development: 35000001--35000008;
- confirmation: 36000001--36000064.

Development requires finite payloads, exact budget feasibility, stable upper
certificates, terminal parameter-risk ratios at most 1.05 and 1.10 for the
moderate and strong shields, and finite-horizon ratios at most 0.95 versus
one-step myopic and 0.98 versus correlation-only.

Confirmation additionally requires all 24 one-sided 99 percent bootstrap
upper means below the corresponding robust certificate and paired-bootstrap
95 percent upper ratios below one for both ablations.  Any failed development
gate stops confirmation; any failed confirmation gate keeps the result out of
the manuscript.

