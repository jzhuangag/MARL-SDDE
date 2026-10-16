## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-free implementation qualification
- Origin Date: 2026-09-07
- Verification Status: OUTCOME-FREE DEVELOPMENT DESIGN
- Version Label: pistonball_smooth_interface_qualification_v1

# Smooth Pistonball learner and score-interface qualification

The ReLU scale audit is permanently retained as a failed gate. Its structural
cause is zero mixed action curvature almost everywhere. This amendment does
not change that result. It replaces ReLU by SiLU in both actor and centralized
critic for prospective runs, preserving a globally smooth nonlinear path and
the one-cross-policy-VJP implementation.

Two isolated development runs test separate necessary conditions. The
fully-fresh run uses new seed `79005`, 4,096 launches, zero packet delay, actor
step `0.01`, critic step `0.0003`, and all 19 optional donor refreshes per
launch. It passes only if terminal decentralized return improves by at least
2.0 from its own initialization and gradient/actor-drift diagnostics are
finite and nonzero. The score-scale run reuses development seed `79004` to
isolate activation choice against the preserved ReLU raw, but its return is
ignored. It uses 2,048 launches, maximum delay eight, terminal budget rate
0.5, and requires:

1. exact reconstruction of every best-edge-minus-null index;
2. nonzero signed learning and cache components;
3. an active dual-queue price;
4. signed learning favoring an edge in at least 1% of scored launches.

Both gates must pass to authorize outcome-free design of an equal-resource
controller-headroom matrix. Any failure stops before that matrix. These seeds
and all resulting trajectories are permanently excluded from pilot and formal
populations. No controller performance claim or statistical inference is
permitted from this qualification.
