# Policy Backpressure feasibility decision

Date: 2026-09-01.
Parent commit: `bab24105c1a86ebad945ac9601dbe5eb47bea377`.
Status: internal development audit, not a preregistration or manuscript result.

## Decision

**STOP THE CURRENT POLICY-BACKPRESSURE CONSTRUCTION.**

The stale unilateral-policy performance bound is valid on the exact declared
grid, but the stationary two-state cooperative Markov-game family does not
contain enough dynamic scheduling value relative to a per-cell best fixed
trust radius and a fresh serial learner.  The two predeclared feasibility gates
therefore fail.  No sampled controller, formal seed, standard MARL benchmark,
GPU run or experiment identifier is authorized by this audit.

This decision rejects the present construction and evidence path.  It is not an
impossibility theorem for every asynchronous MARL algorithm.  In particular,
the audit does not study action-duration asynchrony, nonstationary games,
constrained execution or neural approximation.  Those are different research
premises and must not be introduced retrospectively to reinterpret this result.

## Research question tested

For a discounted common-payoff Markov game with factorized joint policy
\(\pi=\prod_i\pi_i\), can a controller improve wall-clock learning by admitting
an agent proposal computed under an old joint policy \(\mu\), using joint-policy
path length rather than timestamp age as the measure of staleness?

The intended mechanism was a single chain:

1. bound the effect of one stale unilateral policy update;
2. use the bound to choose an online trust radius;
3. show that dynamic admission has material headroom over strong fixed rules;
4. only then derive a Lyapunov debt controller and test standard MARL.

The audit stops at item 3.

## Exact performance-bound interface

Let \(\pi^{i,\eta}\) replace agent \(i\)'s current policy by

\[
\pi_i^{\eta}=(1-\eta)\pi_i+\eta\bar\pi_i,
\qquad \eta\in[0,1].
\]

Let \(d_\mu\) and \(Q^\mu\) be the normalized discounted occupancy and action
value under the proposal's reference policy \(\mu\).  Define the stale
directional surrogate

\[
g_{\mu,i}(\bar\pi_i;\pi_i)
=\mathbb E_{s\sim d_\mu,\,a_{-i}\sim\mu_{-i}}
\left[
\sum_{a_i}(\bar\pi_i-\pi_i)(a_i\mid s)Q^\mu(s,a)
\right].
\]

Write

\[
d_i=\max_s D_{\rm TV}(\bar\pi_i(\cdot\mid s),\pi_i(\cdot\mid s)),
\quad
\delta=\max_sD_{\rm TV}(\pi(\cdot\mid s),\mu(\cdot\mid s)).
\]

For rewards in \([0,R_{\max}]\), the implemented conservative bound is

\[
J(\pi^{i,\eta})-J(\pi)
\ge
\frac{\eta g_{\mu,i}}{1-\gamma}
-\frac{4R_{\max}(1+\gamma)}{(1-\gamma)^3}\eta d_i\delta
-\frac{8\gamma R_{\max}}{(1-\gamma)^3}\eta^2d_i^2.
\]

The first penalty follows from perturbing discounted occupancy, the other
agents' action law and the reference action value.  The second is the usual
unilateral trust-region distribution-shift remainder with deliberately
nonoptimized constants.  For a policy-version path \(\pi_\kappa,\ldots,\pi_t\),
the triangle inequality gives

\[
\delta
\le
\sum_{r=\kappa}^{t-1}
\max_sD_{\rm TV}(\pi_{r+1}(\cdot\mid s),\pi_r(\cdot\mid s)).
\]

The CPU audit checks the displayed bound in 1,080 deterministic configurations.
The minimum `actual improvement - lower bound` is
`0.18449347705841096`; the maximum direct-TV/path-TV ratio is exactly `1.0`.
These checks support the implemented inequality, not novelty, statistical
coverage or a neural-policy guarantee.

## Development scan

The game has two agents, two states, binary actions and a shared reward.  State
zero rewards coordination and state one rewards anti-coordination through a
parity construction; joint action also influences the next-state law.  The
declared grid contains:

- discount factors `0.6` and `0.85`;
- transition-focus probabilities `0.65` and `0.9`;
- latency pairs `(1,3)`, `(1,6)` and `(2,7)`;
- wall-clock horizons `18` and `30`;
- six initial factorized policies.

This gives 144 deterministic development cells.  Every strong static value is
selected per cell from 21 fixed trust radii, including zero.  The strong
baseline is the better of this best asynchronous fixed-radius learner and a
best fresh serial learner.  This is outcome-aware and deliberately favorable
to the baseline.

The dynamic references are also outcome-aware:

- an exact one-step return-maximizing admission rule;
- a non-myopic beam search over five trust radii and width 64.

The beam search is a feasible dynamic lower bound, not an oracle ceiling.  A
positive gap would establish achievable headroom; absence of a gap does not
prove an upper bound.  It is nevertheless sufficient to fail the declared
positive qualification.

## Results and gates

| Quantity | Result |
|---|---:|
| Deterministic cells | 144 |
| Median non-myopic wall-clock headroom | -1.383408% |
| Maximum non-myopic wall-clock headroom | 2.293896% |
| Cells with at least 5% headroom | 0/144 |
| Median endpoint headroom | 0% |
| Maximum endpoint headroom | 0% |
| Path-bound controller active cells | 144/144 |
| Path-bound harmful updates | 0 |
| Median path-bound endpoint gap to strong baseline | -47.783409% |
| Best path-bound endpoint gap to strong baseline | -42.008811% |

Both mandatory gates fail:

1. median dynamic wall-clock headroom at least 10%: **FAIL**;
2. at least 60% of cells with headroom at least 5%: **FAIL**.

The result exposes two separate problems.  First, the declared stationary game
does not provide material dynamic-scheduling value once the fixed radius is
selected per cell.  Second, the valid analytic bound is too conservative to
serve as a competitive controller in this family.  Adding a debt queue cannot
create missing oracle value and would not repair the second issue.

## Research consequence

Do not proceed to a Lyapunov queue theorem, sampled Markov confidence layer or
standard MARL GPU benchmark for this construction.  Doing so would turn a
negative problem-value screen into an unnecessarily elaborate algorithm.

A future asynchronous MARL direction needs a different intrinsic state or
constraint whose value cannot be matched by a fixed trust radius.  Examples
such as asynchronous macro-action execution or shared time-average safety
budgets are separate hypotheses; they require a new bounded novelty and oracle
screen rather than modification of these 144 cells.

The older T-083A data, gates and failure remain unchanged and were not read as
outcomes by this scan.  No prior formal result was used for tuning.

## Reproduction

Source:
`experiments/policy_backpressure/policy_backpressure_feasibility.py`

Targeted tests:
`experiments/policy_backpressure/test_policy_backpressure_feasibility.py`

Commands:

```text
.\.venv\Scripts\python.exe -m pytest experiments/policy_backpressure/test_policy_backpressure_feasibility.py -q
.\.venv\Scripts\python.exe experiments/policy_backpressure/policy_backpressure_feasibility.py
```

The six targeted tests pass.  The complete experiment regression is
`860 passed, 7 skipped in 106.82s`; the seven skips are pre-existing
artifact-dependent tests.  Two independent audit executions are byte-identical
with SHA-256
`c7885c709e0159cde1786321dc61c564cb794f27dedc2a2a2b92058bee9c744f`.
The audit uses exact model evaluation and no random seeds.  It is a feasibility
diagnostic, not scientific efficacy evidence.
