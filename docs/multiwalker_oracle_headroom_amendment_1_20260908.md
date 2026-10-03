## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: prospective methodological Amendment
- Origin Date: 2026-09-08
- Verification Status: FROZEN BEFORE AMENDED OUTCOME GENERATION
- Version Label: multiwalker_oracle_headroom_amendment_1_v1

# Amendment 1: exact prefix-budget policy-profile oracle

## Reason for amendment

The immutable v1 development result failed D6 (`11/16` rather than `12/16`
active cells) and does not authorize critic fitting. It also demonstrated that
the object labelled `oracle_h8` can be beaten by feasible one-edge online
policies under the tight prefix budget. The reason is mathematical: v1 greedily
maximizes the current eight-step branch value and does not optimize token
allocation or persistent cache states over all 40 launches.

This Amendment corrects only that upper-bound diagnostic. It does not change
the environment, public state prefixes, actor initialization or version path,
counterfactual horizon, discount, candidate graph, costs, budgets, seeds,
strong baseline outcomes, or any performance threshold. Original v1 files and
hashes remain immutable.

## Exact finite-horizon construction

For recipient `i`, a local cache state records the last owner-launch index at
which each physical neighbor was refreshed. Given this state and a current
null-or-one-edge action, the actual Box2D branch value is recomputed by exact
prefix replay. The next cache state is deterministic. Dynamic programming
enumerates every local action sequence and retains the best return for each
local edge-cost sequence.

The five owner problems are coupled only by the global communication budget.
One local sequence is selected for each owner by a multiple-choice binary
program with all 40 prefix constraints,

\[
\max_{x}\sum_{i,s}x_{i,s}Y_{i,s},\qquad
\sum_sx_{i,s}=1,
\]

\[
\sum_{i,s}x_{i,s}C_{i,s}(t)
\le \lfloor b(t+1)\rfloor,
\qquad t=0,\ldots,39.
\]

Every binary program must terminate with the certified optimal status and
zero reported MIP gap. The no-refresh local sequences must reproduce the
immutable v1 no-refresh totals within `1e-10`. The selected solution must have
five owner sequences and nonpositive excess at every prefix.

The exact oracle uses the method's declared null-or-one-edge action set. It
contains the action sequences of the one-edge age, mismatch, fixed, random,
round-robin, and one-step-oracle baselines. The complete-neighbor baseline has
a broader per-launch action and remains in the strong envelope; if it beats
the exact one-edge oracle, the proposed action class fails the headroom gate.

## Frozen data and decision rule

The Amendment reuses development seeds `95700--95707` only because it repairs
the diagnostic on the same immutable design data. It is not independent
evidence. It verifies the original `rows.json` SHA-256
`AD79327D849C820C1062C88BB1502A2BB5756064D7EA23EBCFFE6203662B2D42`
before reading baseline aggregates.

The original D4--D7 thresholds remain unchanged:

- active exact-oracle gain over no refresh must be positive;
- aggregate recovery gap over the strong envelope must be at least 10%;
- strict active direction must be at least 75%;
- median active normalized absolute headroom must be at least 0.1%.

Four additional validity gates require finite results, exact MIP optimality,
no-refresh replay equality, and all prefix budgets. Any failed gate stops
Multiwalker before critic fitting. A pass authorizes only an independent-seed
oracle confirmation preregistration. It does not authorize controller fitting,
formal claims, GPU, or HPC4.

To reduce wall time without changing the scientific computation, the eight
seeds may be executed as four isolated two-seed CPU chunks. Chunk output is
merged only after verifying disjoint complete seed sets; the merged analyzer
recomputes every gate from exact rows.
