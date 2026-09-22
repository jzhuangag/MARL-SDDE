# T-029 outcome-free Blackjack static preregistration

## Status and purpose

T-029 is the theorem-facing CPU layer of a two-layer nonlinear benchmark
program.  It asks whether a standard intrinsically stochastic fixed-policy
task has enough prospective fixed-participation value to justify a sampled
Blackjack learning pilot.  It creates no sampled scientific trajectory and
does not authorize MinAtar, HPC4, or GPU execution.

Frozen configuration SHA-256:
`c2f0001a2144e93d3ed37983e3fc8baf702b0d1e114891e837f0ae6d9c289e7b`.

The external layer is MinAtar Asterix, but native Asterix observations are
partially observable and have no official uniform mixing certificate.  Any
future Asterix experiment will therefore be performance evidence only, under
a different experiment identifier and independently frozen gates.  It cannot
be used to validate the Markov mixing theorem.

## Frozen Markov task

- task: Gymnasium `Blackjack-v1`, `sab=False`, `natural=False`; disabling the
  natural bonus keeps the registered observation sufficient for the reward
  law as well as the transition law in a later value-prediction pilot;
- fixed policy: choose the threshold-20 preferred action with probability
  `0.9` and the other action with probability `0.1`;
- continuing convention: after every terminal action, draw an independent
  standard reset hand before the next transition;
- state: `(player sum, dealer showing card, usable ace)` over the exact
  reachable closure of the reset distribution;
- transition: exact enumeration of the public 13-card-value multiset, not
  Monte Carlo estimation.

The policy has stick probability at least `0.1` in every state.  Since stick
is followed by an independent reset, every row of the continuing transition
matrix minorizes `0.1` times the reset distribution.  T-029 checks both this
analytic certificate and the exact worst-state total-variation mixing stride
to tolerance `0.05`.

## Frozen participation and resource geometry

- nonlinear model for a later learning pilot: normalized 3-input MLP
  `3-32-32-1`, exactly 1,217 trainable scalar parameters;
- `q in {1,2,4,8,16,32}`;
- `rho in {0,.1,.3,.5,.7,.9}`;
- target horizons `{512,2048}`;
- delay fractions `{0,.05,.2}`;
- message cost `65536 + 4*1217*q` bytes per update;
- message ray: budget equals the cost of `H` updates at `q=4`, while the
  environment budget is nonbinding;
- environment ray: environment rounds equal `H`, while the message budget is
  nonbinding up to `q=32`.

The registered risk proxy is

```text
[rho + (1-rho)/q] / usable_updates(q).
```

It is a mechanism/value certificate, not a claim about realized neural TD
error.  A strong fixed fallback is selected separately for every
`target-horizon x budget-ray` group before any cellwise oracle comparison.

## Prospectively frozen populations

Message-binding cells are **adaptation-active** because their horizon changes
with q and can support an internal optimum.  Environment-binding cells are
**inactive** because their horizon is q-independent and the registered risk
is monotone toward `q=32`.  This partition uses only public cost algebra and
is frozen before the scan.

## Mandatory gates

1. exact transition rows, stationarity, and finite values are valid;
2. reset minorization holds and the exact TV mixing stride is at most 128;
3. cellwise oracle geometric risk improves the strong fallback by at least 5%
   over all cells;
4. strict oracle improvement occurs in at least 60% of active cells;
5. every inactive cell has the predicted `q=32` boundary oracle;
6. at least three q values are optimal somewhere in the full grid;
7. `q_environment >= q_message` for every matched cell and the message-optimal
   q is non-increasing in rho on every registered path;
8. zero sampled scientific trajectories and no outcome-tainted input.

Any failed gate stops the sampled Blackjack pilot.  Passing all gates only
authorizes a separate CPU pilot preregistration; it does not authorize GPU,
Asterix, formal seeds, or an ICML claim.

## Two-layer ICML decision rule

Blackjack alone is a theorem-aligned calibration benchmark and is not enough
for an ICML submission.  The project becomes competitively ICML-shaped only
if a later independently preregistered Asterix layer also clears a full-cost
static value gate, a nontrivial prospective neural pilot, active-cell
directionality, inactive-cell no-harm, and independent formal replication.
Until then the current fixed-participation theory and EXP-016B/018B results
remain credible but not submission-complete.
