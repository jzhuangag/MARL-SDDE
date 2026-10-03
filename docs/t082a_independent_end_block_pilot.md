# T-082A independent causal end-block pilot preregistration

T-082A is the first independently seeded test of the causal block-end
primal-dual controller.  T-081 is declared as tainted design information.  No
controller constant, primary population, comparator, or threshold changes
after T-081.  The 64 pilot seeds are deterministic SHA-256-derived values with
zero overlap with T-071A seeds.

The primary 96-cell identifiable class and all 336 boundary/control cells are
unchanged.  The comparator is the per-cell T-079 continuous static graph run
under identical block-end timing and common random numbers.  Every observation
is a learning update; fingerprints consume messages but no extra environment
transition.

The scientific artifacts are `endpoints.csv`, `cells.csv`, and `summary.json`.
Runtime is stored separately in `execution.json`, so P13 has an unambiguous
byte-identity target.  Any mandatory failure stops reproduction interpretation,
formal registration, nonlinear benchmarks, and GPU work.  T-082A is local CPU
only.
