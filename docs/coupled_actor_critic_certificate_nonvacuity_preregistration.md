# Preregistration: optimistic certificate nonvacuity audit

Date: 2026-09-05.

Status: frozen before exact static outcomes.  This audit samples no trajectory,
uses no prior learning result, and authorizes neither efficacy nor GPU work.

## Question

Can the observable all-packet empirical-Bernstein shield admit a nonzero actor
step at a practical fully charged packet size?  A formally valid theorem is
not enough if its bounded-range term makes every packet uncertified.

The audit intentionally favors the proposed controller:

- coordinate sample variance is set to exactly zero, its best possible value;
- policy and critic version displacement are zero;
- the actor packet uses the exact value critic, so its critic-bias term is zero;
- only the unavoidable bounded-range term in the empirical-Bernstein radius
  remains;
- critic tracking starts at a public Euclidean radius `c=0.5`.

For each seeded two-agent, two-state, two-action finite Markov game, horizon,
discount and owner, the exact finite-horizon gradient supplies the largest
possible signal available at that snapshot.  The actor is nonvacuous when its
range-only radius is smaller than that gradient norm.  The critic is
nonvacuous when its range-only radius is smaller than `mu_c*c`.  The minimum
trajectory-grid point satisfying both is multiplied by the horizon to obtain
fully charged transitions.

Because the variance is optimistically zero, failure is decisive for this
specific all-packet high-probability interface: realized empirical variance
can only increase the radius.  Success would be necessary but not sufficient
for a useful algorithm.

## Frozen population and gates

The machine-readable population, thresholds and stopping rule are in
`coupled_actor_critic_certificate_nonvacuity_gates.json`.  There are 576 owner
cases.  The main practical cap is 8,192 transitions per returned packet; an
extended 16,384-transition check is reported separately.  N1--N7 are
mandatory.  No threshold may be changed after the independent preregistration
commit.

The audit must be executed twice in isolated output directories.  If any gate
fails, the result is recorded and the high-probability shield is stopped.  A
possible successor may use expectation-level predictable sample splitting,
but it must receive a new theory and experiment contract rather than inheriting
these gates.
