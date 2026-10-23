# TSP thirteen-page revision plan

## Publication thesis

Parallel Markov trajectories are not interchangeable samples: their learning value is jointly determined by spatial dependence, temporal persistence, communication cost, and delayed application.
The paper converts this observation into one resource-control problem whose decision variables are participation `q`, physical spacing `b`, and learning step `eta`.
A finite-time Lyapunov certificate designs the learning action, and a sequential information-value test determines when estimating the dependence regime improves the remaining learner enough to justify its full cost.

## Required theory additions

1. State the dual-budget usable-horizon lemma before the Lyapunov theorem.
2. Separate actual mean-square risk from its computable certificate throughout.
3. Promote the affine delayed Markov result to the named main convergence theorem and expose the role of each coefficient.
4. Add the effective-participation corollary and the resource-optimal phase interpretation.
5. Give complete algorithm pseudocode for the known-dependence and unknown-dependence cases.
6. State online arithmetic and memory complexity, including the scalar step search.
7. Retain the stochastic delay differential equation as a controlled local representation and keep the discrete recursion as the source of the finite-time guarantee.
8. Expand the stopped likelihood and threshold-sandwich proofs so the information lower and learning upper bounds use the same budgets and delay horizon.

## Required experiment additions

1. Regenerate the existing endpoint figures from frozen evidence.
2. Run a new, seed-separated CPU convergence study on the registered seven-state Markov reward process.
3. Plot squared parameter error and initial-state discounted-return estimation error against charged resource.
4. Compare the joint certificate controller with a development-selected strong fixed-participation baseline and the q=1/q=32 extremes, while allowing every baseline its own certified spacing and step.
5. Report mean curves with paired 95% intervals, normalized area under the learning curve, terminal risk, resource validity, and controller runtime.
6. Use development seeds only to select the single fixed-participation comparator; report claims only on disjoint confirmation seeds.
7. Add an ablation separating participation, spacing, and step adaptation.

## Stop rules

- Do not claim episodic policy improvement from a fixed-policy evaluation experiment.
- Label the new curve as discounted-return estimation, not realized control return.
- Do not add a result if its data or seed split cannot be reproduced.
- Preserve all raw outputs and exact source/config hashes in the internal evidence register.
