# Thirteen-page reference-structure audit

The reference PDF is used only as a structural and length-calibration source.
Its read-integrity preflight passed for all 13 pages with SHA-256
`1bddb08b16b58667a2a0bb7cebe615e5aa1ae6c5f2e884c05c3682717d20431c`.

## Observed allocation

| Pages | Function |
|---|---|
| 1--2 | Abstract, problem definition, challenge relative to the single-agent case, progressive related-work gap, proposed mechanism, four contributions, organization |
| 2--4 | Formal problem, state/policy/return definitions, classical baselines, motivating geometry, stochastic recursion |
| 4--6 | Lyapunov construction, conditional drift model, closed-form controller, algorithm summary |
| 7--9 | Finite-time and policy-space theory, practical implementation |
| 9--11 | Experiments from controlled mechanism to tabular/neural learning curves and finite-trajectory validation |
| 11--13 | Conclusion, proof appendices, environment specification, references |

The reference therefore devotes roughly nine pages to nonexperimental content before presenting two pages of experiments.
Figures are used as arguments: one explains the mechanism and two show convergence and policy performance.

## Adopted organization principle

The TSP paper will follow the same argumentative cadence without copying subject matter or prose:

1. Define the collaborative Markov policy-evaluation system, including what a state, trajectory, participant, retained sample, and delayed update mean.
2. Explain why correlated agents, temporal persistence, finite resources, and delayed application create a coupled decision absent from single-chain temporal-difference learning.
3. Organize prior approaches into Markov stochastic approximation, distributed/federated learning, delayed optimization, and controlled sensing; expose the missing joint finite-budget decision progressively.
4. State exactly what is optimized: the Lyapunov certificate selects `(q,b,eta)`, while the information layer decides whether dependence identification is worth buying.
5. Present contributions only after the problem, gap, and proposed mechanism are concrete.
6. Allocate the experiment section to mechanism, calibration, learning curves, resource sensitivity, and computation rather than a list of endpoint studies.

## Page target for the revision

| Section | Target pages |
|---|---:|
| Abstract and Introduction | 1.8--2.0 |
| Model and finite-budget objective | 1.5--1.8 |
| Dependence geometry | 1.0--1.2 |
| Lyapunov joint control | 2.2--2.5 |
| Learning-aware adaptation | 2.0--2.3 |
| Implementation and SDDE interpretation | 1.0--1.3 |
| Experiments | 2.0--2.5 |
| Conclusion, appendices, references | 1.5--2.0 |

The resulting target is 13 IEEE double-column pages with 9--11 pages of nonexperimental content.
