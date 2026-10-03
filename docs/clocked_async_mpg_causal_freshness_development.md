# Causal Lyapunov freshness development

## Outcome

The causal resource-debt policy captures a substantial part of the equal-cost
freshness oracle headroom over a broad, public tradeoff curve.  This closes the
mechanism-development question positively, but it is not independent evidence:
the same Markov risk paths used for the oracle feasibility scan were used to
select `V=4`.  A clean CPU confirmation must use new seeds and unchanged
selection rules.

## Controller tested

For a refresh with conditional MSE reduction `R_k`, unit interaction cost, and
average refresh allowance `alpha`, the resource queue is

```
Z_{k+1} = [Z_k + u_k-alpha]_+.
```

The causal action is

```
u_k = 1  iff  V R_k > Z_k
```

and remaining finite-horizon tokens may veto a refresh.  The action loop reads
only the current `R_k`, current queue, and remaining budget.  For each realized
schedule, it is compared with the best cyclic phase of an evenly spaced
periodic schedule using the same actual refresh count, as well as the
same-count noncausal oracle.

## Full tradeoff curve

| `V` | Geometric LSFF/periodic risk | Dynamic rows better | Median oracle headroom captured | Mean budget utilization |
|---:|---:|---:|---:|---:|
| .25 | .929884 | 77.12% | 10.57% | 100% |
| .5 | .902673 | 88.41% | 30.43% | 100% |
| 1 | .880512 | 93.80% | 55.00% | 100% |
| 2 | .862869 | 95.09% | 75.37% | 100% |
| 4 | .850166 | 95.49% | 85.00% | 100% |

The predeclared development selector chooses the smallest geometric risk ratio
among tradeoffs with at least 90% mean utilization, hence `V=4`.  The monotone
improvement across all five values is stronger evidence than an isolated
successful tuning point.  Still, `V=4` is development-tainted and cannot be
evaluated on these paths again as confirmation.

## Runtime and equivalence audit

The initial research-grid implementation allocated several arrays and frozen
dataclasses at each of 53 million events and repeatedly evaluated the strong
periodic comparator.  It completed correctly in 3154.72 seconds.  A scalar
implementation of the identical recursion completed in 56.39 seconds, a
55.95-fold engineering speedup.  Unit tests establish fieldwise equivalence,
and the complete 103,680-row output and summary are byte-identical:

- `rows.csv`:
  `7ac37f0a1d68f70fa8b26ed17cafcb050fd17d03ac34e508bbccb97191e0ad49`
- `summary.json`:
  `d58293536e0cce068241d088261082fa1ba101e46ee6ff6e6ed47b5a0d3e9c5a`

The optimized scan runtime is not the online algorithm's complexity claim.
The online action itself is one scalar comparison and one scalar queue update
per event, in addition to O(d) gradient fusion.

## Interpretation and next gate

This result demonstrates that causality and budget enforcement do not destroy
the analytic oracle opportunity.  It does not yet demonstrate that a MARL
learner can estimate `R_k` conservatively or that reduced conditional gradient
MSE improves return under refresh service cost.

Before any GPU work, the following CPU stages remain mandatory:

1. freeze `V=4` and run new Markov risk paths to confirm the full causal result;
2. implement an arrival-fresh MPE measurement with exact actor-transition and
   service-time charges;
3. validate observable birth-bias and variance certificates, never/always and
   fixed-period baselines, return, tail return, and wall-clock/transition Pareto
   behavior on new development and confirmation seeds;
4. close the Markov-estimator and wall-clock theorem interfaces used by those
   measurements.

No formal, GPU, or HPC4 experiment is authorized by this development scan.
