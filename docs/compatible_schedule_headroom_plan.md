# Compatible asynchronous-update scheduling headroom plan

## Material Passport

- Artifact ID: `CAUS-HEADROOM-STATIC-20260901`
- Artifact type: deterministic queue/scheduling design ceiling
- Scientific efficacy population: none
- Formal/GPU authorization: none
- Parent decision commit: `31b74e8`
- Design hash: `06eba45b60bb1ba0b7bd7e2c43152bc09cad1910e87128ad08b2c6fd72aa1cac`

## Frozen question

Does compatible-set scheduling have material finite-horizon queueing value not
only over sequential heterogeneous-agent policy updates, but also over strong
fixed-color compatible scheduling under heterogeneous readiness?

This calculation contains no policy return, gradient sample, MSE, random seed
or learned interaction graph.  It is necessary but not sufficient for a paper.

## Design

The 48 scenarios cross agent count `{16,36,64}`, path/tree/grid/clustered
conflict graphs and homogeneous/two-tier/rotating-burst/color-skew deterministic
arrival traces.  Each trace has 240 scheduling epochs and proposals from every
actor.

Five policies receive identical arrivals and positive progress weights:

1. compatible MaxWeight (exact on paths; weighted greedy plus a best-color
   fallback elsewhere);
2. best ready color by current Lyapunov weight;
3. cyclic fixed coloring;
4. static-priority maximal compatible sets;
5. sequential maximum-weight service.

The strong compatible baseline is the best total queue-cost member of policies
2--4 for each scenario.  Cost is queue area plus 240 times terminal backlog.
No policy may serve adjacent actors or an empty queue.

## Mandatory gates

1. all 240 scenario-policy runs are finite, complete and compatible;
2. geometric dynamic/strong-compatible cost ratio at most 0.85;
3. at least 60% of scenarios obtain at least 10% cost reduction;
4. geometric completed-update ratio over sequential service at least 2.0;
5. every actor with arrivals receives service under the dynamic scheduler;
6. dynamic cost is no worse than the strong compatible baseline in any cell.

Any failed gate stops this scheduling candidate before stochastic learning.
Passing all gates authorizes only causal-cone theorem work and a separately
preregistered CPU potential-game pilot.  It does not authorize GPU work.

## Commands

```powershell
.\.venv\Scripts\python.exe -m experiments.policy_update_backpressure.compatible_schedule_headroom validate
.\.venv\Scripts\python.exe -m experiments.policy_update_backpressure.compatible_schedule_headroom run --output docs/compatible_schedule_headroom_results.json
```

The run command is forbidden until this plan, manifest, runner and tests are
committed and pushed.

