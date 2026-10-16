# Pinned actor replication: verified operational result

The design and source were frozen at `778ec50` before execution. Every one of
the 32 CPU payloads is finite and satisfies the original duplication tolerance.
An independent rerun in a disjoint output directory produces byte-identical
audit JSON. No environment, evaluation return, or scientific task outcome was
used, and no runtime or source dependency was modified.

- HARL: clean b1af98b0dbab72a2eee9d160751cd09aedbb8ce2.
- Torch: 2.6.0+cpu, one thread.
- Two seeds, Discrete(5)/Box(3), tanh/relu, q={1,2,4,8}.
- Five matched full-batch MAPPO updates per payload; exact repeated buffers.
- Maximum actor parameter difference from q=1: 3.2782554626464844e-7.
- Prespecified tolerance: 1e-5.
- Policy KL is computed from distributions on the same fixed buffer; it is
  neither a return certificate nor a held-out task-generalization measure.
- Model-row work grows with q; optimizer steps remain five. Neither synthetic
  quantity is labelled an environment-transition count.

Command, from the repository root:

```powershell
D:/anaconda/envs/ust2/python.exe -c "import sys,runpy;sys.path.append(r'D:/anaconda/Lib/site-packages');sys.argv=['audit_harl_actor_replication','--out',r'ICML-2027/results/harl_actor_replication_audit_new_reproduction'];runpy.run_path(r'ICML-2027/experiments/audit_harl_actor_replication.py',run_name='__main__')"
```

The base-environment path supplies its already installed pure-Python PyYAML;
Torch and the execution runtime remain ust2. Existing outputs must not be
overwritten. The result establishes an implementation invariance, not that
additional independent data or policy-dependent trajectories have no value.

The operational prerequisite now remaining is genuine independent-data and
evolving-policy actor/critic response, followed by the score-to-risk transfer
interface. No new GPU or MARL performance conclusion is inferred here.
