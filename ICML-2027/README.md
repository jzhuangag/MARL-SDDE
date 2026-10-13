# Learning value under dependent reinforcement-learning data

This independent research package extends the TSP-V3 programme without editing
the TSP manuscripts, historical experiments, or their conclusions.

The first qualification is a data-driven finite-budget controller for a
Gaussian Markov reward process with a constant-feature TD critic. It estimates
the initial signal and spatial noise moments from a disjoint, fully charged
calibration stream, then evaluates the exact delayed discrete-time critic risk
over the complete participation catalogue. The true context parameters are
available only to the evaluator.

This is a controlled RL policy-evaluation experiment, not a deep-MARL return
benchmark. In particular, passing it does not establish why PPO on MaMuJoCo has
a shared-regime optimum at q=2. A separately qualified learning-response model
for the actual critic/optimizer is required before that claim can be made.

Run from the repository root:

```powershell
D:/anaconda/envs/ust2/python.exe -m pytest ICML-2027/tests -q
D:/anaconda/envs/ust2/python.exe ICML-2027/experiments/run_cpu.py --phase development --out ICML-2027/results/cpu_development_20261013
```

The preregistration must be committed before a scientific run. Confirmation
requires every development gate to pass and an independent confirmation lock.
The original TSP-V3 positive results remain available in their own package.
