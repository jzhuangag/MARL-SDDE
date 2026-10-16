# Original constant-step CPU qualification confirmation

- Status: original frozen gates and five-file scientific byte replay pass.
- Confirmation lock committed at `01d5a00` before any reserved seed was run.
- Seeds: 48000001--48000064; 360 contexts; 184,320 method rows.
- Full q={1,2,4,8}, spacing={1,4,16}, and stable finite-step catalogue retained.
- Exact conditional terminal-MSE ratio to the strongest global-single-q
  comparator: 0.824741582, paired-seed 95% interval [0.824494740,0.824988498].
- Ratio to the same-cost oracle: 1.002860673.
- Interior-qualified contexts: 106; correct q selection: 98.3343%.
- Joint parameter coverage: 99.0625%.
- Fresh-training Monte Carlo diagnostics: 189 checkpoints, all pass.

The one-step-then-commit myopic comparator is an ablation, not a strong
continuously replanning scheduler. Its very large gap is not a headline
algorithm-superiority claim. The fixed comparator class here is global-single-q
constant-step TD with privileged spacing/step choices, not the later
budget-wise efficient learner.

The stronger efficient/acquisition study is the current primary qualification;
this original result is preserved rather than substituted for it. Its endpoint
is exact risk over learning noise after random calibration, not 184,320 sampled
training trajectories or nonlinear MARL return runs.

| Scientific file | SHA-256 |
|---|---|
| endpoints.csv | 8c828b6441d0c6786bd2c35c94de37016b2f4069aef16a0e1b826f5403d12157 |
| certificates.csv | 6f38fbe3e842084c92c36c720c274b82ddb1715c128cc0c5a6e2e77fb0ab6988 |
| curves.csv | e1f034bfbca228dd2297dffa2b8f162043084d2b68529431f395f245762ec981 |
| monte_carlo.csv | 762fe22469ab7f217d36443f7d836c1fb63ce84cd7f18e953f80d28979966091 |
| summary.json | 39f39446b36f4f2afe8703200ad853fa199b7d1e33c88ea0efe2e2a1594566db |

Scientific files reproduce exactly in a separate directory. Runtime seconds
and execution provenance are recorded separately, not required to be identical.
All prior manuscript and scientific artifacts remain unchanged.
