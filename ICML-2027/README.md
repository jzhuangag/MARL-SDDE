# Learning value under dependent reinforcement-learning data

This independent research package extends the TSP-V3 programme without editing
the TSP manuscripts, historical experiments, or their conclusions.

The confirmed CPU upgrade controls the value and cost of acquiring dependence
information, not just participation. A public, pre-observation futility test
first decides whether calibration can help. If it can, the controller pays for
a disjoint calibration stream and selects from every q in {1,2,4,8} and spacing
in {1,4,16}. During learning, a causal scalar gain minimizes the exact quadratic
Lyapunov risk. Efficient fixed-q opponents use the same learner, optimal
spacing, and their entire message and TOTAL actor-transition budgets.

The latest independent confirmation retains 360 registered contexts and 64
fresh calibration seeds. Geometric terminal MSE is 5.2258% below the strong
public-budget-wise fixed-q control, with 99.7396% correct interior-q selection.
The controller skips acquisition in 80 contexts where its futility certificate
applies. These are population-level results, not universal cellwise dominance.

See [the confirmation report](internal/acquisition_confirmation_validation.md),
[the exact learning-gain derivation](theory/efficient_critic.md), and
[the acquisition theorem](theory/acquisition_gate.md).
The three-panel stochastic learning figure is
`results/acquisition_stochastic_validation_20261013/stochastic_value_curves.png`.

This is controlled, fixed-policy evaluation in a known-mixing stationary
Gaussian Markov reward process with a constant-feature critic. It is not a
deep-MARL return benchmark. In this efficient subclass the gain does not depend
on the target, and recorded scalar parameter staleness can be cancelled exactly.
The 360 cells therefore include target/delay controls; they are not 360 distinct
game tasks or 360 independent statistical replicates. A separately qualified
actor/critic response model is required for nonlinear MARL claims.

The pinned HARL optimizer audit confirms that simply repeating an identical
critic batch does not create a meaningful one-step learning advantage. The
[MARL transfer plan](internal/marl_response_qualification_plan.md) consequently
requires actual actor and critic response, matched optimizer work, complete
q controls, and independent development/confirmation before a GPU return study.

Run from the repository root:

```powershell
D:/anaconda/envs/ust2/python.exe -m pytest ICML-2027/tests -q
D:/anaconda/envs/ust2/python.exe ICML-2027/experiments/run_acquisition_cpu.py --confirmation-lock ICML-2027/experiments/acquisition_confirmation_lock.json --out ICML-2027/results/acquisition_confirmation_new_reproduction
```

The output path must not already exist. Scientific output is never overwritten.
The original constant-step qualification and the subsequent stronger-comparator
diagnostic remain preserved; the confirmed acquisition-aware version is the
current CPU result. Raw endpoint/certificate tables and replay directories are
retained locally, with generation commands, frozen sources, compact summaries,
and SHA-256 manifests versioned. Neither TSP-V3 nor any historical result has
been replaced.
