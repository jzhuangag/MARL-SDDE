# SDDE research experiment index

## Active direction

The current feasibility question is whether persistent cross-agent dependence
and heterogeneous delay create a useful participation-control problem beyond
the standard linear-speedup analysis.

## Experiment records

1. [`experiment_001_dependence_delay_go_nogo.md`](experiment_001_dependence_delay_go_nogo.md)
   records the registered baseline, the exact linear model, the four decision
   gates, EXP-001 results, and the EXP-002 common-factor alignment sensitivity.
2. [`experiment_003_transient_stationary_crossover.md`](experiment_003_transient_stationary_crossover.md)
   records the post-baseline finding that the best fixed-step participation
   level changes between transient and stationary regimes.
3. [`reproducibility_exp001.md`](reproducibility_exp001.md) records analytic
   implementation checks, Monte Carlo agreement, and the independent rerun.
4. [`experiment_004_stagewise_controller.md`](experiment_004_stagewise_controller.md)
   records the predictable low-complexity joint step–participation experiment.
   The controller detected the dependence shift but did not improve MSE beyond
   dependence-aware step-size adaptation.
5. [`validation_exp004.md`](validation_exp004.md) records the paired-bootstrap
   interpretation, 11/11 statistical fallacy scan, and incomplete reproduction
   status.
6. [`experiment_005a_budget_participation_surface.md`](experiment_005a_budget_participation_surface.md)
   records the pre-registered resource-matched participation surface. Its five
   gates passed: independent-noise cells selected all 32 agents, whereas the
   high-correlation cells selected one agent under the primary message budget.
7. [`validation_exp005a.md`](validation_exp005a.md) records the numerical
   validation, 11/11 fallacy scan, scope warnings, and byte-identical rerun.
8. [`experiment_005b_online_probe_controller.md`](experiment_005b_online_probe_controller.md)
   records the charged online controller. It learned the correct participation
   direction and beat all-agent control, but failed to beat fixed \(q=1\).
9. [`validation_exp005b.md`](validation_exp005b.md) records the paired-bootstrap
   audit, failed overall gate, 11/11 fallacy scan, and exact reproduction.
10. [`experiment_005c_sparse_dynamic_controller.md`](experiment_005c_sparse_dynamic_controller.md)
    records the final sparse, nonstationary go/no-go design, its initial
    timeout, authorized execution v2, and failed registered decision.
11. [`validation_exp005c_timeout.md`](validation_exp005c_timeout.md) preserves
    the initial `CANNOT_VERIFY` audit.
12. [`validation_exp005c.md`](validation_exp005c.md) records the completed
    64-seed result, exact reproduction, oracle-gate mismatch, and 11/11 fallacy
    scan.
13. [`experiment_006a_oracle_phase_diagram.md`](experiment_006a_oracle_phase_diagram.md)
    preregisters the oracle-first participation phase diagram used to decide
    whether agent-number adaptation remains a viable main direction.
14. [`validation_exp006a.md`](validation_exp006a.md) records the 9,720-cell
    scan, exact reproduction, failed delay-relevance gate, ten actionable
    correlation/state regions, and 11/11 fallacy scan.
15. [`experiment_006b_state_correlation_controller.md`](experiment_006b_state_correlation_controller.md)
    preregisters an observable, low-complexity state-and-correlation controller
    evaluated only on the oracle-supported domain.
16. [`validation_exp006b.md`](validation_exp006b.md) records the exact
    reproduction, failed 2/6 overall result, and state-proxy root cause.
17. [`experiment_006c_lyapunov_state_controller.md`](experiment_006c_lyapunov_state_controller.md)
    preregisters an independent-seed test of a scalar Lyapunov-surrogate
    controller that replaces the failed raw gradient-magnitude state proxy.
18. [`validation_exp006c.md`](validation_exp006c.md) records the 4/7 failed
    result, exact reproduction, improved but conservative state surrogate, and
    the information mismatch with a clairvoyant realized-state oracle.
19. [`theory_program_icml2027.md`](theory_program_icml2027.md) narrows the
    proposed ICML contribution to correlation-limited speedup, finite-budget
    participation phase transitions, and an SDDE-to-discrete-time proof route.
20. [`experiment_007a_linear_td_correlation.md`](experiment_007a_linear_td_correlation.md)
    preregisters a delayed linear-TD test of effective participation and the
    finite-budget agent-count phase transition under shared Markov trajectories.
21. [`validation_exp007a.md`](validation_exp007a.md) records the formal 6/6
    pass, exact reproduction, exchangeable-LRV fit, held-out-half diagnostic,
    and the critical finding that the delay gate passed only by equality.
22. [`experiment_007b_delay_stability.md`](experiment_007b_delay_stability.md)
    preregisters an exact spectral-boundary and Monte Carlo stress test for the
    delayed TD/SDDE stability layer.
23. [`validation_exp007b.md`](validation_exp007b.md) records the reproducible
    4/6 failure, the mean-versus-mean-square stability gap, and the newly
    exposed correlation--delay interaction through random TD Jacobians.
24. [`experiment_007c_joint_mean_square_step.md`](experiment_007c_joint_mean_square_step.md)
    preregisters a scalar parallel-sum step rule combining the delayed mean
    boundary with the correlation-dependent Jacobian second moment.
25. [`validation_exp007c.md`](validation_exp007c.md) preserves its formal 3/6
    failure and shows that catastrophic crossing was too coarse to identify
    finite but noncontracting blind policies.
26. [`experiment_007d_joint_ms_confirmation.md`](experiment_007d_joint_ms_confirmation.md)
    preregisters an unchanged-rule confirmation with 64 new seeds and
    mean-square confidence limits.
27. [`validation_exp007d.md`](validation_exp007d.md) records the 7/7 pass,
    9,216-run result, exact reproduction, correlation/agent-count effect, and
    nonvacuous joint safe-step boundary.
28. [`proof_program_joint_ms.md`](proof_program_joint_ms.md) proves the exact
    exchangeable Jacobian identity and no-delay mean-square contraction,
    specifies the delayed Markov and SDDE proof obligations, and gives the
    \(O(d)\)-memory scalar estimator route.
29. [`experiment_008a_exact_lifted_boundary.md`](experiment_008a_exact_lifted_boundary.md)
    preregisters a deterministic comparison between the scalar joint rule and
    the exact heterogeneous-delay covariance operator.
30. [`validation_exp008a.md`](validation_exp008a.md) records the reproducible
    4/7 result: scalar safety passes in all cells, exact agent-count saturation
    is confirmed, and three overly strong tightness/dominance gates fail.
31. [`experiment_008b_markov_jump_boundary.md`](experiment_008b_markov_jump_boundary.md)
    and [`validation_exp008b.md`](validation_exp008b.md) validate the exact
    Markov-jump operator and retain a weak-persistence negative control.
32. [`experiment_008c_expanding_markov_td.md`](experiment_008c_expanding_markov_td.md)
    and [`validation_exp008c.md`](validation_exp008c.md) show that a persistent
    locally expanding TD regime collapses the boundary and defeats an i.i.d.
    step rule.
33. [`experiment_008d_decorrelated_theorem.md`](experiment_008d_decorrelated_theorem.md)
    through [`validation_exp008e.md`](validation_exp008e.md) derive and verify
    the sharp predictable-decorrelation/RMS-delay theorem.
34. [`experiment_009a_predictable_mixing_controller.md`](experiment_009a_predictable_mixing_controller.md)
    through [`validation_exp009c.md`](validation_exp009c.md) audit static
    high-confidence \((q,b,\eta)\) controllers, confirming safety and
    correlation response but rejecting uniform near-oracle efficiency.
35. [`experiment_009d_progressive_anytime_controller.md`](experiment_009d_progressive_anytime_controller.md)
    and [`validation_exp009d.md`](validation_exp009d.md) validate time-uniform
    progressive safety and reject a uniform oracle ratio as mixing vanishes.
36. [`experiment_010a_multistate_certificate_transfer.md`](experiment_010a_multistate_certificate_transfer.md)
    and [`validation_exp010a.md`](validation_exp010a.md) transfer the sharp
    homogeneous certificate to seven-state vector TD, while isolating the
    still-open affine Markov finite-time bound.
37. [`experiment_010b_affine_finite_time_certificate.md`](experiment_010b_affine_finite_time_certificate.md)
    and [`validation_exp010b.md`](validation_exp010b.md) close and calibrate
    the finite-gap affine Markov-TD bound without conditional-centering or
    Jacobian--innovation orthogonality assumptions.
38. [`experiment_011a_correlation_minimax_phase.md`](experiment_011a_correlation_minimax_phase.md)
    and [`validation_exp011a.md`](validation_exp011a.md) prove and audit the
    exact correlation-limited minimax speedup ceiling and the
    resource-optimal participation phase.

## Code and canonical outputs

- Source and usage:
  `experiments/dependence_delay_linear/README.md`
- Registered baseline:
  `experiments/dependence_delay_linear/results/baseline/`
- Common-factor alignment sensitivity:
  `experiments/dependence_delay_linear/results/server_time_sensitivity/`
- Transient/stationary crossover:
  `experiments/dependence_delay_linear/results/crossover/`
- Stagewise controller:
  `experiments/dependence_delay_linear/results/stagewise/`
- Budget-matched participation:
  `experiments/dependence_delay_linear/results/budget_participation/`
- Online probe-charging controller:
  `experiments/dependence_delay_linear/results/online_participation/`

The `results/smoke/` directory is an implementation smoke test and must not be
used as scientific evidence. The same-seed reproduction directory is retained
locally but excluded from the public repository because it duplicates the
canonical outputs; its verification result is recorded in
`reproducibility_exp001.md`.

## Current decision

The project has strong evidence for correlation-limited speedup and
dependence-aware scalar step-size tuning. EXP-004 showed that a controller can
reduce its selected count from 32 to 4 after a common-noise shift, but this did
not improve MSE over retaining all agents and adapting only the step size.
EXP-005A subsequently confirmed that participation can be a strong
resource-control mechanism: the resource-matched optimum changes from all
agents under independent noise to one agent under strong common noise. The
next experiment tested an online controller that observes only participating
agents and charges every exploration probe. It correctly reduced participation
under correlated noise but its 18% full-probe cost prevented it from beating
the best fixed-\(q\) policy. EXP-005C reduced exploration to 2.4% and introduced
within-run regime shifts. Its authorized optimized execution was exactly
reproduced but failed three scientific gates. The audit also showed that the
piecewise oracle retained median \(q=32\) in every regime, so the current test
rejects the controller without resolving the broader participation hypothesis.
EXP-006A resolves the oracle surface: correlation/state adaptation has ten
contiguous actionable regions, but delay changes pointwise optimal \(q\) in
only 5.31% of groups. The combined gate fails, while a narrower
correlation/state controller remains mechanistically supported.
EXP-006B then rejected a raw gradient-magnitude state proxy. EXP-006C replaced
it with a scalar Lyapunov recursion and obtained clear improvements over both
raw-state and correlation-only controllers, but still lost to fixed \(q=4\)
and a clairvoyant realized-state oracle. The online state-controller line is
therefore stopped. The retained ICML direction is a theorem-first account of
how cross-agent Markov correlation invalidates linear speedup and induces an
optimal finite participation level under delay and communication budgets.
EXP-007A validates this narrower mechanism in linear TD: \(N_{\rm eff}(32)\)
falls from 30.996 under independent paths to 1.111 at correlation 0.9, while
the long-budget optimal count falls from 16 to 1. The formal experiment passes
all gates and reproduces exactly, but delay is empirically inactive; the next
stage must separately stress the SDDE stability mechanism.
EXP-007B performs that stress test. It activates a strong mean delay boundary
but rejects mean-spectral step-size control: high cross-agent correlation can
make stochastic TD diverge even below the exact mean boundary. The retained
theory must therefore establish correlation-aware mean-square delayed
stability, not only mean convergence or additive-noise diffusion.
EXP-007C introduces an analytic aggregate-Jacobian second moment and a scalar
joint correlation--delay step. Its formal crossing-based verdict fails because
finite blind policies can remain far above the initial error without crossing
the catastrophic threshold. EXP-007D corrects only the endpoint, not the
algorithm, and confirms the unchanged rule on 64 new seeds. All seven gates
pass: the largest 99% upper mean-error limit is 0.649, the weakest
correlation-aware paired advantage is 8.19 at the 99% lower limit, and the
joint step is at least 54.5% of the useful fixed-grid boundary. The active
mainline is now a theorem for correlation-limited mean-square stability under
delay, followed by an online scalar estimator and nonlinear breadth.
EXP-008B--E complete the first theorem layer. The exact finite-state
Markov-jump operator distinguishes benign persistence from persistent local
expansion, and the sharp decorrelated theorem passes all exact safety and
envelope gates while using only \(K_q\), a mixing certificate, and RMS delay.
EXP-009A--C show that a one-shot 99% mixing certificate is safe and selects
much smaller \(q\) under high correlation, but cannot be uniformly near-oracle
at \(p=0.98,D=2\). The next algorithmic step is a progressive anytime
certificate that reuses transitions observed between updates; the static
pilot must not be described as near-oracle.
EXP-009D completes that progressive audit: simultaneous coverage and exact
safety pass, and high-persistence gaps refine across blocks, but the worst
oracle ratio remains 7.57. The manuscript must expose the resulting
confidence/mixing penalty instead of promising a uniform near-oracle bound.
EXP-010A then validates multistate transfer: all homogeneous certificates are
strict, correlation changes \(q\) in every matched cell, mixing changes the
median gap by \(18.36\times\), and all artifacts reproduce. The controller is
never worse than the better endpoint in mean, but its strict endpoint gate
fails by equality in five cells. Generic affine Markov-TD finite-time risk
remains a proof obligation.
EXP-010B closes that finite-gap obligation. The affine theorem passes every
registered validity, nonvacuity, response, calibration, and informativeness
gate; its median bound/mean ratio is 20.77 and all eight artifacts reproduce.
EXP-011A then closes the lower-bound obligation. In a Gaussian
one-step-mixing subclass, the exact speedup is
\(q/[1+(q-1)\rho]\), and no predictable adaptive participation rule can beat
the best Fisher-information-per-cost action under a pathwise budget. All nine
registered gates pass and all five core artifacts reproduce byte-for-byte.
The next blocking step is a simultaneous correlation/mixing/curvature
confidence theorem and end-to-end predictable controller; the unthinned
Poisson-equation and SDDE approximation extensions are non-blocking.
