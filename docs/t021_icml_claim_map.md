# T-021 ICML claim map

## Main decision

The project remains potentially ICML-worthy only after removing the failed
online-controller claim. The defensible mainline is:

> **Beyond Linear Speedup: Correlation- and Delay-Limited Participation under
> Multi-Agent Markov Data.**

鈥淏eyond linear speedup鈥?means going beyond the customary independent-agent
linear-speedup assumption: the attainable factor is
`q/[1+(q-1)rho]`, delay and mixing reduce usable progress, and dual budgets can
make an interior fixed participation level optimal. It does not mean a
superlinear rate.

## Claims permitted in the main paper

1. A delayed, decorrelated Markov-TD recursion has a rigorous mean-square
   contraction and affine finite-time error bound (Theorems 3 and 4 in
   `proof_program_joint_ms.md`).
2. In the Gaussian common-factor subclass, correlation gives the exact
   effective speedup `q/[1+(q-1)rho]`, and fixed message/environment budgets
   yield an explicit participation threshold and lower bound (Theorem 5).
3. Without a separated mixing certificate, unrestricted unknown mixing is
   not uniformly identifiable at finite budget; this is a genuine negative
   boundary, not a technical omission.
4. At a frozen nonlinear parameter, averaging agent TD gradients obeys the
   same correlation-limited variance identity (Theorem 9). A nonlinear
   experiment may validate this mechanism, but cannot be advertised as a
   general nonlinear convergence theorem.
5. SDDE and Lyapunov-Krasovskii analysis explain the continuous-time
   delay/stability geometry. The discrete theorem remains the convergence
   guarantee until an SDDE-to-discrete error theorem is proved.

## Claims excluded

- no online correlation/mixing-adaptive controller as a headline algorithm;
- no universal no-harm or near-oracle claim;
- no unrestricted unknown-mixing positive result;
- no unthinned Markov-TD finite-time theorem;
- no nonlinear convergence theorem derived from the fixed-parameter variance
  identity;
- no claim that the two EXP-017A pilot seeds constitute formal evidence.

## Why the mainline can still be ICML-level

T-020 shows that the existing nonlinear benchmark has almost no adaptive
controller value (0.3846% optimistic oracle gain), but it does not contradict
the correlation-limited fixed-q phenomenon. T-019 displays all four fixed-q
optima and the predicted rho, delay, and budget directions in 22/24, 21/24,
and 36/36 descriptive paths. The formal contribution is therefore the theorem
package and its mechanistic validation, not controller performance.

This is a narrower paper than the earlier adaptive title, but also a more
coherent one: one exact speedup law, one delayed finite-time route, one
finite-budget phase transition, one impossibility boundary, and experiments
that directly measure those objects.

## Remaining ICML gates

The mainline is **conditionally viable**, not submission-ready. It still needs
an outcome-free preregistration and formal nonlinear fixed-q evidence with a
direct gradient covariance endpoint. A learning-curve-only experiment is not
sufficient. If the direct variance identity or at least one terminal
correlation/delay interaction fails prospectively, the nonlinear bridge must
be removed and the venue target reassessed.

