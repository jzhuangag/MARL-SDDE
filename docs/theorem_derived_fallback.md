# T-016: theorem-derived fallback and known-mixing upper bound

## Finite-budget risk objects

Fix public \(\lambda\) (or a separately valid upper mixing bound), a finite
stable action catalogue, and dual budget \(B=(B_{\rm msg},B_{\rm env})\).
Let \(R_j(a;B,D)\) be the proved finite-budget risk of committing to action
\(a\) on instance \(j\), using exactly the existing affine/delayed or
registered Gaussian risk certificate. Define

\[
R_j^\star(B,D)=\min_aR_j(a;B,D),\qquad
R_j^{\rm all}(B,D)=R_j(a_{\rm all};B,D).
\]

For probe design \(d=(q,b,n)\), charge
\(C_d=(n(h+q),nb)\). Let \(B_d\) be the post-probe budget before latency and
\(B_d^D\) the budget/horizon after charging \(D\). Infeasible budgets have
infinite risk.

For directional error target \(\delta\), choose the first
\(n=n_{\rm UB}^{\rm dir}(q,b,\delta)\) for which the exact Bhattacharyya
distance satisfies

\[
{\cal B}_n(q,b)\ge\log(1/\delta).
\]

The equal-prior likelihood-ratio test then obeys
\((\alpha_0+\alpha_1)/2\le\frac12e^{-{\cal B}_n}\le\delta/2\), hence each
nonnegative directional error is at most \(\delta\). This threshold is
deliberately stricter than EXP-015A's frozen average-error threshold
\(\log(1/(2\delta))\); the pilot is not retroactively changed. A sequential
likelihood-ratio test with two Ville boundaries is another directionally
valid construction, but no sample-complexity matching claim for it is used
here.

## Explicit safety slack

For \(j\in\{0,1\}\), define

\[
\begin{aligned}
C^{\rm probe}_j(d)
&=R_j^{\rm all}(B_d,0)-R_j^{\rm all}(B,0),\\
C^{D}_j(d)
&=R_j^{\rm all}(B_d^D,D)-R_j^{\rm all}(B_d,0),\\
G_j(d)
&=R_j^{\rm all}(B_d^D,D)-R_j^\star(B_d^D,D),\\
W_j(d)
&=\max_{k\ne j}
[R_j(a_k^\star;B_d^D,D)-R_j^\star(B_d^D,D)]_+ .
\end{aligned}
\]

All differences are clipped below at zero when monotonicity is not already
part of the risk theorem. The advertised safety-slack function is

\[
\boxed{
\epsilon_{\rm safe}(B_{\rm msg},B_{\rm env},D,\delta,
\theta_0,\theta_1,\lambda,h,Q;d)
=
\left[
C^{\rm probe}_0(d)+C^D_0(d)-G_0(d)+\delta W_0(d)
\right]_+ .
}
\tag{7}
\]

Every dependence in (7) is computable from the exact probe threshold and
finite-budget risk catalogue; \(\theta\)-gap and \(\lambda^b\) enter
\(n_{\rm UB}\), \(q,h,b\) enter both costs, and \(D\) enters usable updates.

## Certified fallback rule

Explore with a feasible design only if

\[
C^{\rm probe}_1(d)+C^D_1(d)+\delta W_1(d)<G_1(d)
\tag{8}
\]

and \(\epsilon_{\rm safe}(d)\le\bar\epsilon\), where
\(\bar\epsilon\) is the declared safety budget. Otherwise execute all-agent.
Among qualifying designs, minimize the certified high-instance excess-risk
bound. The inequality is strict: ties fall back.

This rule predicts fallback cell by cell. It does not target or modify the
EXP-015A empirical fallback gate of \(0.80\); the frozen observation remains
\(0.777778\).

## Proposition 1: finite-horizon guarantees

For every design admitted by (8):

1. the decision error is bounded by the registered likelihood-test bound;
2. expected instance-\(j\) risk satisfies

   \[
   \mathbb E_jR_{\rm ETC}
   \le R_j^\star(B_d^D,D)+\delta W_j(d);
   \]

3. high-instance excess over all-agent is at most
   \(C^{\rm probe}_1+C^D_1-G_1+\delta W_1<0\);
4. low-instance safety deficit is at most \(\epsilon_{\rm safe}(d)\);
5. if \(D\) exceeds the post-probe usable horizon, or either budget cannot
   pay the block, no design qualifies and fallback is automatic.

The proof conditions on the correct/wrong decision, applies the finite-risk
certificate in each branch, and expands the difference from the full-budget
all-agent risk using the four definitions above.

## Proposition 2: short fallback and long commit

Short-horizon fallback is immediate whenever every sufficient test block is
infeasible or violates (8). For long-horizon commit, additionally assume:

1. the catalogue contains a design with finite \(n_{\rm UB}\);
2. its fixed probe and delay costs are \(o(B_{\rm msg})\) and
   \(o(B_{\rm env})\);
3. the high-instance all-agent/oracle risk gap eventually dominates
   \(C^{\rm probe}_1+C^D_1+\delta W_1\);
4. the declared low-instance slack eventually admits (7).

Then that design qualifies for all sufficiently large dual budgets and
commits with the stated error bound. These assumptions expose rather than
hide the amortization condition.

## AC-9 audit: not closed

For each fixed registered instance and finite catalogue, the block procedure
uses \(n_{\rm UB}^{\rm dir}\), while AC-7 requires at least
\(n_{\rm LB}\); its explicit instance factor is

\[
\kappa(d)=n_{\rm UB}^{\rm dir}(d)/n_{\rm LB}(d).
\]

This gives a computable finite-instance comparison, but the present analysis
does not prove a universal constant or logarithmic bound on \(\kappa\), nor
does it prove that the block allocation matches the controlled-belief
occupation optimum (5). Thus the error, regret, safety, short-fallback, and
conditional long-commit statements above are proved, but the requested
near-matching AC-9 claim remains open. Stop gate **B** applies, so no
EXP-016A preregistration or run is authorized.
