# V3 proof-repair memo — read-only audit, 2026-10-16

This memo neither changes the frozen manuscript nor claims that repairs have already been implemented. It gives explicit mathematical reasons and bounded repair paths. None requires new GPU experiments. The positive empirical records remain separate from the truth of the theorems.

## T1. Adaptive Gaussian information is not uniformly bounded per history

Location: Appendix F, appendices.tex:168; main Eq. (30).

The proof claims compact parameters and lambda <= 1-gamma upper-bound **every** feasible observation's conditional directional KL by a finite i_+. The conditional KL contains the squared difference of the two predictive means. Those Gaussian means are not bounded pathwise.

A concrete admissible example uses q=b=1, theta_0=1, theta_1=2, lambda=1/2 and stationary zero-mean initialization. After the first observation Y_1, the two next predictive means are Y_1/4 and Y_1/3. The instance-1 innovation variance is 8/3. The squared-mean contribution to the next conditional KL is therefore Y_1^2/768. It is unbounded even though all model parameters lie in a separated compact set. This is a counterexample to that proof step, not a proof that the threshold theorem is false.

Repair the necessary bound through **expected** information. Prove uniform second-moment bounds for the true common state and both predictable filters under the finite q,b catalogue and lambda <= 1-gamma. A contracting filter recursion and bounded Gaussian innovation second moments supply finite uniform L2 bounds. Consequently E_j I_j(Z_k,A_k) <= C uniformly in k, where C depends on the compact class and catalogue. For bounded tau <= n_max(s), nonnegativity gives

    E_j sum_{k<=tau} I_k <= sum_{k<=n_max(s)} E_j I_k <= C n_max(s).

The information lower bound then implies s >= c d_delta because the budget bounds n_max(s) linearly. This must be written with the adaptive spacing/state/filter moment bounds and constants; the outline alone is not a completed proof.

## T2. Strict feasibility is not compact; scalar global accuracy is not automatic

Location: Proposition 2, main.tex:420; Appendix A.D, appendices.tex:102.

Removing c_aff >= 1 from the closure can remove the minimizer. Within the current written assumptions, take D=delta=0, H=1, symmetric xi in {-1,1}, L=G=mu=K=Omega=1, eta in [0,1/4] and R_0=0. Then

    c = 1-eta+2 eta^2, d = 2 eta^2,
    E = (0,1/4], U = R_star = 2 eta/(1-2 eta).

The infimum is zero as eta tends to zero, but is not attained on E. The manuscript never explicitly requires eta_min>0. An empty certified set is another unhandled case. The extreme-value theorem cannot be invoked on an arbitrary strictly feasible set.

A sufficient repair is a **nonempty closed certified subset** with eta_min>0, a_delta>=a_min>0 and c_aff<=1-kappa for declared positive margins. Otherwise state an infimum/epsilon minimizer and handle empty domains. Check that any revised domain includes the saved experimental actions before connecting the theorem to those results.

Bounded one-dimensional optimization is not inherently global. The saved sensitivity helper uses a geometric grid plus scipy bounded minimization. Either prove the needed shape property, give certified interval/global enumeration with an error bound, or explicitly restrict the theorem and implementation claim to an enumerated finite step catalogue/approximate scalar solve. Do not silently modify old experiments or call a local solver an exact continuous oracle.

## T3. Replacement-block envelope needs a positive-part recurrence

Locations: Appendix A after Eq. (54); Eq. (57); Corollary 2's stated R_{m+1} recursion.

The existing two-case argument establishes contraction after a full replacement block when the envelope exceeds R_star, and invariance below R_star. It does **not** establish the unqualified affine replacement-block recurrence used in the tracking proof. Nor is a sliding maximum forced to contract after every single step. For example, the abstract one-step upper inequality u_new<=c R+d with c=1/2,d=1 and initial window maximum 1 permits new values 1.5,1.75,1.875. After three replacements the maximum 1.875 exceeds c*1+d=1.5. This exposes an inference gap, not an affine-TD counterexample.

Use Gamma_r=(Z_r-R_star)_+. A valid fixed-target replacement-block argument yields Gamma'_r<=c Gamma_r. A target shift bounded by v then gives

    Gamma_{r+1} <= (1+chi)c Gamma_r + chi R_star + (1+chi^{-1})v^2.

Telescoping with bar_c=(1+chi)c<1 yields

    Z_n <= bar_c^n (Z_0-R_star)_+
           + ((1+chi)d+(1+chi^{-1})v^2)/(1-bar_c),

which is no greater than the manuscript's displayed upper bound using bar_c^n Z_0. This shows a plausible repair of the stated final bound, subject to the next assumption audit. Use distinct symbols for update time, replacement-block index and block length; state the robust one-step inequality for E||e_{k+1}||^2, not for the sliding maximum.

### Delayed-history / moving-target assumptions must also be explicit

The appendix allows a random delay profile while the envelope is max_s E||e_s||^2, not E max_s||e_s||^2. A history-dependent random selected index need not satisfy E||e_{k-tau}||^2 <= max_s E||e_s||^2. A deterministic profile, explicit sufficient exogeneity, or a different envelope/coefficient bound is needed. The saved controlled studies use declared deterministic profiles; narrowing the theorem to that model would not erase those observations.

At an actor-epoch boundary, the first delayed telescoping window also contains updates made under the old target/action. Per-epoch assumptions alone do not automatically bound that mixed window using the new target's innovation constants. Specify a valid flush/consistent-history protocol, a common bound across the mixed history, or include the target-motion residual explicitly. This interface is not closed merely by the positive-part algebra.

## T4. Certificate-relative safety is not true-baseline risk superiority

Location: Theorem 4 and Appendix E, especially Eq. (63).

The proof subtracts the **upper certificate** Rbar_all, not the unknown actual all-agent MSE. From E L_A<=U_A<U_all and E L_all<=U_all one cannot conclude E L_A<E L_all. Preserve the valid certificate comparison and write the low-instance allowance explicitly relative to the fallback certificate unless a lower/equality baseline bound is supplied.

The guarantee is also conditional on simultaneous certificate coverage. An unconditional MSE bound needs a bad-event loss/moment control: if loss<=M then add at most epsilon*M. Coverage probability alone cannot be added as epsilon to an unbounded squared-error risk. Alternatively retain a precise coverage-conditional statement.

## T5. Applicability and complexity claims need narrower interfaces

- The threshold theorem assumes a uniform K(a)/s+O(s^-2) risk expansion. A fixed positive step with a positive innovation floor does not acquire this expansion just by increasing the budget. Identify the averaging/vanishing-step learner to which the expansion applies, or explicitly keep this phase theorem as a separate conditional risk class. Do not claim it follows from the constant-step affine certificate.
- Rank-one TD gradients permit O(qd) update aggregation. They do not by themselves make exact estimation of lambda_max(E Hbar^T Hbar) an O(d)-memory operation. Distinguish cheap scalar conservative bounds/supplied certificates from exact spectral calibration; charge whichever is actually used.
- Define consistent initialization for the delayed negative-index history. A telescoping formula cannot use unspecified pre-training updates.

## Recommended non-GPU repair sequence

1. Correct the EXP-016B estimand/denominator wording before touching any scientific threshold or result.
2. Close T1's expected-information proof and T2's existence/solver domain.
3. Rewrite T3 with the positive-part envelope and explicit delayed/mixed-epoch history assumptions; if the moving-policy extension cannot be proved, remove only that unsupported extension rather than the valid fixed-policy evidence.
4. Make T4 certificate-relative and coverage-conditional; narrow T5 where needed.
5. Verify every retained action/claim against immutable saved evidence, repeat citation-context/originality and final PDF audits, then prepare submission metadata. No wait for the new40-hour actor study is required.
