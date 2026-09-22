# T-039 first-order downstream-risk oracle matching

## Decision

T-039 closes the downstream-risk version of AC-9c **to first order on the
declared separated class**.  It does not claim finite-budget equality with
the controlled-belief occupation program or uniformity near an oracle switch,
zero hypothesis separation, or unit mixing.

## Assumptions

Let the budget lie on the public ray indexed by \(s\).  There are finitely
many regimes \(j\) and a stability-screened downstream policy catalogue.  Let
\(V_j^*(s)\) be the best risk when regime \(j\) is known.  Uniformly over the
compact separated class, assume:

1. **nondegenerate oracle risk:**
   \(k_-/s\le V_j^*(s)\le k_+/s\), with \(k_->0\);
2. **budget sensitivity:** for \(0\le c\le s/2\),
   \[
   V_j^*(s-c)-V_j^*(s)\le Lc/s^2;
   \tag{1}
   \]
3. **separated probe:** a public fixed probe has Bhattacharyya exponent at
   least \(i_->0\) per sample and costs at most \(c_p\) units of the budget
   ray per sample;
4. **bounded wrong commitment:** every screened committed policy has risk at
   most \(W/(s-c)\);
5. delay contributes a public fixed ray cost \(c_D\), and mixing is known or
   independently certified with \(\lambda\le1-\gamma\).

For the finite Gaussian catalogue in T-017, positive \(i_-\) follows from
hypothesis separation and compactness.  The expansion already proved in
`adaptation_threshold_sandwich.md` supplies (1) away from an oracle switch.
These assumptions deliberately exclude exactly the degeneracies identified
by AC-8 and AC-9a.

## Algorithm

For \(s>1\), take

\[
n_s=\left\lceil\frac{3\log s}{i_-}\right\rceil
\tag{2}
\]

fixed probe samples, run the likelihood-ratio test, and execute the
known-regime oracle policy on the remaining budgets.  If
\(c_s=c_pn_s+c_D>s/2\), execute the theorem-derived fallback.  The positive
matching claim begins only after this explicit feasibility threshold.

## Theorem

Whenever \(c_s\le s/2\), the explore-then-commit risk satisfies, for every
regime \(j\),

\[
R_j^{\rm ETC}(s)-V_j^*(s)
\le
\frac{Lc_s}{s^2}
+\frac{W}{2(s-c_s)s^3}
=O\!\left(\frac{\log s}{s^2}\right),
\tag{3}
\]

and hence

\[
\frac{R_j^{\rm ETC}(s)}{V_j^*(s)}
\le 1+O\!\left(\frac{\log s}{s}\right).
\tag{4}
\]

Measure a single unknown-regime policy by its worst oracle-normalized risk and
let

\[
U^*(s)=\inf_{\pi\ \mathrm{predictable}}
       \max_j\frac{R_j^\pi(s)}{V_j^*(s)},
\]

where the infimum includes all controlled-belief policies but no policy is
given \(j\).  A known-regime oracle lower-bounds every numerator, while ETC is
one feasible common policy.  Consequently (4) yields

\[
1\le U^*(s)
\le\max_j\frac{R_j^{\rm ETC}(s)}{V_j^*(s)}
\le1+O\!\left(\frac{\log s}{s}\right).
\tag{5}
\]

Thus the explicit policy matches the best common unknown-regime adaptive
policy to first order in oracle-normalized minimax risk on the separated
class.

### Proof

The Bhattacharyya test bound and (2) give
\(p_{\rm err}\le\tfrac12e^{-i_-n_s}\le(2s^3)^{-1}\).  On a correct decision,
only the charged probe/delay scale is lost, and (1) bounds that loss by
\(Lc_s/s^2\).  On a wrong decision, Assumption 4 bounds its contribution by
\(p_{\rm err}W/(s-c_s)\).  This proves (3).  Dividing by \(k_-/s\) proves
(4).  A policy that knows \(j\) has a superset of the information available
to a common unknown-regime policy, so every oracle-normalized risk is at least
one.  ETC supplies the common upper comparator, proving (5).

## Safety and finite-budget boundary

The same calculation gives an asymptotically vanishing baseline-relative
deficit \(O(\log s/s^2)\) whenever the known-regime oracle is no worse than
the baseline.  It does not imply exact pathwise no-harm at every finite
budget.  Below the feasibility/safety threshold, the theorem-derived
fallback remains mandatory.

The result also does not second-order match the identification opportunity
cost in `adaptive_pareto_lower_bound.md`.  A Track-and-Stop-style equality to
that finite-budget occupation value remains open and is not needed for the
first-order claim in (5).
