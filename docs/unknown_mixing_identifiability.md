# T-016: unknown-mixing identifiability audit

> **Superseded status (T-017):** `unknown_mixing_impossibility.md` upgrades
> the boundary warning below to a formal uniform negative theorem. Fully
> unrestricted unknown mixing is closed negatively; only the separated
> \(\lambda\leq1-\gamma\) positive route remains open.

## Population identifiability

For fixed \(q,b\) and \(n\ge2\), the common-direction covariance is

\[
A(\theta,\lambda)=I_n+q\theta R_n(\lambda^b).
\]

With known unit private variance, equality
\(A(\theta,\lambda)=A(\theta',\lambda')\) first equates the diagonal, giving
\(\theta=\theta'\). If \(\theta>0\), the first off-diagonal then gives
\(\lambda^b=(\lambda')^b\), hence \(\lambda=\lambda'\) on \([0,1]\).
Thus a single stride is algebraically sufficient for this exact stationary
Gaussian model. Two strides are useful for conditioning and model checking,
but are not mathematically necessary for point identifiability.

At \(\theta=0\), \(\lambda\) is unidentifiable because there is no common
factor. With unknown private variance, \(q\ge2\) spatial contrasts identify
that nuisance variance. A \(q=1\) stream can identify temporal covariance
under additional lag conditions, but it cannot support the registered
cross-agent correlation interpretation.

## Why point identifiability is insufficient

Uniform finite-budget adaptation fails without separation from
\(\lambda=1\). At \(\lambda=1\), the common factor is constant over the
experiment and

\[
A=I_n+q\theta{\bf1}{\bf1}^{\mathsf T}.
\]

The Fisher information for \(\theta\) is

\[
\frac12\left\{\frac{qn}{1+q\theta n}\right\}^2
\longrightarrow \frac1{2\theta^2},
\]

so information about \(\theta\) saturates rather than diverges. For
\(\lambda\) arbitrarily close to one, finite-horizon laws are arbitrarily
close to this random-intercept experiment. Therefore no fixed finite dual
budget yields a uniform identification guarantee over
\(\lambda\in[0,1]\), and no uniform-consistency wording is justified on a
parameter space whose closure includes one.

Multiple strides can improve effective rank when a useful larger gap fits,
but cannot overcome the boundary if every affordable \(b\) has
\(\lambda^b\approx1\). The obstruction is finite-horizon near-equivalence,
not exact covariance equivalence in the interior.

## Defensible restricted route

A future composite theorem may assume

\[
\theta\in[\theta_{\min},\theta_{\max}],\quad
0<\theta_{\min},\qquad
\lambda\le1-\gamma
\]

for public \(\gamma>0\), or may consume a separately certified simultaneous
upper bound on mixing time before using the known-mixing theorem. A
profile/mixture likelihood must then establish uniform coverage over this
compact separated set and charge the certification samples and both budgets.
No such composite stopping/matching proof is supplied here.

Accordingly the T-016 audit left AC-8 open for fully unknown mixing. T-017
now closes that unrestricted statement negatively. The valid mainline
wording is:

> adaptive participation with known mixing, or with a separately certified
> upper bound bounded away from non-mixing.

The project is not qualified to use unrestricted unknown-mixing adaptation
as a title or main theorem claim. T-017 makes this scope restriction **D**
permanent and separately repairs AC-9 on a compact separated class.
