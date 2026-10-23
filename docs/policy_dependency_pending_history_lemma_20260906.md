# Pending-rollout history lemma for policy synchronization

Date: 2026-09-06

Status: deterministic event-time lemma.  The stochastic Markov estimator and
the complete finite-time theorem remain open.

## 1. Pending packet radius

Let `P_k` be the rollouts in flight just before a parameter-update event.  A
packet `p` has owner `r_p` and was born under the mixed policy-version vector
`zeta_p`.  Assume the owner-gradient mean has block cross-sensitivity

\[
\|G_{r_p}(x)-G_{r_p}(y)\|
\le\sum_j L_{r_pj}\|x_j-y_j\|.
\tag{1}
\]

If `beta_p` pays the packet's truncation, critic, and declared Markov bias,
define the observable radius

\[
Z_{p,k}=\beta_p+
\sum_jL_{r_pj}\|\theta_j^k-\zeta_{p,j}\|.
\tag{2}
\]

Then

\[
\|G_{r_p}(\zeta_p)-\nabla_{r_p}F(\theta^k)\|
\le Z_{p,k}.
\tag{3}
\]

Equation (3) is a deterministic consequence of (1) after the meaning of
`beta_p` is fixed.  It does not require treating transitions within a rollout
as independent.

## 2. History energy and one-block motion

Use

\[
\mathcal H_k=\frac12\sum_{p\in\mathcal P_k}w_pZ_{p,k}^2,
\qquad w_p>0.
\tag{4}
\]

Suppose block `i` moves by `u_i`.  For every other pending packet,

\[
Z_{p,k+1}\le Z_{p,k}+L_{r_p i}\|u_i\|.
\tag{5}
\]

Therefore

\[
\Delta\mathcal H_k^{\rm remain}
\le h_{i,k}\|u_i\|
+\frac{\kappa_{i,k}}2\|u_i\|^2,
\tag{6}
\]

where

\[
h_{i,k}=\sum_{p\in\mathcal P_k^{\rm remain}}
w_pZ_{p,k}L_{r_p i},
\qquad
\kappa_{i,k}=\sum_{p\in\mathcal P_k^{\rm remain}}
w_pL_{r_p i}^2.
\tag{7}
\]

### Proof

Apply (5), square both sides, subtract `Z_(p,k)^2`, multiply by `w_p/2`,
and sum over remaining packets.  All quantities are nonnegative, so the
inequality direction is preserved.  Expanding the squares gives (6)--(7).

## 3. Completion and replacement

Let packet `q` complete at event `k`.  Remove its old history contribution
before applying its update.  If a replacement rollout is born after the
update with initial radius `Z_new`, then

\[
\Delta\mathcal H_k
\le
-\frac{w_q}{2}Z_{q,k}^2
+h_{i,k}\|u_i\|
+\frac{\kappa_{i,k}}2\|u_i\|^2
+\frac{w_{\rm new}}2Z_{\rm new}^2.
\tag{8}
\]

No replacement term is added until the rollout is actually born.  A refresh
message in flight is a different object and needs its own delivery accounting;
it is not prematurely counted as a gradient packet.

## 4. Composite return-time drift

For `u_i=-alpha g_i`, combine (8) with objective block smoothness and the exact
outgoing-cache identity.  Conditional on fixed packet and state values,

\[
\begin{aligned}
\Delta(F+C+\mathcal H)
\le{}&-\frac{w_q}{2}Z_{q,k}^2
+\frac{w_{\rm new}}2Z_{\rm new}^2\\
&-\alpha\left[
\langle\nabla_iF+d_i^{\rm out},g_i\rangle
-h_{i,k}\|g_i\|
\right]\\
&+\frac{\alpha^2}{2}
(L_i+P_i^{\rm out}+\kappa_{i,k})\|g_i\|^2.
\end{aligned}
\tag{9}
\]

The scalar minimizer of the last two lines is the return-time rule already
implemented in `signed_drift.py`.  The negative completed-radius term prevents
the history energy from merely accumulating forever; the new-radius term is
the explicit cost of dispatching a stale mixed-policy rollout.

## 5. No-double-counting rule

The radius `Z_p` already pays the difference between the packet birth policy
and the current policy.  A finite-time proof may use either:

- this radius as the Lyapunov--Krasovskii history state; or
- a separate path-energy functional that proves the same gradient discrepancy.

It may not add both full penalties.  Any Markov truncation or critic bias placed
inside `beta_p` must likewise be excluded from a second generic noise term,
except for a separately identified centered variance contribution.

## 6. Statistical interface still required

If the application step is selected using the same realized packet noise that
forms the update direction, adaptive correlation must be handled explicitly.
The clean first theorem will instead split each fully charged reset-trajectory
mini-batch into:

- a control half that selects the return-time scalar step; and
- a conditionally independent update half whose gradient is applied.

Both halves count toward the same environment budget.  This is a theorem
device, not free sensing.  Its sample-efficiency cost must be included in the
next outcome-free nonvacuity audit.  Continuing chains require separated
blocks or a martingale argument and are not silently covered by reset-batch
independence.

The next result must combine (9) with expectation-level noisy graph selection,
queue accounting, and activation coverage to obtain the complete stationarity
bound.
