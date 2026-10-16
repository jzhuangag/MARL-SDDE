## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theorem-interface closure during CPU gate execution
- Origin Date: 2026-09-08
- Verification Status: DETERMINISTIC EPOCH CONVERSION PROVED; MOTION REMAINDER REQUIRED

# Full-gradient stationarity under asynchronous permutation ownership

## Why the earlier corollary is not the right tool

The existing randomized-owner corollary assumes that, immediately before
every owner draw, every block has probability at least `pi_min`. Sampling a
random permutation without replacement violates that assumption after an
owner has already appeared. It is therefore incorrect to invoke the
per-launch corollary for the frozen Multiwalker alignment gate.

The correct comparison is deterministic and epoch based. It works for any
permutation, including a random permutation revealed sequentially, but it
must carry the motion of the asynchronous iterate within the epoch.

## Epoch-motion lemma

Let an epoch start at `s` and contain each of the `n` owners exactly once.
Owner `i` launches at event `p(i)`. Suppose its block gradient is Lipschitz:

\[
 \|\nabla_iF(\theta_{p(i)})-\nabla_iF(\theta_s)\|
 \le L_i P_{i,s},
\]

where `P_(i,s)` is any predictable upper bound on the total parameter-path
motion from the epoch start to that launch. For weights
`0<kappa_min<=kappa_p<=kappa_max`, the elementary inequality
`||a+b||^2 >= ||a||^2/2-||b||^2` gives

\[
 \sum_i\kappa_{p(i)}\|\nabla_iF(\theta_{p(i)})\|^2
 \ge {\kappa_{\min}\over2}\|\nabla F(\theta_s)\|^2
 -\kappa_{\max}\sum_i L_i^2P_{i,s}^2.
\tag{1}
\]

No independence or identical worker-speed assumption is used in (1). Receipt
events may occur between launches; their actual clipped update lengths are
included in `P_(i,s)`. Random delays therefore enter through measurable path
motion rather than disappearing inside an informal synchronized epoch.

Summing (1) over complete epochs and inserting the selected-block bound from
the core Lyapunov theorem yields

\[
 {\kappa_{\min}\over2}
 \sum_e\mathbb E\|\nabla F(\theta_{s_e})\|^2
 \le B_N+
 \kappa_{\max}\sum_{e,i}\mathbb E[L_i^2P_{i,e}^2],
\tag{2}
\]

where `B_N` is exactly the right-hand side already proved for selected blocks.
Thus full-gradient stationarity follows when the accumulated epoch-motion
term is sublinear at the chosen step-size scale. With uniformly bounded
per-epoch path length `P_e`, the additional term is at most
`n L_max^2 P_e^2` per epoch.

## Consequence for the mainline

The theorem-facing algorithm may use random owner permutations to guarantee
balanced finite-horizon coverage without falsely applying the positive
per-launch sampling-probability corollary. The price is the explicit term in
(2). An experiment must log receipt-time update path length so this term can
be audited. The i.i.d.-owner corollary remains valid as a separate option;
neither result licenses omitting owner-selection or within-epoch motion.

`owner_epoch_stationarity.py` evaluates the exact algebraic lower bound and
the accompanying tests include heterogeneous block dimensions, weights,
smoothness constants, and path-motion bounds.
