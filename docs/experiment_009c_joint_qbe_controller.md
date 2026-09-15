# EXP-009C preregistration: joint \((q,b,\eta)\) controller

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: PREREGISTERED
- Version Label: exp009c_preregistration_v1

## Registered correction

EXP-009C retains EXP-009B's scenarios, 128 pilot seeds, 2,048-transition
pilot, 99% one-sided Clopper--Pearson certificate, resource budget, costs,
noise model, participation set, and policy information sets.

The only change is that decorrelation gap becomes a decision variable. For a
certified persistence upper bound \(p^+\), define

\[
\delta^+(b)=\tfrac12(2p^+-1)^b.
\]

Search only gaps satisfying
\(\delta^+(b)<\mu/(2L)\). Let \(b_{\min}\) be the smallest such integer. The
frozen finite candidate set is

\[
\mathcal B(p^+)
=
\left\{
\left\lceil b_{\min}r\right\rceil:
r\in\operatorname{geomspace}(1,4,17)
\right\},
\]

with duplicates removed. For \(p^+=0.5\), set \(b_{\min}=1\).

For every \(b\in\mathcal B(p^+)\) and registered \(q\), optimize \(\eta\) over
the sharp proved-safe interval using EXP-009B's exact finite-budget risk.
Select the minimum-risk \((q,b,\eta)\); ties go to smaller \(q\), then smaller
\(b\), then smaller \(\eta\).

## Evaluation

Primary performance is exact expected final squared error conditional on each
pilot-selected action. No new Monte Carlo trajectory claim is introduced;
EXP-009A/B already validate trajectory reproduction and zero divergence.

## Preregistered gates

1. Certificate coverage is at least 98.5%.
2. Every covered online action has exact covariance radius below one.
3. In every scenario, the median online-to-oracle exact expected-error ratio is
   at most five.
4. Online beats worst-mixing in at least six of eight \(p\le0.9\) scenarios.
5. Median selected \(q\) under \(\rho=0.9\) is no larger than under
   \(\rho=0\).
6. The largest scenario-median oracle ratio is strictly smaller than
   EXP-009B's frozen value 11.6346535121.
7. Joint search selects at least two distinct median gaps across the 12
   scenarios, demonstrating that \(b\) is an active decision rather than a
   fixed reparameterization.

Safety failure invalidates the method. Failure of the oracle-ratio gate
rejects a near-oracle claim but does not invalidate the proved controller.
