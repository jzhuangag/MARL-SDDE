"""Deterministic covariance, quadrature and algebra fixtures. No science samples."""
import os
import time
START = time.process_time()
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "experiments/v3_risk_gate8h"))
import core
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import controller

OUTPUT = HERE / "proof_checks.json"
if OUTPUT.exists():
    raise RuntimeError("Preserve existing fixture attempt")
checks = []


def no_science(*args, **kwargs):
    raise AssertionError("New scientific observations are not authorized")


for name in ("potential_observations", "observations", "direct_learner", "group_rows"):
    setattr(core, name, no_science)


def check(label, condition, **details):
    if time.process_time() - START > 180:
        raise RuntimeError("Mathematical-fixture CPU ceiling exceeded")
    checks.append(dict(label=label, passed=bool(condition), **details))
    if not condition:
        raise AssertionError(label)


try:
    # Dense Schur complements independently verify filtering and terminal loss.
    for theta, lam, q, n in ((.05, .3, 8, 3), (1., .9, 4, 5), (6., .9, 1, 1)):
        y = np.linspace(-.7, .9, n)
        times = np.arange(n)
        Sigma = theta * lam ** abs(times[:, None] - times[None, :]) + np.eye(n) / q
        cross = theta * lam ** (n - 1 - times)
        dense_mean = cross @ np.linalg.solve(Sigma, y)
        dense_explained = cross @ np.linalg.solve(Sigma, cross)
        post = controller.filter_probe(y, theta, lam, q)
        check("posterior_dense", abs(post.mean - dense_mean) < 1e-12 and
              abs(post.explained_variance - dense_explained) < 1e-12,
              theta=theta, n=n)
        dense_ll = -.5 * (n * np.log(2*np.pi) + np.linalg.slogdet(Sigma)[1]
                          + y @ np.linalg.solve(Sigma, y))
        check("density_dense", abs(post.log_density - dense_ll) < 1e-12)
        for learner in core.LEARNERS:
            for a in (core.Action(1, 1), core.Action(q, 2)):
                f = core.Family(q, theta + 1, lam, 1)
                J, d, k = core.program_risk(learner, a, 45, 16, f, theta)
                _, w = core.weights(learner, core.scheduled(a, 45, 16, f.h), f.D)
                t = a.b * np.arange(1, len(w) + 1)
                K = theta * lam ** abs(t[:, None]-t[None, :]) + np.eye(len(w))/a.q
                terminal_cross = theta * lam ** (t[:, None] + (n-1-times)[None, :])
                cov = w @ terminal_cross
                mu = cov @ np.linalg.solve(Sigma, y)
                variance = w @ K @ w - cov @ np.linalg.solve(Sigma, cov)
                exact = (d + mu)**2 + variance
                actual = controller.conditional_loss((J, d, k), post)
                check("terminal_dense_schur", abs(actual-exact) < 1e-11,
                      learner=learner, action=[a.q, a.b])

    # Reflection identity: direct integration versus the derived positive part.
    for A0, A2, B1 in ((0., .3, 1.), (.5, .2, .1), (-.5, .1, 1.3), (1., 0., 0.)):
        A = lambda x: A0 + A2*x*x
        B = lambda x: B1*x
        # Pair positive/negative points explicitly; no random draws.
        direct = quad(lambda x: (abs(A(x)+B(x))+abs(A(x)-B(x))-2*abs(A(x)))
                      * norm.pdf(x)/2, 0, 12, epsabs=2e-10, limit=250)[0]
        derived = quad(lambda x: max(0., abs(B(x))-abs(A(x))) * norm.pdf(x),
                       0, 12, epsabs=2e-10, limit=250)[0]
        check("symmetry_gap_quadrature", abs(direct-derived) < 2e-8,
              coefficients=[A0, A2, B1], gap=derived)

    witnesses = []
    for c in (0., 3., 4.5, 5.4, 6.):
        V = 7.
        v = V-c*c/V
        fixed = [1., (1+V)/4, (1+2*V+2*c)/9]
        if v >= 4 or c == 0:
            gain = formula = 0.
            even_gap = 0.
        else:
            sigma = c / np.sqrt(V)
            b = np.sqrt(4-v)
            lo, hi = (-b-1)/sigma, (b-1)/sigma
            gain = quad(lambda z: (1-((1+sigma*z)**2+v)/4)*norm.pdf(z),
                        lo, hi, epsabs=1e-12)[0]
            P = norm.cdf(hi)-norm.cdf(lo)
            M1 = sigma*(norm.pdf(lo)-norm.pdf(hi))
            M2 = sigma*sigma*(P+lo*norm.pdf(lo)-hi*norm.pdf(hi))
            formula = ((3-v)*P-2*M1-M2)/4
            # Best even rule has A=(v+1+sigma^2*z^2)/4-1, B=sigma*z/2.
            even_gap = quad(lambda z: max(0., abs(sigma*z/2)-
                            abs((v+1+sigma*sigma*z*z)/4-1))*norm.pdf(z),
                            0, 12, epsabs=2e-10, limit=250)[0]
        check("paid_boundary", abs(gain-formula)<1e-11 and min(fixed)==1. and
              ((gain>0)==(v<4 and c>0)), c=c, v=v, gain=gain)
        witnesses.append(dict(V=V,c=c,conditional_variance=v,full_fixed_risks=fixed,
                              optimal_risk=1-gain,even_rule_excess=even_gap))

    f = core.Family(8, 1., .9, 1)
    programs = tuple(tuple(core.program_risk("running_mean", a, 100, 32, f, theta)
                           for a in core.catalogue(f))
                     for theta in (f.theta_low, f.theta_high))
    for y in (np.array([.2, -.7, .3]), np.zeros(3), np.array([100., -100., 100.])):
        for mode in ("full", "even", "marginal"):
            a, scores, probabilities = controller.choose(y, f, programs, mode=mode)
            b, reflected, _ = controller.choose(-y, f, programs, mode=mode)
            check("prototype_finite_argmin", scores[a]==min(scores) and
                  abs(sum(probabilities)-1)<1e-14 and min(scores)>=-1e-10,
                  mode=mode)
            if mode != "full":
                check("even_ablation_reflection", a==b and np.allclose(scores,reflected), mode=mode)
    check("zero_updates", controller.conditional_loss((1.,1.,0.),
          controller.filter_probe([3.],1.,.5,8)) == 1.)
    for invalid in ([], [float("nan")], [[1.]]):
        rejected=False
        try: controller.filter_probe(invalid,1.,.5,8)
        except ValueError: rejected=True
        check("invalid_probe_rejected", rejected)
    # No scientific inputs were consumed. The immutable exact-risk source is hashed.
    result = dict(status="PASS",checks=checks,passed=len(checks),witnesses=witnesses,
        new_scientific_samples=0,random_number_calls=0,
        scope="Deterministic tiny fixtures, not independent empirical validation",
        locked_core_sha256=hashlib.sha256((REPO/"experiments/v3_risk_gate8h/core.py").read_bytes()).hexdigest(),
        cpu_seconds=time.process_time()-START)
except BaseException as exc:
    result=dict(status="FAIL",checks=checks,error=repr(exc),cpu_seconds=time.process_time()-START)
    raise
finally:
    if "result" in globals():
        with OUTPUT.open("x", encoding="utf-8") as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
        print(json.dumps({k:result[k] for k in ("status","cpu_seconds")},allow_nan=False))
