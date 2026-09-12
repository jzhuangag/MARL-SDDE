"""Qualification of the complete reference contract, not efficacy evidence."""

from dataclasses import replace

import numpy as np
import pytest

from predictable_collaboration_reference import (
    PredictableReadout, ReferenceConfig, innovation_mean, innovation_variance,
    mixture_boundary, project_simplex,
)


@pytest.mark.parametrize("vector", [[1, 0], [-2, 3, 1], [.2, .3, .5], [2], [0, 0, 0]])
def test_projection_kkt(vector):
    y = np.array(vector, dtype=float)
    w = project_simplex(y)
    np.testing.assert_allclose(w.sum(), 1, atol=1e-14)
    assert np.all(w >= 0)
    threshold = (y - w)[w > 0][0]
    np.testing.assert_allclose((y - w)[w > 0], threshold)
    assert np.all(y[w == 0] <= threshold + 1e-14)


@pytest.mark.parametrize("lam", [0., .5, .9])
def test_innovation_constant_target_exact(lam):
    cfg = ReferenceConfig(mixing=lam, block_size=8)
    residual = 5.0
    theta = .7
    us = np.cos(np.arange(cfg.block_size))
    ys = []
    for u in us:
        residual = lam * residual + u
        ys.append(theta + residual)
    h = innovation_mean([ys], lam)[0]
    np.testing.assert_allclose(h, theta + us[1:].mean() / (1-lam))
    c = np.zeros(cfg.block_size)
    c[1:] += 1
    c[:-1] -= lam
    c /= (cfg.block_size - 1) * (1-lam)
    covariance = lam**np.abs(np.subtract.outer(np.arange(cfg.block_size), np.arange(cfg.block_size)))
    np.testing.assert_allclose(c @ covariance @ c, innovation_variance(lam, cfg.block_size, 1))


def test_post_update_risk_identity_and_gaussian_mean():
    # Gauss-Hermite is exact for this degree-two contrast; not a sampled test.
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    eps = np.sqrt(2 * .7) * nodes
    weights /= np.sqrt(np.pi)
    a, z, c, theta = .8, -.3, .7, .2
    h = theta + eps
    x = a*z + (1-a)*h
    s = a*c + (1-a)*h
    loss_contrast = (z-h)**2 - (c-h)**2
    contrast = (x-theta)**2 - (s-theta)**2
    np.testing.assert_allclose(contrast, a*a*loss_contrast + 2*a*(z-c)*eps, atol=1e-14)
    np.testing.assert_allclose(weights @ contrast, a*a*((z-theta)**2-(c-theta)**2), atol=1e-14)


def test_endogenous_prediction_breaks_mean_identity():
    # Demonstrates why choose-after-observe cannot inherit the reference proof.
    eps = np.array([-1., 1.])
    a, theta, c = .8, 0., 0.
    z = eps
    x = a*z + (1-a)*eps
    s = (1-a)*eps
    assert not np.isclose(np.mean(x*x-s*s), a*a*np.mean(z*z-c*c))


def test_pathwise_dynamic_ogd_bound():
    alpha = .07
    w = np.array([1., 0., 0.])
    losses = 0.
    norm_sum = 0.
    path = 0.
    previous_u = None
    for t in range(70):
        v = np.sin(np.arange(3) + .2*t)
        h = np.cos(.3*t)
        u = np.roll(np.array([.1, .3, .6]), t//17)
        if previous_u is not None:
            path += np.linalg.norm(u-previous_u)
        losses += (w@v-h)**2 - (u@v-h)**2
        g = 2*(w@v-h)*v
        norm_sum += g@g
        w = project_simplex(w-alpha*g)
        previous_u = u
    bound = (2 + 2*np.sqrt(2)*path)/(2*alpha) + alpha*norm_sum/2
    assert losses <= bound + 1e-12


def test_two_phase_and_no_duplicate_accounting():
    cfg = ReferenceConfig(agents=3, block_size=4, horizon=2)
    c = PredictableReadout(cfg)
    block = np.ones((3, 4))
    with pytest.raises(RuntimeError):
        c.finish_block(block)
    action = c.begin_block()
    with pytest.raises(RuntimeError):
        c.begin_block()
    action["candidates"][:] = 1e5  # caller cannot change the committed action
    out = c.finish_block(block)
    np.testing.assert_allclose(out["pre_prediction"], 0)
    assert c.donor_scalars == 6 and c.actor_transitions == 12
    with pytest.raises(RuntimeError):
        c.finish_block(block)
    c.begin_block()
    c.finish_block(block)
    with pytest.raises(RuntimeError):
        c.begin_block()


def test_delay_history_is_causal():
    cfg = ReferenceConfig(agents=2, block_size=2, delay=2)
    c = PredictableReadout(cfg, [-1., 1.])
    bank = [c.local_models]
    for t in range(6):
        action = c.begin_block()
        np.testing.assert_allclose(action["candidates"][:, 0], bank[-1])
        np.testing.assert_allclose(action["candidates"][:, 1], bank[max(0, t-2)][::-1])
        c.finish_block(np.full((2, 2), .1*t))
        bank.append(c.local_models)


def test_future_perturbation_cannot_change_current_action():
    cfg = ReferenceConfig(agents=2, block_size=2)
    a, b = PredictableReadout(cfg, [-.4, .6]), PredictableReadout(cfg, [-.4, .6])
    for t in range(5):
        aa, bb = a.begin_block(), b.begin_block()
        np.testing.assert_array_equal(aa["weights"], bb["weights"])
        np.testing.assert_array_equal(aa["prediction"], bb["prediction"])
        block = np.full((2, 2), .1*t)
        a.finish_block(block)
        b.finish_block(block if t < 4 else block + 10)


def test_donor_bank_independent_of_collaboration_and_delay():
    cfg = ReferenceConfig(agents=3, block_size=3)
    fast = PredictableReadout(cfg, [-1., 0., 1.])
    slow = PredictableReadout(replace(cfg, delay=4), [-1., 0., 1.])
    for t in range(10):
        fast.begin_block()
        slow.begin_block()
        block = np.sin(np.arange(9).reshape(3, 3)+t)
        fast.finish_block(block)
        slow.finish_block(block)
        np.testing.assert_array_equal(fast.local_models, slow.local_models)


def test_noiseless_positive_opportunity_does_not_absorb_at_local():
    # A constructed contract witness only: not a benchmark or statistical result.
    cfg = ReferenceConfig(agents=2, horizon=64, block_size=2, mixing=0.,
                          marginal_variance_bound=0., radius=1., retention=.95)
    c = PredictableReadout(cfg, [-1., 1.])
    adaptive, local = 0., 0.
    for _ in range(cfg.horizon):
        c.begin_block()
        out = c.finish_block(np.ones((2, 2)))
        adaptive += (out["output"][0]-1)**2
        local += (out["local_output"][0]-1)**2
    assert c.weights[0, 1] > 0
    assert adaptive < local


def test_observable_certificate_and_composite_drift():
    cfg = ReferenceConfig(agents=2, horizon=32, block_size=2, mixing=0.,
                          radius=1., retention=.8, marginal_variance_bound=.2)
    # Quadrature of a single step: projection/clipping inequalities hold at each
    # node; centering and the quadratic moment are integrated exactly.
    nodes, probs = np.polynomial.hermite.hermgauss(12)
    probs /= np.sqrt(np.pi)
    lhs, rhs_grad = 0., 0.
    a, alpha, theta = cfg.retention, cfg.weight_step, .3
    e0 = np.array([1., 0.])
    for noise, prob in zip(np.sqrt(2*.2)*nodes, probs):
        c = PredictableReadout(cfg, [-.4, .7])
        before = a*a/(1-a*a)*(-.4-theta)**2
        c.begin_block()
        out = c.finish_block(np.array([[theta+noise, theta+noise], [0., 0.]]))
        after = (a*a*np.sum((c.weights[0]-e0)**2)/(2*alpha)
                 + a*a/(1-a*a)*(c.local_models[0]-theta)**2)
        lhs += prob*(after-before+(out["output"][0]-theta)**2)
        rhs_grad += prob*np.sum(out["gradient"][0]**2)
        expected_certificate = (a*a*alpha*c.gradient_norm_sum/2
                                + mixture_boundary(c.excess_noise_variance, cfg.delta/cfg.agents))
        np.testing.assert_allclose(out["observable_local_excess_bound"], expected_certificate)
    assert lhs <= (1-a)**2*.2/(1-a*a) + a*a*alpha*rhs_grad/2 + 1e-12


@pytest.mark.parametrize("kwargs", [{"mixing": 1.}, {"block_size": 1},
                                    {"agents": 0}, {"delay": -1}, {"radius": float("nan")},
                                    {"marginal_variance_bound": -1}, {"horizon": 2.5}])
def test_invalid_contract_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ReferenceConfig(**kwargs)
