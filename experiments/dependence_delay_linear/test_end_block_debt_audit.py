"""Deterministic algebra checks; no efficacy data or frozen-code modification."""
import numpy as np

from experiments.dependence_delay_linear.t071_sampled_graph_controller import local_updates


def test_paired_excess_exact_identity():
    eta = .2
    theta = np.array([.3, -.2])
    noise = np.array([[.4, -.1], [-.2, .8], [.1, .2]])
    x, shadow = np.array([1., -1.]), np.array([0., .4])
    local = local_updates(x, theta + noise, eta)
    ref = local_updates(shadow, theta + noise, eta)
    difference = (1-eta)**3 * (x-shadow)
    np.testing.assert_allclose(local-ref, difference)
    observed = (local-theta-noise.mean(0))**2-(ref-theta-noise.mean(0))**2
    truth = (local-theta)**2-(ref-theta)**2
    np.testing.assert_allclose(observed-truth, -2*difference*noise.mean(0))


def test_ar1_conditional_bias_by_symmetric_innovation_quadrature():
    # Symmetric zero-mean innovations suffice for the linear bias identity.
    lam, xi0, m, d = .8, .7, 3, .4
    import itertools
    means = []
    for signs in itertools.product([-1., 1.], repeat=m):
        state, values = xi0, []
        for innovation in signs:
            state = lam*state + innovation
            values.append(state)
        means.append(np.mean(values))
    expected = xi0*sum(lam**s for s in range(1,m+1))/m
    np.testing.assert_allclose(-2*d*np.mean(means), -2*d*expected)
    assert expected > 0


def test_queue_drift_and_telescoping():
    q, initial, epsilon, total = .3, .3, .1, 0.
    for g in [-1., .4, 2., -.7, .2]:
        u = g-epsilon
        nxt = max(0., q+u)
        assert (nxt*nxt-q*q)/2 <= q*u+u*u/2+1e-12
        q, total = nxt, total+g
    assert total <= 5*epsilon+q-initial+1e-12


def test_uniform_target_interval_shield():
    center, radius = .2, .7
    for theta in np.linspace(center-radius, center+radius, 11):
        for candidate in np.linspace(-2,2,9):
            for shadow in [-1., 0., 1.]:
                bound = (candidate-center)**2-(shadow-center)**2+2*radius*abs(candidate-shadow)
                truth = (candidate-theta)**2-(shadow-theta)**2
                assert truth <= bound+1e-12


def test_pre_mix_debt_does_not_bound_final_mix():
    theta = local = shadow = 0.
    observed_excess = (local-theta)**2-(shadow-theta)**2
    q = max(0., observed_excess)
    mixed = 10.  # A feasible donor-only mix in a graph with such a donor.
    assert q == 0
    assert (mixed-theta)**2-(shadow-theta)**2 == 100
