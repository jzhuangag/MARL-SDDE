import numpy as np

from .halfcheetah_tail import (
    ACTION_DIM,
    PARAM_DIM,
    local_features,
    mixed_block,
    physical_neighbors,
    random_degree_matched_supports,
    retained_fraction,
    summarize_influence,
)


def test_physical_chain_degrees_and_symmetry():
    neighbors = physical_neighbors()
    assert sorted(len(neighbors[i]) for i in range(ACTION_DIM)) == [1, 1, 2, 2, 2, 2]
    for owner, donors in neighbors.items():
        for donor in donors:
            assert owner in neighbors[donor]


def test_local_features_match_halfcheetah_layout():
    observation = np.arange(17, dtype=np.float64)
    assert np.array_equal(local_features(observation, 3), [1.0, 1.0, 5.0, 14.0])


def test_mixed_block_recovers_quadratic_cross_term():
    theta = np.zeros((ACTION_DIM, PARAM_DIM), dtype=np.float64)
    target = np.arange(PARAM_DIM**2, dtype=np.float64).reshape(PARAM_DIM, PARAM_DIM) / 10.0

    def objective(candidate):
        return float(candidate[1] @ target @ candidate[4])

    estimated = mixed_block(objective, theta, 1, 4, 0.03, 0.07)
    assert np.allclose(estimated, target, atol=1e-12)


def test_retained_fraction_and_random_graph_degree():
    influence = np.ones((ACTION_DIM, ACTION_DIM), dtype=np.float64) - np.eye(ACTION_DIM)
    neighbors = physical_neighbors()
    assert np.isclose(retained_fraction(influence, neighbors), 10.0 / 30.0)
    random_support = random_degree_matched_supports(
        np.random.default_rng(1), [len(neighbors[i]) for i in range(ACTION_DIM)]
    )
    assert [len(random_support[i]) for i in range(ACTION_DIM)] == [
        len(neighbors[i]) for i in range(ACTION_DIM)
    ]


def test_summary_favors_chain_when_only_chain_edges_are_nonzero():
    influence = np.zeros((ACTION_DIM, ACTION_DIM), dtype=np.float64)
    for owner, donors in physical_neighbors().items():
        for donor in donors:
            influence[owner, donor] = 1.0
    summary = summarize_influence(influence, random_graphs=512, graph_seed=8)
    assert summary.physical_fraction == 1.0
    assert summary.neighbor_top_rate == 1.0
    assert summary.physical_minus_random_median > 0.5
