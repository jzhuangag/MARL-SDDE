import numpy as np

from experiments.dependence_delay_linear.t068_safe_personalized_mixing import (
    initial_moment_state,
)
from experiments.dependence_delay_linear.t070_nonstationary_graph import (
    propagate_graph_block,
    recipient_actions,
    registered_static_graphs,
    retarget_state,
    static_graph_components,
    static_graph_risks,
)


def test_catalogue_and_static_graph_count():
    assert len(recipient_actions(4, 0, [0.5, 1.0])) == 7
    graphs = registered_static_graphs(7, 4)
    assert graphs.shape == (2401, 4)
    assert np.array_equal(graphs[0], np.zeros(4))
    assert np.array_equal(graphs[-1], np.repeat(6, 4))


def test_retarget_preserves_parameters():
    old = np.asarray([-1.0, -1.0, 1.0, 1.0])
    new = np.asarray([-1.0, 1.0, -1.0, 1.0])
    parameters = np.asarray([0.2, -0.3, 0.4, 0.1])
    state = initial_moment_state(old, parameters, delay=2)
    shifted = retarget_state(state, old, new)
    assert np.allclose(shifted.mean[:4] + new, parameters)
    assert np.array_equal(shifted.covariance, state.covariance)


def test_safe_oracle_is_checkpoint_safe():
    targets = np.asarray([-0.5, -0.5, 0.5, 0.5])
    state = initial_moment_state(targets, np.ones(4), delay=1)
    result = propagate_graph_block(
        state,
        targets=targets,
        gain=0.04,
        curvature=1.0,
        local_steps=8,
        noise_scale=0.5,
        spatial_correlation=0.0,
        temporal_correlation=0.6,
        alpha_grid=[0.5, 1.0],
        safe_oracle=True,
    )
    assert np.all(result.personalized_risk <= result.shadow_risk + 1e-12)


def test_batched_local_graph_matches_scalar_propagation():
    agents = 4
    blocks = 3
    schedule = np.repeat(np.asarray([[-1.0, -1.0, 1.0, 1.0]]), blocks, axis=0)
    graphs = np.zeros((1, agents), dtype=np.int16)
    components = static_graph_components(
        graphs=graphs,
        agents=agents,
        delay=1,
        blocks=blocks,
        decision_blocks=[0, 2],
        gain=0.04,
        curvature=1.0,
        local_steps=10,
        alpha_grid=[0.5, 1.0],
        unit_target_schedule=schedule,
    )
    auc, terminal = static_graph_risks(
        components,
        initial_parameter=0.5,
        target_scale=0.3,
        gain=0.04,
        curvature=1.0,
        local_steps=10,
        noise_scale=0.5,
        spatial_correlation=0.0,
        temporal_correlation=0.6,
    )
    targets = 0.3 * schedule[0]
    state = initial_moment_state(targets, np.repeat(0.5, agents), delay=1)
    path = []
    for _ in range(blocks):
        result = propagate_graph_block(
            state,
            targets=targets,
            gain=0.04,
            curvature=1.0,
            local_steps=10,
            noise_scale=0.5,
            spatial_correlation=0.0,
            temporal_correlation=0.6,
            alpha_grid=[0.5, 1.0],
            fixed_action_indices=np.zeros(agents, dtype=int),
        )
        state = result.state
        path.append(np.mean(result.personalized_risk))
    assert np.allclose(auc[0], np.mean(path), rtol=2e-12, atol=2e-12)
    assert np.allclose(terminal[0], path[-1], rtol=2e-12, atol=2e-12)


def test_batched_nonlocal_graph_matches_scalar_under_target_switch():
    agents = 4
    schedule = np.asarray(
        [
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0],
        ]
    )
    # Catalogue indices are recipient-specific because the self donor is omitted.
    graph = np.asarray([[1, 1, 5, 5]], dtype=np.int16)
    components = static_graph_components(
        graphs=graph,
        agents=agents,
        delay=1,
        blocks=4,
        decision_blocks=[0, 2],
        gain=0.04,
        curvature=1.0,
        local_steps=10,
        alpha_grid=[0.5, 1.0],
        unit_target_schedule=schedule,
    )
    auc, terminal = static_graph_risks(
        components,
        initial_parameter=0.5,
        target_scale=0.3,
        gain=0.04,
        curvature=1.0,
        local_steps=10,
        noise_scale=0.5,
        spatial_correlation=0.9,
        temporal_correlation=0.6,
    )
    scaled = 0.3 * schedule
    state = initial_moment_state(scaled[0], np.repeat(0.5, agents), delay=1)
    path = []
    previous = scaled[0]
    for block in range(4):
        target = scaled[block]
        if block > 0 and not np.array_equal(previous, target):
            state = retarget_state(state, previous, target)
        indices = graph[0] if block in {0, 2} else np.zeros(agents, dtype=int)
        result = propagate_graph_block(
            state,
            targets=target,
            gain=0.04,
            curvature=1.0,
            local_steps=10,
            noise_scale=0.5,
            spatial_correlation=0.9,
            temporal_correlation=0.6,
            alpha_grid=[0.5, 1.0],
            fixed_action_indices=indices,
        )
        state = result.state
        path.append(np.mean(result.personalized_risk))
        previous = target
    assert np.allclose(auc[0], np.mean(path), rtol=2e-12, atol=2e-12)
    assert np.allclose(terminal[0], path[-1], rtol=2e-12, atol=2e-12)
