import numpy as np

from experiments.dependence_delay_linear.t071_sampled_graph_controller import (
    action_change_statistics,
    choose_observable_actions,
    sample_markov_observations,
    simulate_policy,
    stale_snapshot,
)


def test_markov_observations_are_reproducible_and_finite():
    targets = np.zeros((4, 4))
    first, first_noise = sample_markov_observations(
        targets=targets,
        steps_per_block=10,
        noise_scale=0.5,
        spatial_correlation=0.9,
        temporal_correlation=0.6,
        seed=17,
    )
    second, second_noise = sample_markov_observations(
        targets=targets,
        steps_per_block=10,
        noise_scale=0.5,
        spatial_correlation=0.9,
        temporal_correlation=0.6,
        seed=17,
    )
    assert np.array_equal(first, second)
    assert np.array_equal(first_noise, second_noise)
    assert np.isfinite(first).all()


def test_probe_halves_are_crossfit_and_can_reject():
    local = np.asarray([0.0, 0.0, 1.0, 1.0])
    shadow = local.copy()
    donor = np.asarray([0.0, 4.0, 1.0, 1.0])
    selection = np.asarray([[4.0, 0.0, 1.0, 1.0], [4.0, 0.0, 1.0, 1.0]])
    validation = np.asarray([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    output, selected, shadowed, scores = choose_observable_actions(
        local_post=local,
        shadow_post=shadow,
        donor_snapshot=donor,
        selection_probe=selection,
        validation_probe=validation,
        alpha_grid=[0.5, 1.0],
    )
    assert output[0] == shadow[0]
    assert selected[0] == 0
    assert shadowed[0]
    assert scores == 28


def test_probe_steps_are_charged_but_not_learning_updates():
    targets = np.zeros((3, 4))
    observations = np.ones((3, 10, 4))
    controller = simulate_policy(
        observations=observations,
        targets=targets,
        initial_parameter=0.0,
        gain=0.04,
        delay=1,
        decision_blocks=[0, 2],
        probe_steps=4,
        selection_steps=2,
        alpha_grid=[0.5, 1.0],
        policy="observable",
    )
    local = simulate_policy(
        observations=observations,
        targets=targets,
        initial_parameter=0.0,
        gain=0.04,
        delay=1,
        decision_blocks=[0, 2],
        probe_steps=4,
        selection_steps=2,
        alpha_grid=[0.5, 1.0],
        policy="local_no_probe",
    )
    assert controller.learning_transitions == 22
    assert controller.probe_transitions == 8
    assert local.learning_transitions == 30
    assert local.probe_transitions == 0
    assert controller.candidate_scores == 2 * 4 * 7


def test_delay_snapshot_is_exact():
    history = [np.repeat(1.0, 4), np.repeat(2.0, 4), np.repeat(3.0, 4)]
    assert np.array_equal(stale_snapshot(history, np.repeat(4.0, 4), 3), history[0])
    assert np.array_equal(stale_snapshot([], np.repeat(4.0, 4), 0), np.repeat(4.0, 4))


def test_action_change_statistics_separate_shift_boundaries():
    actions = np.asarray([[0, 0], [1, 0], [1, 2], [1, 2]])
    result = action_change_statistics(actions, [2])
    assert result == {
        "shift_changes": 1,
        "shift_total": 1,
        "other_changes": 1,
        "other_total": 2,
    }
