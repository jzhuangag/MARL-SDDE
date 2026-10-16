from __future__ import annotations

import numpy as np
import torch

from .async_pistonball_ctde import (
    CompressedReplay,
    OwnerGradientQueue,
    PistonActor,
    PistonCentralCritic,
    PolicyCacheBank,
    apply_refresh_action,
    changed_parameter_groups,
    clone_parameter_groups,
    collect_parallel_segment,
    critic_td_loss,
    hard_budget_remaining,
    joint_policy_actions,
    owner_critic_gradient,
    polyak_update,
    resize_uint8_images,
    select_age_refresh,
    select_cache_lyapunov_refresh,
    select_mismatch_refresh,
    signed_refresh_for_batch,
)


def tiny_actors(count: int = 4) -> tuple[torch.nn.Module, ...]:
    torch.manual_seed(3)
    return tuple(PistonActor() for _ in range(count))


def test_models_have_expected_ctde_shapes() -> None:
    actors = tiny_actors()
    critic = PistonCentralCritic(4)
    observations = torch.zeros(4, 3, 64, 32, dtype=torch.uint8)
    actions = torch.cat(
        tuple(actor(observations[index : index + 1]) for index, actor in enumerate(actors)),
        dim=1,
    )
    state = torch.zeros(1, 3, 64, 64, dtype=torch.uint8)
    assert actions.shape == (1, 4)
    assert critic(state, actions).shape == (1, 1)
    assert not any(
        isinstance(module, torch.nn.AdaptiveAvgPool2d)
        for network in (*actors, critic)
        for module in network.modules()
    )


def test_resize_is_deterministic_and_preserves_batch_axis() -> None:
    images = np.arange(2 * 12 * 8 * 3, dtype=np.uint8).reshape(2, 12, 8, 3)
    first = resize_uint8_images(images, 6, 4)
    second = resize_uint8_images(images, 6, 4)
    assert first.shape == (2, 3, 6, 4)
    assert np.array_equal(first, second)


def test_policy_cache_bank_has_distinct_recipient_donor_modules() -> None:
    actors = tiny_actors()
    caches = PolicyCacheBank(actors)
    assert caches.distinct_cache_modules() == 16
    assert caches.rollout_actor(2, 2, actors) is actors[2]
    assert caches.rollout_actor(2, 1, actors) is caches.cached_actor(2, 1)


def test_refresh_changes_only_selected_recipient_cache_and_charges_payload() -> None:
    actors = tiny_actors()
    caches = PolicyCacheBank(actors)
    with torch.no_grad():
        next(actors[1].parameters()).add_(0.2)
    caches.mark_owner_update(1)
    before = clone_parameter_groups(
        tuple(caches.cached_actor(0, donor) for donor in range(4))
    )
    action = apply_refresh_action(
        recipient=0, donors=(1,), actors=actors, caches=caches
    )
    cache_modules = tuple(caches.cached_actor(0, donor) for donor in range(4))
    assert changed_parameter_groups(before, cache_modules) == (1,)
    assert action.refresh_units == 1
    assert action.optional_policy_bytes == sum(
        parameter.numel() * parameter.element_size()
        for parameter in actors[1].parameters()
    )
    assert caches.age(0, 1) == 0


def test_delayed_packet_changes_only_owner_at_receipt() -> None:
    actors = tiny_actors()
    caches = PolicyCacheBank(actors)
    queue = OwnerGradientQueue()
    gradients = tuple(torch.ones_like(parameter) for parameter in actors[2].parameters())
    queue.launch(owner=2, launch_event=4, delay=3, gradients=gradients)
    before = clone_parameter_groups(actors)
    assert queue.apply_due(event=6, actors=actors, caches=caches, step=0.01) == ()
    applied = queue.apply_due(event=7, actors=actors, caches=caches, step=0.01)
    assert len(applied) == 1
    assert changed_parameter_groups(before, actors) == (2,)
    assert caches.current_versions == [0, 0, 1, 0]


def test_packet_gradient_is_immutable_after_launch() -> None:
    actors = tiny_actors()
    caches = PolicyCacheBank(actors)
    queue = OwnerGradientQueue()
    gradients = [torch.ones_like(parameter) for parameter in actors[1].parameters()]
    queue.launch(owner=1, launch_event=0, delay=1, gradients=gradients)
    for gradient in gradients:
        gradient.zero_()
    before = clone_parameter_groups(actors)
    queue.apply_due(event=1, actors=actors, caches=caches, step=0.01)
    assert changed_parameter_groups(before, actors) == (1,)


def test_packet_carried_weight_scales_receipt_update() -> None:
    actors = tiny_actors()
    caches = PolicyCacheBank(actors)
    queue = OwnerGradientQueue()
    gradients = tuple(torch.ones_like(parameter) for parameter in actors[1].parameters())
    before = tuple(parameter.detach().clone() for parameter in actors[1].parameters())
    queue.launch(
        owner=1,
        launch_event=0,
        delay=1,
        gradients=gradients,
        packet_weight=0.25,
    )
    queue.apply_due(event=1, actors=actors, caches=caches, step=0.04)
    for old, current in zip(before, actors[1].parameters()):
        assert torch.allclose(current, old - 0.01)


def test_zero_weight_receipt_preserves_charged_launch_refresh() -> None:
    actors = tiny_actors()
    caches = PolicyCacheBank(actors)
    with torch.no_grad():
        next(actors[1].parameters()).add_(0.2)
    caches.mark_owner_update(1)
    action = apply_refresh_action(
        recipient=0, donors=(1,), actors=actors, caches=caches
    )
    refreshed = tuple(
        parameter.detach().clone()
        for parameter in caches.cached_actor(0, 1).parameters()
    )
    before_actors = clone_parameter_groups(actors)
    queue = OwnerGradientQueue()
    queue.launch(
        owner=0,
        launch_event=0,
        delay=1,
        gradients=tuple(torch.ones_like(parameter) for parameter in actors[0].parameters()),
        packet_weight=0.0,
    )
    packets = queue.apply_due(event=1, actors=actors, caches=caches, step=0.04)
    assert len(packets) == 1
    assert action.optional_policy_bytes > 0
    assert changed_parameter_groups(before_actors, actors) == ()
    assert caches.current_versions[0] == 0
    assert all(
        torch.equal(saved, current.detach())
        for saved, current in zip(refreshed, caches.cached_actor(0, 1).parameters())
    )


def test_hard_budget_is_prefix_feasible() -> None:
    assert hard_budget_remaining(launches_after_action=1, budget_rate=0.5, spent_units=0) == 0
    assert hard_budget_remaining(launches_after_action=2, budget_rate=0.5, spent_units=0) == 1
    assert hard_budget_remaining(launches_after_action=5, budget_rate=0.5, spent_units=2) == 0


def test_age_and_mismatch_baselines_are_predictable() -> None:
    actors = tiny_actors()
    caches = PolicyCacheBank(actors)
    with torch.no_grad():
        next(actors[1].parameters()).add_(0.1)
        next(actors[2].parameters()).add_(0.3)
    caches.mark_owner_update(1)
    caches.mark_owner_update(2)
    caches.mark_owner_update(2)
    assert select_age_refresh(
        recipient=0, eligible_donors=(1, 2, 3), caches=caches, maximum_edges=1
    ) == (2,)
    assert select_mismatch_refresh(
        recipient=0,
        eligible_donors=(1, 2, 3),
        actors=actors,
        caches=caches,
        maximum_edges=1,
    ) == (2,)


def test_cache_lyapunov_ablation_trades_exact_reset_against_queue_price() -> None:
    actors = tiny_actors()
    caches = PolicyCacheBank(actors)
    with torch.no_grad():
        next(actors[1].parameters()).add_(0.1)
        next(actors[2].parameters()).add_(0.3)
    selected = select_cache_lyapunov_refresh(
        recipient=0,
        eligible_donors=(1, 2, 3),
        actors=actors,
        caches=caches,
        communication_queue=0.0,
        cache_debt_weight=1.0,
    )
    assert selected == (2,)
    rejected = select_cache_lyapunov_refresh(
        recipient=0,
        eligible_donors=(1, 2, 3),
        actors=actors,
        caches=caches,
        communication_queue=1e9,
        cache_debt_weight=1.0,
    )
    assert rejected == ()


def test_age_and_mismatch_do_not_spend_on_fresh_caches() -> None:
    actors = tiny_actors()
    caches = PolicyCacheBank(actors)
    assert select_age_refresh(
        recipient=0, eligible_donors=(1, 2, 3), caches=caches, maximum_edges=2
    ) == ()
    assert select_mismatch_refresh(
        recipient=0,
        eligible_donors=(1, 2, 3),
        actors=actors,
        caches=caches,
        maximum_edges=2,
    ) == ()


def test_cache_refresh_does_not_change_current_actors() -> None:
    actors = tiny_actors()
    caches = PolicyCacheBank(actors)
    before = clone_parameter_groups(actors)
    apply_refresh_action(recipient=0, donors=(1, 3), actors=actors, caches=caches)
    assert changed_parameter_groups(before, actors) == ()


def test_owner_critic_gradient_has_only_owner_structure() -> None:
    actors = tiny_actors()
    caches = PolicyCacheBank(actors)
    critic = PistonCentralCritic(4)
    observations = torch.zeros(2, 4, 3, 64, 32, dtype=torch.uint8)
    states = torch.zeros(2, 3, 64, 64, dtype=torch.uint8)
    gradients = owner_critic_gradient(
        owner=2,
        actors=actors,
        caches=caches,
        critic=critic,
        observations=observations,
        states=states,
    )
    assert len(gradients) == len(tuple(actors[2].parameters()))
    assert all(
        gradient.shape == parameter.shape
        for gradient, parameter in zip(gradients, actors[2].parameters())
    )


def test_signed_refresh_scans_one_sparse_cone_with_one_vjp() -> None:
    actors = tiny_actors()
    caches = PolicyCacheBank(actors)
    critic = PistonCentralCritic(4)
    with torch.no_grad():
        next(actors[1].parameters()).add_(0.05)
        next(actors[2].parameters()).sub_(0.03)
    observations = torch.zeros(2, 4, 3, 64, 32, dtype=torch.uint8)
    states = torch.zeros(2, 3, 64, 64, dtype=torch.uint8)
    choice = signed_refresh_for_batch(
        owner=0,
        eligible_donors=(1, 2),
        actors=actors,
        caches=caches,
        critic=critic,
        observations=observations,
        states=states,
        step=0.01,
        communication_queue=0.0,
        learning_weight=1.0,
        smoothness=1.0,
        packet_second_moment_upper=1.0,
        receipt_motion_upper=0.0,
        taylor_coefficient=0.0,
    )
    assert choice.candidate_count == 3
    assert choice.vjp_calls == 1
    assert choice.donor in (None, 1, 2)


def test_hard_budget_forces_a_null_without_cross_policy_vjp() -> None:
    actors = tiny_actors()
    caches = PolicyCacheBank(actors)
    critic = PistonCentralCritic(4)
    observations = torch.zeros(1, 4, 3, 64, 32, dtype=torch.uint8)
    states = torch.zeros(1, 3, 64, 64, dtype=torch.uint8)
    choice = signed_refresh_for_batch(
        owner=0,
        eligible_donors=(1, 2),
        actors=actors,
        caches=caches,
        critic=critic,
        observations=observations,
        states=states,
        step=0.01,
        communication_queue=0.0,
        learning_weight=1.0,
        smoothness=1.0,
        packet_second_moment_upper=1.0,
        receipt_motion_upper=0.0,
        taylor_coefficient=0.0,
        can_refresh=False,
    )
    assert choice.donor is None
    assert choice.candidate_count == 1
    assert choice.vjp_calls == 0


def test_td_target_is_detached_and_finite() -> None:
    actors = tiny_actors()
    target_actors = tuple(PistonActor() for _ in actors)
    critic = PistonCentralCritic(4)
    target_critic = PistonCentralCritic(4)
    observations = torch.zeros(2, 4, 3, 64, 32, dtype=torch.uint8)
    states = torch.zeros(2, 3, 64, 64, dtype=torch.uint8)
    actions = joint_policy_actions(actors=actors, observations=observations).detach()
    loss = critic_td_loss(
        critic=critic,
        target_critic=target_critic,
        target_actors=target_actors,
        states=states,
        actions=actions,
        rewards=torch.tensor([0.1, -0.2]),
        next_states=states,
        next_observations=observations,
        done=torch.tensor([0.0, 1.0]),
        discount=0.99,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in critic.parameters())
    assert all(parameter.grad is None for parameter in target_critic.parameters())
    assert all(
        parameter.grad is None
        for actor in target_actors
        for parameter in actor.parameters()
    )


def test_polyak_update_moves_targets_toward_sources() -> None:
    source = tiny_actors(2)
    targets = tuple(PistonActor() for _ in source)
    before = tuple(next(module.parameters()).detach().clone() for module in targets)
    polyak_update(source, targets, 0.25)
    for old, current, target in zip(before, source, targets):
        expected = 0.75 * old + 0.25 * next(current.parameters()).detach()
        assert torch.allclose(next(target.parameters()).detach(), expected)


def test_actual_pistonball_segment_enters_replay_and_builds_packet() -> None:
    from pettingzoo.butterfly import pistonball_v6

    torch.manual_seed(11)
    actors = tuple(PistonActor() for _ in range(4))
    caches = PolicyCacheBank(actors)
    environment = pistonball_v6.parallel_env(
        n_pistons=4,
        continuous=True,
        random_drop=False,
        random_rotate=False,
        max_cycles=8,
        render_mode=None,
    )
    try:
        observations, _ = environment.reset(seed=19)
        transitions, _, terminated = collect_parallel_segment(
            environment=environment,
            owner=1,
            actors=actors,
            caches=caches,
            observations=observations,
            horizon=2,
            exploration_std=0.1,
            rng=np.random.default_rng(23),
            device=torch.device("cpu"),
        )
    finally:
        environment.close()
    assert len(transitions) == 2
    assert not terminated
    replay = CompressedReplay(capacity=8)
    for transition in transitions:
        replay.add(transition)
    batch = replay.sample(2, np.random.default_rng(29), torch.device("cpu"))
    critic = PistonCentralCritic(4)
    gradients = owner_critic_gradient(
        owner=1,
        actors=actors,
        caches=caches,
        critic=critic,
        observations=batch.observations,
        states=batch.states,
    )
    queue = OwnerGradientQueue()
    queue.launch(owner=1, launch_event=0, delay=2, gradients=gradients)
    before = clone_parameter_groups(actors)
    queue.apply_due(event=2, actors=actors, caches=caches, step=0.001)
    assert changed_parameter_groups(before, actors) == (1,)
    assert batch.observations.shape == (2, 4, 3, 64, 32)
    assert batch.states.shape == (2, 3, 64, 64)
