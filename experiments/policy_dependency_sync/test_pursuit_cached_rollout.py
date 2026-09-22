from __future__ import annotations

import numpy as np
import torch

from .async_pistonball_ctde import PolicyCacheBank, apply_refresh_action
from .audit_pursuit_cache_semantics import PursuitInterfaceActor
from .pursuit_cached_rollout import pursuit_cached_profile_actions


def _force_action(actor: torch.nn.Module, action: int) -> None:
    final = actor.network[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(-10.0)
        final.bias[int(action)] = 10.0


def test_refresh_changes_owner_cached_profile_not_other_recipient() -> None:
    torch.manual_seed(219)
    actors = tuple(PursuitInterfaceActor() for _ in range(3))
    caches = PolicyCacheBank(actors)
    _force_action(caches.cached_actor(0, 1), 1)
    _force_action(caches.cached_actor(2, 1), 2)
    _force_action(actors[1], 3)
    caches.mark_owner_update(1)
    observations = {
        f"pursuer_{index}": np.zeros((7, 7, 3), dtype=np.float32)
        for index in range(3)
    }
    agents = tuple(observations)
    before_owner = pursuit_cached_profile_actions(
        observations=observations,
        possible_agents=agents,
        owner=0,
        actors=actors,
        caches=caches,
        device=torch.device("cpu"),
    )
    before_other = pursuit_cached_profile_actions(
        observations=observations,
        possible_agents=agents,
        owner=2,
        actors=actors,
        caches=caches,
        device=torch.device("cpu"),
    )
    action = apply_refresh_action(
        recipient=0, donors=(1,), actors=actors, caches=caches
    )
    after_owner = pursuit_cached_profile_actions(
        observations=observations,
        possible_agents=agents,
        owner=0,
        actors=actors,
        caches=caches,
        device=torch.device("cpu"),
    )
    after_other = pursuit_cached_profile_actions(
        observations=observations,
        possible_agents=agents,
        owner=2,
        actors=actors,
        caches=caches,
        device=torch.device("cpu"),
    )
    assert before_owner["pursuer_1"] == 1
    assert after_owner["pursuer_1"] == 3
    assert before_other["pursuer_1"] == after_other["pursuer_1"] == 2
    assert action.optional_policy_bytes > 0


def test_owner_always_uses_current_self_actor() -> None:
    torch.manual_seed(220)
    actors = tuple(PursuitInterfaceActor() for _ in range(2))
    caches = PolicyCacheBank(actors)
    _force_action(caches.cached_actor(1, 1), 0)
    _force_action(actors[1], 4)
    observations = {
        "pursuer_0": np.zeros((7, 7, 3), dtype=np.float32),
        "pursuer_1": np.zeros((7, 7, 3), dtype=np.float32),
    }
    actions = pursuit_cached_profile_actions(
        observations=observations,
        possible_agents=tuple(observations),
        owner=1,
        actors=actors,
        caches=caches,
        device=torch.device("cpu"),
    )
    assert actions["pursuer_1"] == 4
