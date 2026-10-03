"""Owner-specific cached joint behavior for asynchronous Pursuit rollouts."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .async_pistonball_ctde import PolicyCacheBank


def pursuit_cached_profile_actions(
    *,
    observations: Mapping[str, np.ndarray],
    possible_agents: Sequence[str],
    owner: int,
    actors: Sequence[nn.Module],
    caches: PolicyCacheBank,
    device: torch.device,
) -> dict[str, int]:
    """Evaluate the owner worker's current/self and cached/teammate profile."""

    if len(possible_agents) != len(actors):
        raise ValueError("possible-agent and actor counts must match")
    if not 0 <= int(owner) < len(actors):
        raise ValueError("owner index is out of range")
    actions: dict[str, int] = {}
    with torch.no_grad():
        for donor, agent in enumerate(possible_agents):
            if agent not in observations:
                continue
            actor = caches.rollout_actor(int(owner), donor, actors)
            observation = torch.as_tensor(
                observations[agent], dtype=torch.float32, device=device
            ).unsqueeze(0)
            logits = actor(observation)
            if logits.shape != (1, 5):
                raise ValueError("Pursuit actor must return five action logits")
            actions[agent] = int(torch.argmax(logits, dim=-1).item())
    return actions
