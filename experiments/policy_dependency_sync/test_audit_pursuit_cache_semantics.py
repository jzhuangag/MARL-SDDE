from __future__ import annotations

import numpy as np
import torch

from .async_pistonball_ctde import PolicyCacheBank
from .audit_pursuit_cache_semantics import (
    PursuitInterfaceActor,
    _active_weights,
    _fixed_universe_state,
)
from .composite_cache_lyapunov import cache_mismatch_energy


def test_pursuit_interface_actor_shape() -> None:
    actor = PursuitInterfaceActor()
    assert actor(torch.zeros(3, 7, 7, 3)).shape == (3, 5)


def test_fixed_universe_energy_does_not_depend_on_candidate_support() -> None:
    torch.manual_seed(91)
    actors = tuple(PursuitInterfaceActor() for _ in range(3))
    caches = PolicyCacheBank(actors)
    with torch.no_grad():
        next(actors[1].parameters()).add_(0.2)
    current, cached = _fixed_universe_state(actors, caches)
    universe = tuple(cached)
    fixed = {edge: 1.0 for edge in universe}
    first = _active_weights(universe=universe, owner_donor_edges=((0, 1),))
    second = _active_weights(universe=universe, owner_donor_edges=((2, 1),))
    assert first != second
    fixed_energy = cache_mismatch_energy(current, cached, fixed)
    assert np.isfinite(fixed_energy)
    assert fixed_energy == cache_mismatch_energy(current, cached, fixed)
