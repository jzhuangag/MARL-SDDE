from __future__ import annotations

import numpy as np

from .run_harl_layer0_packet_smoke import (
    _discounted_returns,
    _service_duration,
)


def test_discounted_returns_preserve_terminal_reward() -> None:
    values = _discounted_returns(np.asarray([1.0, 2.0, 3.0]), 0.5)
    assert np.allclose(values, np.asarray([2.75, 3.5, 3.0]))


def test_service_law_is_positive_and_heterogeneous() -> None:
    values = [_service_duration(agent, 0) for agent in range(3)]
    assert 0.0 < values[0] < values[1] < values[2]
