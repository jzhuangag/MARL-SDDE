from __future__ import annotations

import numpy as np
import pytest

from .owner_epoch_stationarity import owner_epoch_stationarity_lower_bound


def test_epoch_motion_bound_converts_all_selected_blocks_to_full_gradient() -> None:
    start = [np.asarray([1.0, -0.5]), np.asarray([0.2]), np.asarray([-0.4, 0.3])]
    selected = [
        start[0] + np.asarray([0.05, -0.02]),
        start[1] + np.asarray([-0.03]),
        start[2] + np.asarray([0.02, 0.04]),
    ]
    smoothness = [1.0, 0.5, 2.0]
    motion = [0.06, 0.06, 0.03]
    assert all(
        np.linalg.norm(after - before) <= lipschitz * path + 1e-12
        for before, after, lipschitz, path in zip(start, selected, smoothness, motion)
    )
    result = owner_epoch_stationarity_lower_bound(
        epoch_start_block_gradients=start,
        selected_launch_block_gradients=selected,
        block_smoothness=smoothness,
        launch_path_motion_upper=motion,
        stationarity_weights=[0.4, 0.7, 0.5],
    )
    assert result.selected_weighted_norm_squared + 1e-12 >= result.lower_bound
    assert result.motion_remainder == pytest.approx(
        0.7 * ((1.0 * 0.06) ** 2 + (0.5 * 0.06) ** 2 + (2.0 * 0.03) ** 2)
    )


def test_epoch_bound_is_order_free_but_requires_one_entry_per_owner() -> None:
    arguments = dict(
        epoch_start_block_gradients=[np.asarray([1.0]), np.asarray([2.0])],
        selected_launch_block_gradients=[np.asarray([1.0]), np.asarray([2.0])],
        block_smoothness=[1.0, 1.0],
        launch_path_motion_upper=[0.0, 0.0],
        stationarity_weights=[1.0, 1.0],
    )
    result = owner_epoch_stationarity_lower_bound(**arguments)
    assert result.lower_bound == pytest.approx(2.5)
    assert result.selected_weighted_norm_squared == pytest.approx(5.0)
    with pytest.raises(ValueError):
        owner_epoch_stationarity_lower_bound(
            **(arguments | {"block_smoothness": [1.0]})
        )
