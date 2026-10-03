from __future__ import annotations

from .run_forecast_reversal_confirmation import expanded_grid, load_config


def test_frozen_confirmation_config_hashes_and_grid_are_valid() -> None:
    config = load_config()
    grid = expanded_grid(config)
    assert len(grid) == 24
    assert len({tuple(sorted(cell.items())) for cell in grid}) == 24
    assert config["confirmation_seed_count"] == 32
    assert set(config["development_seeds_excluded"]).isdisjoint(
        range(
            config["confirmation_seed_start"],
            config["confirmation_seed_start"] + config["confirmation_seed_count"],
        )
    )

