from __future__ import annotations

import numpy as np

from experiments.nonlinear_markov_td.analyze_t063a_reward_free_controller_formal import (
    breadth_inference,
    cluster_bootstrap_log_ratio,
    ratio_inference,
)
from experiments.nonlinear_markov_td.run_t061a_reward_free_controller_pilot import (
    load_config as load_pilot_config,
)
from experiments.nonlinear_markov_td.run_t063a_reward_free_controller_formal import (
    ROOT,
    run,
    run_game_seed,
)


def test_complete_cluster_bootstrap_is_deterministic() -> None:
    first = np.arange(1.0, 25.0).reshape(3, 8)
    second = first + 1.0
    left = cluster_bootstrap_log_ratio(first, second, replicates=100, seed=3)
    right = cluster_bootstrap_log_ratio(first, second, replicates=100, seed=3)
    np.testing.assert_array_equal(left, right)


def test_ratio_inference_accepts_uniform_gain() -> None:
    second = np.arange(1.0, 25.0).reshape(3, 8)
    first = 0.8 * second
    result = ratio_inference(
        first,
        second,
        threshold=0.95,
        upper_quantile=0.95,
        replicates=100,
        seed=4,
    )
    assert result["pass"]
    assert abs(result["point_ratio"] - 0.8) < 1e-14


def test_breadth_inference_resamples_seed_columns() -> None:
    second = np.ones((10, 8))
    first = 0.8 * second
    result = breadth_inference(
        first,
        second,
        threshold=0.60,
        lower_quantile=0.05,
        replicates=100,
        seed=5,
    )
    assert result["pass"]
    assert result["one_sided_lower_fraction"] == 1.0


def test_true_minatar_worker_smoke_uses_deterministic_order() -> None:
    config = load_pilot_config(
        ROOT / "docs" / "t061a_reward_free_controller_pilot_preregistration.json"
    )
    config = {
        **config,
        "tasks": {"asterix": config["tasks"]["asterix"]},
        "learning": {**config["learning"], "target_updates_qmax": 16},
        "probe": {**config["probe"], "blocks": 4, "length": 1},
        "grid": {
            **config["grid"],
            "correlations": [0.0, 0.5, 1.0],
            "overheads": [8],
            "delays": [0],
        },
    }
    rows = run_game_seed((config, "asterix", 123))
    assert [(row["rho"], row["overhead"], row["delay"]) for row in rows] == [
        (0.0, 8, 0),
        (0.5, 8, 0),
        (1.0, 8, 0),
    ]
    assert all(np.isfinite(row["controller_risk"]) for row in rows)


def test_two_process_minatar_smoke_writes_frozen_order(tmp_path) -> None:
    config = load_pilot_config(
        ROOT / "docs" / "t061a_reward_free_controller_pilot_preregistration.json"
    )
    config = {
        **config,
        "experiment_id": "T-063A-parallel-smoke",
        "pilot_seeds": [123, 124],
        "tasks": {"asterix": config["tasks"]["asterix"]},
        "learning": {**config["learning"], "target_updates_qmax": 16},
        "probe": {**config["probe"], "blocks": 4, "length": 1},
        "grid": {
            **config["grid"],
            "correlations": [0.0, 0.5, 1.0],
            "overheads": [8],
            "delays": [0],
        },
    }
    output = tmp_path / "parallel-smoke"
    result = run(config, output, workers=2)
    assert result["endpoints"] == 6
    assert result["workers"] == 2
    lines = (output / "endpoints.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 7
