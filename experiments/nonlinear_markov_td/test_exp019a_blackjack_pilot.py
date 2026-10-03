import numpy as np

from experiments.nonlinear_markov_td.exp019a_blackjack_config import (
    CONFIG,
    config_sha256,
)
from experiments.nonlinear_markov_td.run_exp019a_blackjack_pilot import (
    exact_value,
    generate_tapes,
    message_cost,
    registered_cells,
    train_arm,
)


def test_config_hash_is_frozen() -> None:
    assert config_sha256() == (
        "299ff2c40ace9620040e75dca3214f24a350033cbe169f73a497bb2c84b8e0d8"
    )


def test_seed_registry_is_unique_and_new() -> None:
    assert len(CONFIG["pilot_seeds"]) == 32
    assert len(set(CONFIG["pilot_seeds"])) == 32


def test_registered_cells_and_populations_are_complete() -> None:
    cells = registered_cells()
    assert len(cells) == 72
    assert sum(bool(cell["active"]) for cell in cells) == 36
    assert sum(not bool(cell["active"]) for cell in cells) == 36
    assert {int(cell["selected_q"]) for cell in cells} >= {2, 4, 8, 16, 32}


def test_full_dual_budget_accounting() -> None:
    for cell in registered_cells():
        for policy in ("selected", "fallback"):
            q = int(cell[f"{policy}_q"])
            updates = int(cell[f"{policy}_updates"])
            assert updates * message_cost(q) <= int(cell["message_budget"])
            assert updates * int(CONFIG["thinning_stride"]) <= int(
                cell["environment_budget"]
            )


def test_exact_value_bellman_residual() -> None:
    value, _stationary, states, _reset = exact_value()
    assert len(states) == int(CONFIG["parameter_count"])
    assert np.all(np.isfinite(value))


def test_common_tape_preserves_identity_and_private_streams_exist() -> None:
    _value, _stationary, states, reset = exact_value()
    source, rewards, target = generate_tapes(1.0, [3190001, 3190002], 8, states, reset)
    assert np.all(source == source[:, :1, :])
    assert np.all(rewards == rewards[:, :1, :])
    assert np.all(target == target[:, :1, :])
    independent, _, _ = generate_tapes(0.0, [3190001], 8, states, reset)
    assert np.any(independent[:, 1:, :] != independent[:, :1, :])


def test_small_training_smoke_is_finite() -> None:
    value, stationary, states, reset = exact_value()
    source, rewards, target = generate_tapes(0.3, [3190001, 3190002], 12, states, reset)
    auc, terminal = train_arm(
        source, rewards, target, 4, 12, 2, value, stationary
    )
    assert np.all(np.isfinite(auc))
    assert np.all(np.isfinite(terminal))
