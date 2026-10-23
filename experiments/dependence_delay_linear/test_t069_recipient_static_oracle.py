import numpy as np

from experiments.dependence_delay_linear.run_t068a_exact_safe_mixing_scan import (
    execute_policy,
    load_config,
    scenario_rows,
)
from experiments.dependence_delay_linear.t069_recipient_static_oracle import (
    fixed_vector_components,
    registered_alpha_vectors,
    terminal_risks_from_components,
)


def test_registered_vector_count_and_order():
    vectors = registered_alpha_vectors([0.0, 0.5, 1.0], agents=3)
    assert vectors.shape == (27, 3)
    assert np.array_equal(vectors[0], np.zeros(3))
    assert np.array_equal(vectors[-1], np.ones(3))


def test_batched_scalar_vectors_match_t068_exact_runner():
    config = load_config()
    alpha = np.asarray(config["actions"]["alpha"], dtype=float)
    vectors = np.repeat(alpha[:, None], config["model"]["agents"], axis=1)
    for scenario in scenario_rows(config)[::173]:
        components = fixed_vector_components(config, scenario["delay"], vectors)
        risks = terminal_risks_from_components(config, scenario, components)
        direct = np.asarray(
            [
                execute_policy(config, scenario, policy="fixed", early_alpha=value)[
                    "terminal_risk"
                ]
                for value in alpha
            ]
        )
        assert np.allclose(risks, direct, rtol=2e-12, atol=2e-12)


def test_recipient_vector_oracle_contains_common_scalar_class():
    config = load_config()
    scenario = scenario_rows(config)[221]
    all_vectors = registered_alpha_vectors(
        config["actions"]["alpha"], config["model"]["agents"]
    )
    components = fixed_vector_components(config, scenario["delay"], all_vectors)
    vector_best = np.min(terminal_risks_from_components(config, scenario, components))
    scalar_best = min(
        execute_policy(config, scenario, policy="fixed", early_alpha=value)[
            "terminal_risk"
        ]
        for value in config["actions"]["alpha"]
    )
    assert vector_best <= scalar_best + 1e-12
