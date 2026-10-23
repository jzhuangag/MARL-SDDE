import copy

from experiments.dependence_delay_linear.run_t080_chunked_continuous_static_execution import (
    chunk_scenarios,
    load_config,
    validate,
)


def test_t080_partition_is_ordered_complete_and_disjoint():
    config = load_config()
    result = validate(config)
    assert result["chunks"] == 12
    assert result["cells"] == 432
    chunks = [chunk_scenarios(config, index) for index in range(12)]
    assert all(len(chunk) == 36 for chunk in chunks)
    identities = [row["cell_id"] for chunk in chunks for row in chunk]
    assert len(identities) == len(set(identities)) == 432


def test_t080_rejects_changed_scientific_function():
    config = copy.deepcopy(load_config())
    config["execution"]["scientific_function"] = "replacement"
    try:
        validate(config)
    except ValueError as error:
        assert "scientific cell function" in str(error)
    else:
        raise AssertionError("changed scientific function was accepted")


def test_t080_rejects_partial_interpretation():
    config = copy.deepcopy(load_config())
    config["execution"]["partial_chunk_interpretation_forbidden"] = False
    try:
        validate(config)
    except ValueError as error:
        assert "partial interpretation" in str(error)
    else:
        raise AssertionError("partial interpretation was accepted")
