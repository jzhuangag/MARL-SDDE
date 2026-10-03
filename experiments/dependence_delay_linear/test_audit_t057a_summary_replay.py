from experiments.dependence_delay_linear.audit_t057a_summary_replay import execute


def test_strict_t057a_summary_replay_closes_f11_intent():
    result = execute()
    assert result["endpoint_rows"] == 21504
    assert result["unique_cells"] == 84
    assert result["unique_seeds"] == 256
    assert result["duplicate_cell_seed_rows"] == 0
    assert result["coverage_pass"]
    assert result["strict_full_summary_replay_pass"]
    assert result["maximum_absolute_numeric_difference"] < 1e-12
    assert result["F11_textual_intent_satisfied"]
    assert not result["formal_decision_changed"]
