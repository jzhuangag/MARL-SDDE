from experiments.nonlinear_markov_td.validate_t061a_reward_free_controller_pilot import (
    INTEGER_FIELDS,
    FLOAT_FIELDS,
)


def test_validator_schema_covers_claim_critical_numeric_fields() -> None:
    assert {"master_seed", "match_count", "selected_q", "controller_updates"} <= INTEGER_FIELDS
    assert {"rho", "controller_risk", "strong_risk"} <= FLOAT_FIELDS
