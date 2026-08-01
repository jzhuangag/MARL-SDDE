import numpy as np

from exp017b_static_design import (
    FORMAL_SEEDS,
    PROBE_Q,
    PUBLIC_STRONG_FIXED_Q,
    action_charge,
    cached_candidate_features,
    learning_q,
    pairwise_trials,
    probe_trials_after_blocks,
    public_fallback_q,
)


def test_probe_is_independently_identifying() -> None:
    assert PROBE_Q >= 2
    assert pairwise_trials(PROBE_Q) > 0
    assert probe_trials_after_blocks(1, learning_q_value=1) > 0
    assert probe_trials_after_blocks(64, learning_q_value=1) > probe_trials_after_blocks(
        1, learning_q_value=1
    )


def test_no_evidence_fallback_is_public_strong_and_never_q1() -> None:
    for (task, budget), expected in PUBLIC_STRONG_FIXED_Q.items():
        assert public_fallback_q(task, budget) == expected
        assert learning_q(task, budget, evidence_ready=False, proposed_q=1) == expected
        assert expected >= 4


def test_probe_is_charged_to_both_resources() -> None:
    charge = action_charge(PROBE_Q, 1, parameters=4545)
    assert charge.message_bytes > 0
    assert charge.environment_steps == 1
    assert charge.agent_transitions == PROBE_Q


def test_evidence_based_learning_q_does_not_change_probe_trials() -> None:
    counts = [probe_trials_after_blocks(96, q) for q in (1, 4, 16, 32)]
    assert len(set(counts)) == 1


def test_candidate_features_are_vectorized_and_cached() -> None:
    first = cached_candidate_features(4545)
    second = cached_candidate_features(4545)
    assert first is second
    actions, message_cost, inverse_q = first
    assert actions.shape == (12, 2)
    assert message_cost.shape == (12,)
    assert inverse_q.shape == (12,)
    assert np.isfinite(inverse_q).all()


def test_static_design_assigns_no_formal_seeds() -> None:
    assert FORMAL_SEEDS is None
