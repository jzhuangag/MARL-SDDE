"""Exact algebraic check for the HAPPO-prefix interaction correction.

This is a theorem-interface sanity test, not an RL performance experiment.
"""


def test_pairwise_prefix_correction_rejects_harmful_simultaneous_update():
    kappa = 3.0

    def team_return(a1, a2):
        return a1 + a2 - kappa * a1 * a2

    old = team_return(0, 0)
    candidate = team_return(1, 1)
    frozen_local_gain_1 = team_return(1, 0) - old
    frozen_local_gain_2 = team_return(0, 1) - old
    prefix_interaction = kappa

    # The sequential decomposition and the TV-oscillation correction agree
    # exactly; the uncorrected stale sum would falsely predict a gain of 2.
    sequential_term_1 = frozen_local_gain_1
    sequential_term_2 = team_return(1, 1) - team_return(1, 0)
    corrected_frozen_bound = (
        frozen_local_gain_1 + frozen_local_gain_2 - prefix_interaction
    )

    assert frozen_local_gain_1 == 1
    assert frozen_local_gain_2 == 1
    assert candidate - old == -1
    assert sequential_term_1 + sequential_term_2 == candidate - old
    assert corrected_frozen_bound == candidate - old
    assert corrected_frozen_bound < 0
