from __future__ import annotations

from .run_harl_layer0_development_comparison import (
    _service_duration,
    _summarize,
)


def test_service_duration_preserves_heterogeneous_order() -> None:
    values = [_service_duration((1.0, 1.55, 4.0), agent, 0) for agent in range(3)]
    assert values[0] < values[1] < values[2]


def test_summary_uses_paired_higher_is_better_return() -> None:
    rows = []
    for profile in ("balanced", "heterogeneous"):
        for seed in (1, 2):
            rows.extend(
                [
                    {
                        "service_profile": profile,
                        "seed": seed,
                        "mode": "strategic_split",
                        "initial_return": -5.0,
                        "final_return": -2.0,
                        "return_change": 3.0,
                        "mean_scale": 0.5,
                    },
                    {
                        "service_profile": profile,
                        "seed": seed,
                        "mode": "raw_full_data",
                        "initial_return": -5.0,
                        "final_return": -3.0,
                        "return_change": 2.0,
                        "mean_scale": 1.0,
                    },
                    {
                        "service_profile": profile,
                        "seed": seed,
                        "mode": "raw_half_data",
                        "initial_return": -5.0,
                        "final_return": -4.0,
                        "return_change": 1.0,
                        "mean_scale": 1.0,
                    },
                ]
            )
    summary = _summarize(rows)
    contrast = summary["heterogeneous"]["strategic_contrasts"]["raw_full_data"]
    assert contrast["mean_paired_final_return_difference"] == 1.0
    assert contrast["strategic_strictly_better_fraction"] == 1.0
    assert contrast["lower_quartile_mean_return_difference"] == 1.0
    assert contrast["relative_mean_shortfall"] == 0.0
