"""Independent CPU confirmation of Multiwalker dynamic cache-value headroom."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from .multiwalker_budget_oracle_amendment import (
    ExactOracleRow,
    analyze_amendment,
    merge_exact_rows,
    run_exact_from_baselines,
)
from .multiwalker_oracle_headroom_dev import (
    POLICIES,
    ScenarioResult,
    analyze as analyze_baselines,
    run_development,
)


CONFIRMATION_SEEDS = tuple(range(96100, 96108))
DRIFT_SCALES = (0.01, 0.04)
BUDGET_RATES = (0.25, 0.5)
EVENTS = 40
HORIZON = 8
WALKERS = 5


def _canonical_write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load_records(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a record list: {path}")
    return payload


def _cell_key(row: Mapping[str, object]) -> tuple[int, float, float, str]:
    return (
        int(row["seed"]),
        float(row["drift_scale"]),
        float(row["budget_rate"]),
        str(row["policy"]),
    )


def analyze_confirmation(
    baseline_rows: Sequence[ScenarioResult], exact_rows: Sequence[ExactOracleRow]
) -> dict[str, object]:
    baseline = analyze_baselines(baseline_rows)
    exact = analyze_amendment(exact_rows)
    seeds = sorted({row.seed for row in exact_rows})
    expected_baseline = (
        len(CONFIRMATION_SEEDS)
        * len(DRIFT_SCALES)
        * len(BUDGET_RATES)
        * len(POLICIES)
    )
    expected_exact = (
        len(CONFIRMATION_SEEDS) * len(DRIFT_SCALES) * len(BUDGET_RATES)
    )
    gates = {
        "C1_frozen_seed_registry": tuple(seeds) == CONFIRMATION_SEEDS,
        "C2_complete_tables": (
            len(baseline_rows) == expected_baseline
            and len(exact_rows) == expected_exact
        ),
        "C3_baseline_finite_replay_budget": all(
            bool(baseline["gates"][key])
            for key in ("D1_complete_finite", "D2_exact_replay", "D3_prefix_budget")
        ),
        "C4_exact_optimizer_validity": all(
            bool(exact["gates"][key])
            for key in (
                "A1_complete_finite",
                "A2_exact_milp_optimum",
                "A3_no_refresh_replay",
                "A4_prefix_budget",
            )
        ),
        "C5_active_oracle_positive": bool(
            exact["gates"]["D4_active_oracle_positive"]
        ),
        "C6_active_recovery_at_least_10pct": bool(
            exact["gates"]["D5_active_recovery_gap"]
        ),
        "C7_active_direction_at_least_75pct": bool(
            exact["gates"]["D6_active_direction"]
        ),
        "C8_active_effect_at_least_0p1pct": bool(
            exact["gates"]["D7_active_absolute_effect"]
        ),
    }
    return {
        "status": "independent_multiwalker_oracle_headroom_confirmation",
        "seeds": seeds,
        "baseline_rows": len(baseline_rows),
        "exact_rows": len(exact_rows),
        "active_oracle_gain_over_no_refresh": exact[
            "active_oracle_gain_over_no_refresh"
        ],
        "active_strong_gain_over_no_refresh": exact[
            "active_strong_gain_over_no_refresh"
        ],
        "active_oracle_headroom_over_strong": exact[
            "active_oracle_headroom_over_strong"
        ],
        "active_oracle_recovery_gap": exact["active_oracle_recovery_gap"],
        "active_direction_rate": exact["active_direction_rate"],
        "active_median_normalized_absolute_headroom": exact[
            "active_median_normalized_absolute_headroom"
        ],
        "baseline_greedy_diagnostic_all_gates_pass": baseline["all_gates_pass"],
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "cell_metrics": exact["cell_metrics"],
    }


def run_chunk(output_dir: Path, seeds: Sequence[int]) -> None:
    normalized = tuple(sorted(set(int(seed) for seed in seeds)))
    if not normalized or len(normalized) != len(seeds):
        raise ValueError("chunk seeds must be nonempty and unique")
    if not set(normalized).issubset(CONFIRMATION_SEEDS):
        raise ValueError("chunk contains a seed outside the frozen registry")
    baseline_rows, baseline_summary = run_development(
        seeds=normalized,
        drift_scales=DRIFT_SCALES,
        budget_rates=BUDGET_RATES,
        events=EVENTS,
        horizon=HORIZON,
    )
    baseline_records = [asdict(row) for row in baseline_rows]
    exact_rows = run_exact_from_baselines(
        baseline_rows=baseline_records,
        seeds=normalized,
        drift_scales=DRIFT_SCALES,
        budget_rates=BUDGET_RATES,
        walkers=WALKERS,
        events=EVENTS,
        horizon=HORIZON,
    )
    baseline_hash = _canonical_write(
        output_dir / "baseline_rows.json", baseline_records
    )
    exact_hash = _canonical_write(
        output_dir / "exact_rows.json", [asdict(row) for row in exact_rows]
    )
    _canonical_write(output_dir / "baseline_summary.json", baseline_summary)
    _canonical_write(
        output_dir / "chunk_manifest.json",
        {
            "seeds": list(normalized),
            "baseline_rows_sha256": baseline_hash,
            "exact_rows_sha256": exact_hash,
            "events": EVENTS,
            "horizon": HORIZON,
            "drift_scales": list(DRIFT_SCALES),
            "budget_rates": list(BUDGET_RATES),
        },
    )


def merge_chunks(output_dir: Path, chunk_dirs: Sequence[Path]) -> dict[str, object]:
    baseline_records: list[dict[str, object]] = []
    exact_paths: list[Path] = []
    observed_seeds: set[int] = set()
    baseline_keys: set[tuple[int, float, float, str]] = set()
    for directory in chunk_dirs:
        manifest = json.loads(
            (directory / "chunk_manifest.json").read_text(encoding="utf-8")
        )
        chunk_seeds = {int(seed) for seed in manifest["seeds"]}
        if observed_seeds & chunk_seeds:
            raise ValueError("confirmation chunk seeds overlap")
        observed_seeds.update(chunk_seeds)
        baseline_path = directory / "baseline_rows.json"
        exact_path = directory / "exact_rows.json"
        if hashlib.sha256(baseline_path.read_bytes()).hexdigest().upper() != str(
            manifest["baseline_rows_sha256"]
        ):
            raise ValueError("baseline chunk hash mismatch")
        if hashlib.sha256(exact_path.read_bytes()).hexdigest().upper() != str(
            manifest["exact_rows_sha256"]
        ):
            raise ValueError("exact chunk hash mismatch")
        for record in _load_records(baseline_path):
            key = _cell_key(record)
            if key in baseline_keys:
                raise ValueError("duplicate baseline confirmation cell")
            baseline_keys.add(key)
            baseline_records.append(record)
        exact_paths.append(exact_path)
    if tuple(sorted(observed_seeds)) != CONFIRMATION_SEEDS:
        raise ValueError("merged confirmation seeds do not match frozen registry")
    baseline_records.sort(key=_cell_key)
    baseline_rows = [ScenarioResult(**record) for record in baseline_records]
    exact_rows = merge_exact_rows(exact_paths)
    summary = analyze_confirmation(baseline_rows, exact_rows)
    _canonical_write(output_dir / "baseline_rows.json", baseline_records)
    _canonical_write(
        output_dir / "exact_rows.json", [asdict(row) for row in exact_rows]
    )
    _canonical_write(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seeds", type=int, nargs="+")
    group.add_argument("--merge-inputs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.seeds:
        run_chunk(args.output_dir, args.seeds)
        print(json.dumps({"status": "chunk_complete", "seeds": args.seeds}, indent=2))
    else:
        summary = merge_chunks(args.output_dir, args.merge_inputs)
        print(
            json.dumps(
                {key: value for key, value in summary.items() if key != "cell_metrics"},
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
