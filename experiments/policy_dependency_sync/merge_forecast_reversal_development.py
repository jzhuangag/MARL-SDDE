"""Deterministically merge isolated forecast-reversal execution chunks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .run_forecast_reversal_development import POLICIES, default_grid, summarize_rows


def merge_chunks(
    *, output_dir: Path, chunk_dirs: Sequence[Path], expected_seeds: Sequence[int]
) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    for chunk_dir in chunk_dirs:
        with (chunk_dir / "endpoints.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    observed_seeds = sorted({int(row["seed"]) for row in rows})
    if observed_seeds != sorted(int(seed) for seed in expected_seeds):
        raise ValueError(
            f"chunk seeds {observed_seeds} do not match expected {list(expected_seeds)}"
        )
    identity = [
        (
            float(row["cycle_probability"]),
            int(row["maximum_delay"]),
            float(row["communication_budget"]),
            int(row["seed"]),
            str(row["policy"]),
        )
        for row in rows
    ]
    if len(identity) != len(set(identity)):
        raise ValueError("duplicate endpoint identity across chunks")
    rows.sort(
        key=lambda row: (
            float(row["cycle_probability"]),
            int(row["maximum_delay"]),
            float(row["communication_budget"]),
            int(row["seed"]),
            str(row["policy"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "endpoints.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_rows(rows)
    payload = {
        "grid": default_grid(),
        "seeds": list(expected_seeds),
        "policies": list(POLICIES),
    }
    summary["configuration_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary["endpoints_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-dir", type=Path, action="append", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    args = parser.parse_args()
    summary = merge_chunks(
        output_dir=args.output_dir,
        chunk_dirs=args.chunk_dir,
        expected_seeds=range(args.seed_start, args.seed_start + args.seed_count),
    )
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

