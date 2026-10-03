"""Atomic chunked execution of the frozen T-079 scientific computation."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.dependence_delay_linear.run_t070a_exact_nonstationary_graph_scan import (
    scenarios,
)
from experiments.dependence_delay_linear.run_t079_continuous_static_headroom import (
    DEFAULT_CONFIG as T079_CONFIG,
    analyze as analyze_t079,
    load_config as load_t079_config,
    run_cell as run_t079_cell,
    source_cells,
)
from experiments.dependence_delay_linear.run_t072_dual_use_architecture_calibration import (
    write_csv,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs/t080_chunked_continuous_static_execution.json"
DEFAULT_OUTPUT = ROOT / "experiments/dependence_delay_linear/results/t080_chunked_continuous_static_execution"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_scenarios(config: dict[str, Any], index: int) -> list[dict[str, Any]]:
    execution = config["execution"]
    chunks = int(execution["chunks"])
    if not 0 <= index < chunks:
        raise ValueError("chunk index outside frozen range")
    all_rows = scenarios(load_t079_config())
    width = int(execution["cells_per_chunk"])
    return all_rows[index * width:(index + 1) * width]


def validate(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    if sha256(T079_CONFIG) != source["t079_config_sha256"]:
        raise ValueError("T-079 configuration hash mismatch")
    execution = config["execution"]
    if execution["scientific_function"] != "run_t079_continuous_static_headroom.run_cell":
        raise ValueError("scientific cell function changed")
    if execution["analysis_function"] != "run_t079_continuous_static_headroom.analyze":
        raise ValueError("scientific analyzer changed")
    if execution["partial_chunk_interpretation_forbidden"] is not True:
        raise ValueError("partial interpretation must remain forbidden")
    partitions = [chunk_scenarios(config, index) for index in range(execution["chunks"])]
    identities = [row["cell_id"] for chunk in partitions for row in chunk]
    expected = [row["cell_id"] for row in scenarios(load_t079_config())]
    if identities != expected or len(set(identities)) != execution["ordered_cells"]:
        raise ValueError("chunk partition does not preserve exact ordered coverage")
    if any(len(chunk) != execution["cells_per_chunk"] for chunk in partitions):
        raise ValueError("chunk size mismatch")
    return {
        "experiment_id": config["experiment_id"],
        "chunks": execution["chunks"],
        "cells": len(identities),
        "ordered_exact_coverage": True,
        "scientific_configuration_unchanged": True,
    }


def run_chunk(config: dict[str, Any], output: Path, index: int) -> dict[str, Any]:
    validate(config)
    scientific = load_t079_config()
    sources = source_cells()
    selected = chunk_scenarios(config, index)
    payloads = [(scientific, row, sources[row["cell_id"]]) for row in selected]
    with ProcessPoolExecutor(max_workers=config["execution"]["workers_per_chunk"]) as pool:
        cells = list(pool.map(run_t079_cell, payloads, chunksize=1))
    chunk_dir = output / "chunks" / f"chunk-{index:02d}"
    chunk_dir.mkdir(parents=True, exist_ok=False)
    cells_path = chunk_dir / "cells.csv"
    write_csv(cells_path, cells)
    manifest = {
        "experiment_id": config["experiment_id"],
        "chunk_index": index,
        "cell_count": len(cells),
        "first_cell_id": cells[0]["cell_id"],
        "last_cell_id": cells[-1]["cell_id"],
        "cells_sha256": sha256(cells_path),
        "scientific_configuration_unchanged": True,
    }
    (chunk_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _read_chunk(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge(config: dict[str, Any], output: Path) -> dict[str, Any]:
    validate(config)
    all_cells: list[dict[str, str]] = []
    for index in range(config["execution"]["chunks"]):
        chunk_dir = output / "chunks" / f"chunk-{index:02d}"
        cells_path = chunk_dir / "cells.csv"
        manifest_path = chunk_dir / "manifest.json"
        if not cells_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"missing atomic chunk {index}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = _read_chunk(cells_path)
        if manifest["cell_count"] != len(rows) or manifest["cells_sha256"] != sha256(cells_path):
            raise ValueError(f"chunk {index} manifest mismatch")
        all_cells.extend(rows)
    expected = [row["cell_id"] for row in scenarios(load_t079_config())]
    if [row["cell_id"] for row in all_cells] != expected:
        raise ValueError("merged cell order or coverage mismatch")
    summary = analyze_t079(load_t079_config(), all_cells)
    cells_path = output / "cells.csv"
    summary_path = output / "summary.json"
    if cells_path.exists() or summary_path.exists():
        raise FileExistsError("merged output already exists")
    write_csv(cells_path, all_cells)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **summary,
        "chunk_count": config["execution"]["chunks"],
        "merged_cells_sha256": sha256(cells_path),
        "merged_summary_sha256": sha256(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "chunk", "merge"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-index", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "validate":
        result = validate(config)
    elif args.command == "chunk":
        if args.chunk_index is None:
            raise ValueError("chunk command requires --chunk-index")
        result = run_chunk(config, args.output, args.chunk_index)
    else:
        result = merge(config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
