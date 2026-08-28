"""Resume all frozen T-080 chunks and merge after complete coverage."""

from __future__ import annotations

import json
from pathlib import Path
import time

from experiments.dependence_delay_linear.run_t080_chunked_continuous_static_execution import (
    DEFAULT_OUTPUT,
    load_config,
    merge,
    run_chunk,
    sha256,
    validate,
)


def completed_chunk(output: Path, index: int, expected_cells: int) -> bool:
    chunk = output / "chunks" / f"chunk-{index:02d}"
    cells = chunk / "cells.csv"
    manifest = chunk / "manifest.json"
    if not cells.is_file() or not manifest.is_file():
        return False
    record = json.loads(manifest.read_text(encoding="utf-8"))
    return (
        int(record["chunk_index"]) == index
        and int(record["cell_count"]) == expected_cells
        and record["cells_sha256"] == sha256(cells)
        and record["scientific_configuration_unchanged"] is True
    )


def main() -> None:
    config = load_config()
    validation = validate(config)
    print(json.dumps({"event": "validated", **validation}, sort_keys=True), flush=True)
    execution = config["execution"]
    output = DEFAULT_OUTPUT
    for index in range(int(execution["chunks"])):
        if completed_chunk(output, index, int(execution["cells_per_chunk"])):
            print(json.dumps({"event": "skip_verified_chunk", "chunk": index}), flush=True)
            continue
        started = time.perf_counter()
        print(json.dumps({"event": "start_chunk", "chunk": index}), flush=True)
        manifest = run_chunk(config, output, index)
        print(json.dumps({
            "event": "completed_chunk",
            "chunk": index,
            "runtime_seconds": time.perf_counter() - started,
            "manifest": manifest,
        }, sort_keys=True), flush=True)
    result = merge(config, output)
    print(json.dumps({"event": "merged", "result": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
