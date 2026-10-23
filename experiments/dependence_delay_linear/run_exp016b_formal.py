"""Execute EXP-016B formal replication from an implementation-frozen registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from analyze_exp016b_pilot import analyze
from run_exp016b_pilot import repository_root, run


REGISTRY = "docs/exp016b_formal_registry.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry() -> dict[str, object]:
    root = repository_root()
    registry = json.loads((root / REGISTRY).read_text(encoding="utf-8"))
    if registry.get("outcomes_present_at_freeze") is not False:
        raise RuntimeError("formal registry is outcome-tainted")
    for group in ("implementation_sha256", "frozen_input_sha256"):
        for relative, expected in registry[group].items():
            observed = sha256_file(root / relative)
            if observed != expected:
                raise RuntimeError(f"frozen hash mismatch for {relative}: {observed}")
    seeds = formal_seeds(registry)
    encoded = ",".join(str(seed) for seed in seeds).encode("ascii")
    if hashlib.sha256(encoded).hexdigest() != registry["formal_seed_list_sha256"]:
        raise RuntimeError("formal seed-list hash mismatch")
    if len(seeds) != int(registry["formal_seed_count"]) or len(seeds) != len(set(seeds)):
        raise RuntimeError("formal seed registry is incomplete or duplicated")
    return registry


def formal_seeds(registry: dict[str, object]) -> tuple[int, ...]:
    start = int(registry["formal_seed_start"])
    count = int(registry["formal_seed_count"])
    return tuple(range(start, start + count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "run", "analyze"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry()
    seeds = formal_seeds(registry)
    if args.command == "validate":
        print(json.dumps({"status": "PASS", "formal_seed_count": len(seeds)}, indent=2))
        return
    if args.command == "run":
        if args.output_dir is None:
            raise SystemExit("--output-dir is required for run")
        result = run(args.output_dir, seeds, args.workers, mode="formal")
    else:
        if args.metrics is None or args.output_dir is None:
            raise SystemExit("--metrics and --output-dir are required for analyze")
        result = analyze(
            args.metrics,
            args.output_dir,
            expected_seeds=seeds,
            task_label="EXP-016B-formal",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
