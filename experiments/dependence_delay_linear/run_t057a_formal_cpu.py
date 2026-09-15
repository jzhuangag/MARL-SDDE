"""Run the preregistered T-057A formal CPU experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.dependence_delay_linear import run_t053a_sampled_cpu_pilot as base
from experiments.dependence_delay_linear.run_t055a_confirmation_cpu_pilot import (
    run,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "docs" / "t057a_formal_cpu_preregistration.json"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "dependence_delay_linear"
    / "results"
    / "t057a_formal_cpu"
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    registry = config["formal_seed_registry"]
    start = int(registry["inclusive_start"])
    stop = int(registry["inclusive_end"])
    if registry["derivation"] != "every integer in the closed interval":
        raise ValueError("unsupported formal seed derivation")
    config["pilot_seeds"] = list(range(start, stop + 1))
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "estimate", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.mode == "validate":
        result = base.validate(config)
    elif arguments.mode == "estimate":
        result = base.estimate(config)
    else:
        result = run(config, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
