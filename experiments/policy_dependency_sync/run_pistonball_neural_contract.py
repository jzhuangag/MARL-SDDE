"""Run and reproduce the outcome-free Pistonball neural contract smoke."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from .pistonball_neural_contract import (
    run_neural_contract,
    validate_neural_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[73001, 73002])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    results = [run_neural_contract(seed) for seed in args.seeds]
    rows = [result.to_dict() for result in results]
    validations = [validate_neural_contract(result) for result in results]
    gates = {
        key: all(validation[key] for validation in validations)
        for key in validations[0]
    }
    summary = {
        "experiment_id": "PISTONBALL-NEURAL-CONTRACT-DEV",
        "scientific_outcome": False,
        "gpu_authorized": False,
        "status": "pass" if all(gates.values()) else "fail",
        "seeds": list(args.seeds),
        "rows": rows,
        "gates": gates,
        "median_eligible_degree": float(
            np.median([len(row["eligible_donors"]) for row in rows])
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "limitations": [
            "random actors and critic: no return or learning-efficacy claim",
            "declared Taylor coefficient is not a certified neural Hessian bound",
            "single launch snapshot: no asynchronous trajectory training yet",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
