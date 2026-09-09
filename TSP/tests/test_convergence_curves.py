import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "TSP" / "experiments" / "run_convergence_curves.py"
SPEC = importlib.util.spec_from_file_location("tsp_convergence_curves", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_config_has_disjoint_seed_registries():
    config = json.loads(
        (ROOT / "TSP" / "experiments" / "convergence_config.json").read_text(
            encoding="utf-8"
        )
    )
    development = set(MODULE.seed_sequence(config, "development", None))
    confirmation = set(MODULE.seed_sequence(config, "confirmation", None))
    assert development
    assert confirmation
    assert development.isdisjoint(confirmation)


def test_action_table_is_finite_and_joint_is_catalogue_minimum():
    config = MODULE.load_config()
    table, _ = MODULE.action_table(config)
    numeric = table.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(numeric).all()
    for _, cell in table.groupby(["persistence", "rho", "maximum_delay"]):
        joint = cell[cell["policy"] == "joint"].iloc[0]
        fixed = cell[cell["policy"] != "joint"]
        assert joint["finite_time_bound"] <= fixed["finite_time_bound"].min() + 1e-14


def test_trace_starts_at_one_and_respects_budget():
    config = MODULE.load_config()
    config["persistences"] = [0.0]
    config["correlations"] = [0.0]
    config["maximum_delays"] = [0]
    config["candidate_agent_counts"] = [1]
    config["checkpoints"] = 5
    table, lookup = MODULE.action_table(config)
    action = lookup[(0.0, 0.0, 0, 1)]
    model = MODULE.build_transfer_mrp(0.0)
    streams = MODULE.generate_unit_paths(
        999, model, length=int(config["resource_budget"]), num_agents=1
    )
    targets = np.asarray(
        np.floor(np.linspace(0.0, 1.0, 5) * int(action["updates"])), dtype=np.int64
    )
    parameter, returns = MODULE._trace_kernel(
        streams["paths"],
        streams["masks"],
        model["features"],
        model["reward"],
        model["theta_star"],
        np.asarray([0], dtype=np.int64),
        0.0,
        int(action["gap"]),
        float(action["eta"]),
        targets,
    )
    initial_parameter = float(model["theta_star"].dot(model["theta_star"]))
    initial_return = float(model["features"][0].dot(model["theta_star"])) ** 2
    assert np.isclose(parameter[0] / initial_parameter, 1.0)
    assert np.isclose(returns[0] / initial_return, 1.0)
    assert int(targets[-1]) * int(action["update_cost"]) <= int(config["resource_budget"])
    assert np.isfinite(parameter).all()
    assert np.isfinite(returns).all()
