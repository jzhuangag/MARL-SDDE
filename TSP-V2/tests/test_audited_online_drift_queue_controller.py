import importlib.util
import sys
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))
MODULE_PATH = EXPERIMENTS / "audited_online_drift_queue_controller.py"
SPEC = importlib.util.spec_from_file_location(
    "audited_online_drift_queue_controller", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _controller(window=3):
    base = sys.modules["online_drift_queue_controller"]
    return MODULE.AuditedSlidingWindowDriftQueueController(
        costs={
            1: base.OnlineActionCost(messages=10, environment_ticks=5),
            2: base.OnlineActionCost(messages=20, environment_ticks=5),
        },
        total_messages=200,
        total_environment_ticks=100,
        decisions=10,
        window=window,
        bonus_scale=0.1,
        probe_interval=4,
        progress_scale=0.1,
        queue_weight=10.0,
    )


def test_cumulative_counts_and_ledger_do_not_expand_statistical_window():
    controller = _controller(window=3)
    for step in range(10):
        decision = controller.decide()
        controller.observe(risk_before=0.6, risk_after=0.5 + 0.001 * step)
        assert controller.decision_ledger[-1]["q"] == decision.action
    assert sum(controller.selection_counts.values()) == 10
    assert len(controller.decision_ledger) == 10
    assert all(len(history) <= 3 for history in controller.histories.values())
    assert sum(row["messages"] for row in controller.decision_ledger) == (
        200 - controller.message_remaining
    )
    assert sum(row["environment_ticks"] for row in controller.decision_ledger) == (
        100 - controller.environment_remaining
    )


def test_audit_observers_do_not_change_frozen_decision_sequence():
    base = sys.modules["online_drift_queue_controller"]
    kwargs = dict(
        costs={
            1: base.OnlineActionCost(messages=10, environment_ticks=5),
            2: base.OnlineActionCost(messages=20, environment_ticks=5),
        },
        total_messages=200,
        total_environment_ticks=100,
        decisions=10,
        window=3,
        bonus_scale=0.1,
        probe_interval=4,
        progress_scale=0.1,
        queue_weight=10.0,
    )
    frozen = base.SlidingWindowDriftQueueController(**kwargs)
    audited = MODULE.AuditedSlidingWindowDriftQueueController(**kwargs)
    frozen_actions = []
    audited_actions = []
    for step in range(10):
        frozen_actions.append(frozen.decide().action)
        audited_actions.append(audited.decide().action)
        risk_after = 0.5 + 0.001 * step
        frozen.observe(risk_before=0.6, risk_after=risk_after)
        audited.observe(risk_before=0.6, risk_after=risk_after)
    assert audited_actions == frozen_actions
    assert audited.message_remaining == frozen.message_remaining
    assert audited.environment_remaining == frozen.environment_remaining
