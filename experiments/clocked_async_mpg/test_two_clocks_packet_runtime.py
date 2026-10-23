from __future__ import annotations

import pytest

from .two_clocks_packet_runtime import PacketWork, TwoClocksPacketLedger


def _work(scale: int = 1) -> PacketWork:
    return PacketWork(8 * scale, 24 * scale, 11 * scale)


def test_single_flight_path_is_self_fresh_but_teammate_stale() -> None:
    ledger = TwoClocksPacketLedger(2)
    first = ledger.launch(
        0, launch_time=0.0, scheduled_completion_time=2.0, declared_work=_work()
    )
    second = ledger.launch(
        1, launch_time=0.0, scheduled_completion_time=1.0, declared_work=_work()
    )
    with pytest.raises(RuntimeError):
        ledger.launch(
            0, launch_time=0.0, scheduled_completion_time=3.0, declared_work=_work()
        )
    completed_second = ledger.complete(1, ticket_id=second.ticket_id, completion_time=1.0)
    assert completed_second.teammate_version_increments == (0, 0)
    ledger.apply(1, ticket_id=second.ticket_id)
    completed_first = ledger.complete(0, ticket_id=first.ticket_id, completion_time=2.0)
    assert completed_first.event_delay == 2
    assert completed_first.arrival_versions == (0, 1)
    assert completed_first.teammate_version_increments == (0, 1)
    ledger.apply(0, ticket_id=first.ticket_id)
    assert ledger.versions == (1, 1)
    ledger.assert_quiescent()


def test_only_matching_owner_and_ticket_can_apply() -> None:
    ledger = TwoClocksPacketLedger(2)
    ticket = ledger.launch(
        0, launch_time=0.0, scheduled_completion_time=1.0, declared_work=_work()
    )
    with pytest.raises(RuntimeError):
        ledger.complete(1, ticket_id=ticket.ticket_id, completion_time=1.0)
    ledger.complete(0, ticket_id=ticket.ticket_id, completion_time=1.0)
    with pytest.raises(RuntimeError):
        ledger.apply(1, ticket_id=ticket.ticket_id)
    with pytest.raises(RuntimeError):
        ledger.apply(0, ticket_id=ticket.ticket_id + 1)
    ledger.apply(0, ticket_id=ticket.ticket_id)
    ledger.assert_quiescent()


def test_completed_and_cancelled_work_are_both_charged() -> None:
    ledger = TwoClocksPacketLedger(2)
    completed = ledger.launch(
        0, launch_time=0.0, scheduled_completion_time=1.0, declared_work=_work(2)
    )
    cancelled = ledger.launch(
        1, launch_time=0.0, scheduled_completion_time=3.0, declared_work=_work(2)
    )
    ledger.complete(0, ticket_id=completed.ticket_id, completion_time=1.0)
    ledger.apply(0, ticket_id=completed.ticket_id)
    ledger.cancel(
        1,
        ticket_id=cancelled.ticket_id,
        charged_work=_work(),
        reason="registered horizon",
    )
    accounting = ledger.accounting()
    assert accounting["completed_work"] == (16, 48, 22)
    assert accounting["cancelled_work"] == (8, 24, 11)
    assert accounting["total_charged_work"] == (24, 72, 33)
    ledger.assert_quiescent()


def test_cancelled_work_cannot_exceed_predeclared_work() -> None:
    ledger = TwoClocksPacketLedger(1)
    ticket = ledger.launch(
        0, launch_time=0.0, scheduled_completion_time=1.0, declared_work=_work()
    )
    with pytest.raises(ValueError):
        ledger.cancel(
            0,
            ticket_id=ticket.ticket_id,
            charged_work=_work(2),
            reason="too much",
        )


def test_immediate_cancellation_may_charge_zero_executed_work() -> None:
    ledger = TwoClocksPacketLedger(1)
    ticket = ledger.launch(
        0, launch_time=0.0, scheduled_completion_time=1.0, declared_work=_work()
    )
    ledger.cancel(
        0,
        ticket_id=ticket.ticket_id,
        charged_work=PacketWork(0, 0, 0),
        reason="cancelled before execution",
    )
    assert ledger.accounting()["total_charged_work"] == (0, 0, 0)
    ledger.assert_quiescent()


def test_trace_is_deterministic_and_contains_no_outcome_field() -> None:
    traces = []
    for _ in range(2):
        ledger = TwoClocksPacketLedger(1)
        ticket = ledger.launch(
            0, launch_time=0.0, scheduled_completion_time=1.0, declared_work=_work()
        )
        ledger.complete(0, ticket_id=ticket.ticket_id, completion_time=1.0)
        ledger.apply(0, ticket_id=ticket.ticket_id)
        ledger.assert_quiescent()
        traces.append(ledger.trace)
    assert traces[0] == traces[1]
    prohibited = {"reward", "return", "win_rate", "gradient"}
    assert not any(prohibited.intersection(record) for record in traces[0])


def test_runtime_rejects_nonpredeclared_completion_time_and_dirty_teardown() -> None:
    ledger = TwoClocksPacketLedger(1)
    ticket = ledger.launch(
        0, launch_time=0.0, scheduled_completion_time=1.0, declared_work=_work()
    )
    with pytest.raises(ValueError):
        ledger.complete(0, ticket_id=ticket.ticket_id, completion_time=1.1)
    with pytest.raises(RuntimeError):
        ledger.assert_quiescent()
