from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from causal_packet_ledger import CausalPacketLedger, public_delay


def test_rejecting_a_returned_packet_does_not_refund_work_or_transmission():
    ledger = CausalPacketLedger(agents=2, optimizer_row_budget=100, message_byte_budget=80)
    ledger.charge_joint_rollout(25)
    job = ledger.launch(agent=0, service_ticks=3, optimizer_rows=100, message_bytes=80)
    with pytest.raises(ValueError):
        ledger.complete(job.job_id, [1.0])
    ledger.advance(3)
    ledger.complete(job.job_id, [1.0])
    update, drift = ledger.consume(job.job_id, alpha=0.0)
    np.testing.assert_array_equal(update, [0.0])
    assert drift == (0, 0)
    assert ledger.versions == [0, 0]
    assert ledger.accounting() == {"jobs_launched": 1, "packets_transmitted": 1,
        "packets_consumed": 1, "pending_jobs": 0, "ready_packets": 0,
        "optimizer_rows": 100, "reserved_message_bytes": 80,
        "transmitted_message_bytes": 80, "joint_transitions": 25, "actor_transitions": 50}
    with pytest.raises(ValueError):
        ledger.consume(job.job_id, alpha=1.0)
    with pytest.raises(ValueError):
        ledger.launch(agent=1, service_ticks=1, optimizer_rows=1, message_bytes=1)


def test_overlapping_jobs_record_cross_actor_policy_motion_and_arrive_once():
    ledger = CausalPacketLedger(agents=2, optimizer_row_budget=12, message_byte_budget=24)
    slow = ledger.launch(agent=0, service_ticks=4, optimizer_rows=4, message_bytes=8)
    fast = ledger.launch(agent=1, service_ticks=1, optimizer_rows=4, message_bytes=8)
    with pytest.raises(ValueError):
        ledger.launch(agent=0, service_ticks=1, optimizer_rows=1, message_bytes=1)
    ledger.advance(1)
    ledger.complete(fast.job_id, [2.0, 3.0])
    ledger.consume(fast.job_id, alpha=0.5)
    ledger.advance(4)
    returned = ledger.complete(slow.job_id, [1.0, 1.0])
    assert not returned.gradient.flags.writeable
    update, drift = ledger.consume(slow.job_id, alpha=0.5)
    assert drift == (0, 1)
    np.testing.assert_array_equal(update, [0.5, 0.5])
    with pytest.raises(ValueError):
        ledger.complete(slow.job_id, [1.0, 1.0])
    assert ledger.versions == [1, 1]


def test_service_sequences_do_not_share_the_cyclic_action_clock():
    sequences = {i: [public_delay(73000101, i, j) for j in range(32)] for i in range(3)}
    assert all(len(set(values)) > 1 for values in sequences.values())
    assert sequences[0] != sequences[1]
    reverse_order = {i: [public_delay(73000101, i, j) for j in range(32)] for i in reversed(range(3))}
    assert sequences == reverse_order


def test_a_different_actor_may_launch_while_a_completed_packet_waits():
    ledger = CausalPacketLedger(agents=2, optimizer_row_budget=10, message_byte_budget=10)
    first = ledger.launch(agent=0, service_ticks=1, optimizer_rows=2, message_bytes=2)
    ledger.advance(1)
    ledger.complete(first.job_id, [1.0])
    second = ledger.launch(agent=1, service_ticks=1, optimizer_rows=2, message_bytes=2)
    assert second.source_versions == (0, 0)
    ledger.consume(first.job_id, alpha=1.0)
    ledger.advance(2)
    ledger.complete(second.job_id, [1.0])
    _, drift = ledger.consume(second.job_id, alpha=1.0)
    assert drift == (1, 0)
