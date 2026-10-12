"""End-to-end tests for job priorities: reordering one's own queued jobs.

Each test parks a long job on the only GPU so later submissions stay queued,
then inspects the queue the daemon reports and the order jobs actually run in.
"""

import os

import pytest

from gpuq import protocol
from tests.conftest import wait_until

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork()")

ONE_GPU = [(0, 0, 24576, 24576)]


def _queue(daemon):
    """Queued job ids in the order `gpuq status` lists them."""
    return [j["job_id"] for j in daemon.status() if j["status"] == "queued"]


def _job(daemon, job_id):
    return next(j for j in daemon.status() if j["job_id"] == job_id)


def _hold_gpu(daemon, submit, user="holder", seconds=60):
    """Occupy the single GPU and return the holder's job id."""
    daemon.set_gpus(ONE_GPU)
    _, holder, _ = submit(f"sleep {seconds}", user=user)
    assert wait_until(lambda: any(
        j["job_id"] == holder and j["status"] == "running" for j in daemon.status()))
    return holder


def _stamp(second):
    return f"2026-01-01T00:00:{second:02d}Z"


def test_reprioritized_job_runs_before_own_older_jobs(daemon, submit, tmp_path):
    holder = _hold_gpu(daemon, submit)
    order = tmp_path / "order.txt"
    ids = {}
    for n, name in enumerate(["first", "second", "urgent"]):
        _, ids[name], _ = submit(f"echo {name} >> {order}", submitted_at=_stamp(n))
    assert wait_until(lambda: len(_queue(daemon)) == 3)
    assert _queue(daemon) == [ids["first"], ids["second"], ids["urgent"]]

    reply = daemon.request(protocol.prioritize(ids["urgent"], "tester", top=True))
    assert reply["type"] == "prioritize_response"
    assert reply["previous"] == 0 and reply["priority"] == 1
    assert reply["queue_position"] == 1
    assert _queue(daemon) == [ids["urgent"], ids["first"], ids["second"]]
    assert _job(daemon, ids["urgent"])["priority"] == 1

    # Free the GPU: the queue must drain in the new order.
    daemon.request(protocol.cancel(holder, "holder", force=True))
    assert wait_until(lambda: order.exists()
                      and len(order.read_text().split()) == 3, timeout=15)
    assert order.read_text().split() == ["urgent", "first", "second"]


def test_priority_cannot_jump_ahead_of_another_user(daemon, submit, tmp_path):
    holder = _hold_gpu(daemon, submit)
    order = tmp_path / "order.txt"
    _, alice_job, _ = submit(f"echo alice >> {order}", user="alice",
                             submitted_at=_stamp(1))
    _, bob_job, _ = submit(f"echo bob >> {order}", user="bob",
                           submitted_at=_stamp(2), priority=1000)
    assert wait_until(lambda: len(_queue(daemon)) == 2)
    daemon.request(protocol.prioritize(bob_job, "bob", priority=protocol.PRIORITY_LIMIT))
    assert _queue(daemon) == [alice_job, bob_job]

    daemon.request(protocol.cancel(holder, "holder", force=True))
    assert wait_until(lambda: order.exists()
                      and len(order.read_text().split()) == 2, timeout=15)
    assert order.read_text().split() == ["alice", "bob"]


def test_submit_with_top_goes_ahead_of_own_queue(daemon, submit):
    _hold_gpu(daemon, submit)
    _, a, _ = submit("echo a", submitted_at=_stamp(1), priority=4)
    _, b, _ = submit("echo b", submitted_at=_stamp(2))
    _, urgent, _ = submit("echo urgent", submitted_at=_stamp(3), top=True)
    assert wait_until(lambda: len(_queue(daemon)) == 3)
    assert _queue(daemon) == [urgent, a, b]
    assert _job(daemon, urgent)["priority"] == 5   # one above the owner's highest


def test_explicit_priority_and_idempotent_top(daemon, submit):
    _hold_gpu(daemon, submit)
    _, a, _ = submit("echo a", submitted_at=_stamp(1))
    _, b, _ = submit("echo b", submitted_at=_stamp(2))
    assert wait_until(lambda: len(_queue(daemon)) == 2)

    reply = daemon.request(protocol.prioritize(a, "tester", priority=-5))
    assert reply["priority"] == -5 and reply["queue_position"] == 2
    assert _queue(daemon) == [b, a]

    # b is already ahead of everything else: --top must not inflate it.
    reply = daemon.request(protocol.prioritize(b, "tester", top=True))
    assert reply["priority"] == 0 and reply["previous"] == 0


def test_cross_user_prioritize_rejected(daemon, submit):
    _hold_gpu(daemon, submit)
    _, a, _ = submit("echo a", user="alice")
    assert wait_until(lambda: len(_queue(daemon)) == 1)
    reply = daemon.request(protocol.prioritize(a, "mallory", priority=9))
    assert reply["type"] == "error" and reply["code"] == "permission_denied"
    assert _job(daemon, a)["priority"] == 0


def test_root_may_prioritize_any_job(daemon, submit):
    _hold_gpu(daemon, submit)
    _, a, _ = submit("echo a", user="alice")
    assert wait_until(lambda: len(_queue(daemon)) == 1)
    reply = daemon.request(protocol.prioritize(a, "root", priority=3))
    assert reply["type"] == "prioritize_response" and reply["priority"] == 3


def test_prioritize_running_job_rejected(daemon, submit):
    holder = _hold_gpu(daemon, submit)
    reply = daemon.request(protocol.prioritize(holder, "holder", top=True))
    assert reply["type"] == "error" and reply["code"] == "not_queued"


def test_prioritize_unknown_job_rejected(daemon):
    reply = daemon.request(protocol.prioritize("no-such-job", "tester", priority=1))
    assert reply["type"] == "error" and reply["code"] == "not_found"


@pytest.mark.parametrize("bad", [None, "5", 1.5, True, protocol.PRIORITY_LIMIT + 1])
def test_prioritize_bad_value_rejected(daemon, submit, bad):
    _hold_gpu(daemon, submit)
    _, a, _ = submit("echo a")
    assert wait_until(lambda: len(_queue(daemon)) == 1)
    reply = daemon.request(protocol.prioritize(a, "tester", priority=bad))
    assert reply["type"] == "error" and reply["code"] == "bad_priority"
    assert _job(daemon, a)["priority"] == 0


def test_submit_with_out_of_range_priority_rejected(daemon, submit):
    daemon.set_gpus(ONE_GPU)
    result = submit("echo x", priority=protocol.PRIORITY_LIMIT + 1)
    assert result[0] == "error" and result[1]["code"] == "bad_priority"
