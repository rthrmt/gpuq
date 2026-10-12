"""Restarting gpuqd must not lose jobs.

A job's waiter process outlives the daemon: when its connection drops it
reconnects and ``reattach``es. These tests stop/kill and restart a real gpuqd
with jobs queued and running, and check that the queue, the GPU reservations
and the exit codes all carry over -- and that jobs whose waiter is really gone
are finalised once the grace period (4s in tests, see conftest) is over.
"""

import os
import signal
import sqlite3
import time

import pytest

from gpuq import protocol
from tests.conftest import kill_waiters, wait_until

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork()")

ONE_GPU = [(0, 0, 24576, 24576)]


def _db(daemon, job_id):
    """(status, exit_code) straight from the DB (covers finished jobs too)."""
    conn = sqlite3.connect(daemon.db_path)
    row = conn.execute("SELECT status, exit_code FROM jobs WHERE job_id=?",
                       (job_id,)).fetchone()
    conn.close()
    return tuple(row) if row else (None, None)


def _view(daemon):
    """What `gpuq status` shows, reduced to the parts a restart must preserve."""
    return [(j["job_id"], j["status"], j["gpu_ids"], j["queue_position"], j["priority"])
            for j in daemon.status()]


def _stamp(second):
    return f"2026-01-01T00:00:{second:02d}Z"


def _attached(daemon, n):
    return daemon.log().count("waiter reattached") == n


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL],
                         ids=["stopped", "crashed"])
def test_queue_and_running_job_survive_restart(daemon, submit, tmp_path, sig):
    daemon.set_gpus(ONE_GPU)
    order = tmp_path / "order.txt"
    _, holder, _ = submit(f"sleep 4; echo holder >> {order}")
    assert wait_until(lambda: _db(daemon, holder)[0] == "running")
    _, a, _ = submit(f"echo a >> {order}", submitted_at=_stamp(1))
    _, b, _ = submit(f"echo b >> {order}", submitted_at=_stamp(2))
    _, c, _ = submit(f"echo c >> {order}", submitted_at=_stamp(3), top=True)
    assert wait_until(lambda: len(daemon.status()) == 4)
    before = _view(daemon)
    assert [j[0] for j in before] == [holder, c, a, b]

    daemon.restart(sig)

    # Same jobs, same states, same GPU, same queue order and priorities.
    assert _view(daemon) == before
    assert wait_until(lambda: _attached(daemon, 4))

    # The holder still owns the GPU (the fake nvidia-smi reports it as idle, so
    # only the restored reservation keeps the queue off it); once it is done
    # the queue drains in its original order.
    assert wait_until(lambda: order.exists()
                      and len(order.read_text().split()) == 4, timeout=15)
    assert order.read_text().split() == ["holder", "c", "a", "b"]
    assert wait_until(lambda: all(_db(daemon, j) == ("done", 0)
                                  for j in (holder, a, b, c)))


def test_exit_code_reported_when_job_ends_while_daemon_is_down(daemon, submit):
    _, ok, _ = submit("sleep 1.5")
    _, bad, _ = submit("sleep 1.5; exit 3")
    assert wait_until(lambda: _db(daemon, ok)[0] == "running"
                      and _db(daemon, bad)[0] == "running")
    daemon.stop()
    time.sleep(3)            # both commands end with no daemon to tell
    assert _db(daemon, ok) == ("running", None)
    daemon.start()
    assert wait_until(lambda: _db(daemon, ok) == ("done", 0))
    assert wait_until(lambda: _db(daemon, bad) == ("failed", 3))
    assert daemon.status() == []


def test_running_job_can_be_cancelled_after_restart(daemon, submit):
    daemon.set_gpus(ONE_GPU)
    _, job, _ = submit("sleep 60")
    assert wait_until(lambda: _db(daemon, job)[0] == "running")
    daemon.restart()
    assert wait_until(lambda: _attached(daemon, 1))
    reply = daemon.request(protocol.cancel(job, "tester", force=True))
    assert reply["type"] == "cancel_response"
    assert wait_until(lambda: _db(daemon, job)[0] == "cancelled", timeout=15)


def test_queued_job_cancelled_before_it_reattaches_stays_cancelled(daemon, submit):
    daemon.set_gpus(ONE_GPU)
    _, holder, _ = submit("sleep 60")
    assert wait_until(lambda: _db(daemon, holder)[0] == "running")
    _, queued, _ = submit("echo never")
    assert wait_until(lambda: len(daemon.status()) == 2)
    daemon.stop()
    # Cancel straight in the database, as if the request had been handled just
    # before the daemon went down: the waiter must be turned away, not revived.
    conn = sqlite3.connect(daemon.db_path)
    conn.execute("UPDATE jobs SET status='cancelled' WHERE job_id=?", (queued,))
    conn.commit()
    conn.close()
    daemon.start()
    assert wait_until(lambda: _attached(daemon, 1))
    assert [j["job_id"] for j in daemon.status()] == [holder]
    assert _db(daemon, queued)[0] == "cancelled"


def test_jobs_whose_waiter_is_gone_are_given_up_after_grace(daemon, submit):
    daemon.set_gpus(ONE_GPU)
    _, running, _ = submit("sleep 60")
    assert wait_until(lambda: _db(daemon, running)[0] == "running")
    _, queued, _ = submit("echo never")
    assert wait_until(lambda: len(daemon.status()) == 2)

    daemon.stop()
    kill_waiters(submit.waiter_pids)      # nobody is left to reattach
    daemon.start()

    # During the grace period nothing is written off, and the queue is held:
    # a new job must not overtake jobs that may still come back.
    _, fresh, _ = submit("echo fresh")
    assert _db(daemon, running)[0] == "running"
    assert _db(daemon, queued)[0] == "queued"
    assert _db(daemon, fresh)[0] == "queued"

    assert wait_until(lambda: _db(daemon, running)[0] == "failed", timeout=10)
    assert _db(daemon, queued)[0] == "cancelled"
    # ...and the GPU the lost job held is free again.
    assert wait_until(lambda: _db(daemon, fresh) == ("done", 0))


def _raw_conn(daemon):
    """A bare client socket standing in for a waiter. The timeout turns a
    message that never comes into a failed test instead of a hung one."""
    conn = daemon.connect()
    conn.settimeout(10)
    return conn, protocol.iter_msgs(conn)


def _raw_job(daemon, job_id, second=0):
    """Register a job over a bare socket. Returns (connection, message
    iterator) after the 'queued' ack."""
    conn, msgs = _raw_conn(daemon)
    protocol.send_msg(conn, protocol.register(
        job_id=job_id, user="tester", cmd="sleep 60", name=None, gpus_requested=1,
        mem_requested=None, shareable=False, log_file="/dev/null",
        submitted_at=_stamp(second)))
    assert next(msgs)["type"] == "queued"
    return conn, msgs


def _raw_reattach(daemon, job_id, state):
    conn, msgs = _raw_conn(daemon)
    protocol.send_msg(conn, protocol.reattach(job_id, "tester", state))
    return conn, msgs


def test_go_that_never_arrived_is_repeated_on_reattach(daemon):
    """The daemon records a placement before sending 'go'. If it dies in
    between, the waiter comes back still believing it is queued."""
    daemon.set_gpus(ONE_GPU)
    conn, msgs = _raw_job(daemon, "lost-go")
    assert wait_until(lambda: _db(daemon, "lost-go")[0] == "running")
    # Kill first, close after: the daemon must not get to see the EOF (it
    # would rightly treat that as the waiter dying). The 'go' is dropped unread.
    daemon.stop(signal.SIGKILL)
    conn.close()
    daemon.start()

    conn, msgs = _raw_reattach(daemon, "lost-go", protocol.STATE_QUEUED)
    assert next(msgs)["type"] == "reattached"
    assert next(msgs) == {"type": "go", "gpu_ids": [0]}
    protocol.send_msg(conn, protocol.done("lost-go", 0))
    assert next(msgs)["type"] == "done_ack"
    conn.close()
    assert _db(daemon, "lost-go") == ("done", 0)


def test_queue_is_held_until_older_jobs_have_reattached(daemon):
    """A job submitted right after a restart must not overtake an older queued
    job just because that job's waiter has not reconnected yet."""
    busy = [(0, 20000, 24576, 4576)]          # an external process on the GPU
    daemon.set_gpus(busy)
    conn, _ = _raw_job(daemon, "older", second=1)
    daemon.stop(signal.SIGKILL)               # kill first: it must not see the EOF
    conn.close()
    daemon.set_gpus(ONE_GPU)                  # the GPU is free when it returns
    daemon.start()

    fresh_conn, fresh = _raw_job(daemon, "fresh", second=2)
    time.sleep(1)                             # several scheduler wake-ups
    assert _db(daemon, "fresh")[0] == "queued"

    conn, msgs = _raw_reattach(daemon, "older", protocol.STATE_QUEUED)
    assert next(msgs)["type"] == "reattached"
    assert next(msgs)["type"] == "go"         # the older job gets the GPU
    assert _db(daemon, "fresh")[0] == "queued"
    protocol.send_msg(conn, protocol.done("older", 0))
    assert next(msgs)["type"] == "done_ack"
    assert next(fresh)["type"] == "go"
    conn.close()
    fresh_conn.close()


def test_reattach_refusals(daemon):
    daemon.set_gpus(ONE_GPU)

    reply = daemon.request(protocol.reattach("nope", "tester", protocol.STATE_QUEUED))
    assert reply["type"] == "error" and reply["code"] == "not_found"

    # A job whose waiter is attached cannot be taken over by a second client.
    conn, msgs = _raw_job(daemon, "live")
    assert next(msgs)["type"] == "go"
    for state in (protocol.STATE_QUEUED, protocol.STATE_RUNNING):
        reply = daemon.request(protocol.reattach("live", "tester", state))
        assert reply["type"] == "error" and reply["code"] == "not_reattachable"
    reply = daemon.request(protocol.reattach("live", "tester",
                                             protocol.STATE_FINISHED, exit_code=0))
    assert reply["type"] == "error" and reply["code"] == "not_reattachable"
    assert _db(daemon, "live")[0] == "running"

    reply = daemon.request(protocol.reattach("live", "tester", "bogus"))
    assert reply["type"] == "error" and reply["code"] == "bad_request"

    # Once it has finished it stays finished; repeating the report is harmless.
    protocol.send_msg(conn, protocol.done("live", 5))
    assert next(msgs)["type"] == "done_ack"
    conn.close()
    reply = daemon.request(protocol.reattach("live", "tester", protocol.STATE_RUNNING))
    assert reply["type"] == "error" and reply["code"] == "not_reattachable"
    reply = daemon.request(protocol.reattach("live", "tester",
                                             protocol.STATE_FINISHED, exit_code=0))
    assert reply["type"] == "reattached"
    assert _db(daemon, "live") == ("failed", 5)


def test_restart_check_against_running_daemon(daemon, submit, capsys):
    from gpuq.daemon import restart_check
    daemon.set_gpus(ONE_GPU)
    submit("sleep 60")
    assert wait_until(lambda: len(daemon.status()) == 1)
    assert restart_check(daemon.socket_path) == 0
    assert "1 active job(s); they will reattach" in capsys.readouterr().out
    daemon.stop()
    assert restart_check(daemon.socket_path) == 2
