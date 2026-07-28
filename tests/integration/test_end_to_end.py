"""End-to-end integration tests against a real gpuqd over a temp socket.

GPU memory in the fake shim is expressed in MiB; 24576 MiB = 24 GiB.
"""

import os
import stat

import pytest

from tests.conftest import wait_until

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork()")


def _status_by_id(daemon, job_id):
    for j in daemon.status():
        if j["job_id"] == job_id:
            return j
    return None


def _db_status(daemon, job_id):
    """Read a job's status straight from the DB (covers finished jobs too)."""
    import sqlite3
    conn = sqlite3.connect(daemon.db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT status, exit_code FROM jobs WHERE job_id=?",
                        (job_id,)).fetchone()
    conn.close()
    return (row["status"], row["exit_code"]) if row else (None, None)


def test_socket_is_world_accessible(daemon):
    # No group gating: the socket must be mode 0666 so any local user can connect.
    # os.chmod in the daemon overrides the process umask, so this is exact.
    mode = stat.S_IMODE(os.stat(daemon.socket_path).st_mode)
    assert mode == 0o666


def test_job_lifecycle_and_log(daemon, submit, tmp_path):
    kind, job_id, log_file = submit("echo hello-gpuq && true")
    assert kind == "queued"
    # It should run and finish.
    assert wait_until(lambda: _db_status(daemon, job_id)[0] == "done")
    status, exit_code = _db_status(daemon, job_id)
    assert status == "done" and exit_code == 0
    # Log file exists with header + output.
    assert wait_until(lambda: os.path.exists(log_file))
    content = open(log_file).read()
    assert "# gpuq job log" in content
    assert "hello-gpuq" in content


def test_exit_code_propagates(daemon, submit):
    kind, job_id, _ = submit("exit 7")
    assert wait_until(lambda: _db_status(daemon, job_id)[0] in ("done", "failed"))
    status, exit_code = _db_status(daemon, job_id)
    assert status == "failed" and exit_code == 7


def test_two_exclusive_jobs_one_gpu_serialize(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])  # single GPU
    k1, j1, _ = submit("sleep 1")
    k2, j2, _ = submit("echo second")
    # Only one runs at a time; the second stays queued while the first runs.
    assert wait_until(lambda: _db_status(daemon, j1)[0] == "running")
    j2s = _status_by_id(daemon, j2)
    assert j2s is not None and j2s["status"] == "queued"
    # Eventually both complete.
    assert wait_until(lambda: _db_status(daemon, j2)[0] == "done", timeout=10)


def test_cuda_visible_devices_set(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    kind, job_id, log_file = submit('echo CVD=$CUDA_VISIBLE_DEVICES')
    assert wait_until(lambda: _db_status(daemon, job_id)[0] == "done")
    assert wait_until(lambda: os.path.exists(log_file))
    assert "CVD=0" in open(log_file).read()


def test_multigpu_waits_then_runs(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576), (1, 0, 24576, 24576)])
    kind, job_id, log_file = submit("echo multi", gpus=2)
    assert wait_until(lambda: _db_status(daemon, job_id)[0] == "done")
    # Both GPUs were assigned.
    import sqlite3, json
    conn = sqlite3.connect(daemon.db_path)
    row = conn.execute("SELECT gpu_ids FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    conn.close()
    assert sorted(json.loads(row[0])) == [0, 1]


def test_packing_two_share_jobs_colocate(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])  # one 24 GiB GPU
    k1, j1, _ = submit("sleep 1", share=True, mem="8G")
    k2, j2, _ = submit("sleep 1", share=True, mem="8G")
    assert k1 == "queued" and k2 == "queued"
    # Both should be running concurrently on GPU 0.
    assert wait_until(
        lambda: _db_status(daemon, j1)[0] == "running"
        and _db_status(daemon, j2)[0] == "running"
    )


def test_share_without_mem_rejected(daemon, submit):
    result = submit("echo x", share=True)  # no mem
    assert result[0] == "error"
    assert result[1]["code"] == "share_requires_mem"


def test_too_many_gpus_rejected(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    result = submit("echo x", gpus=99)
    assert result[0] == "error"
    assert result[1]["code"] == "too_many_gpus"


def test_mem_too_large_rejected(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])  # 24 GiB
    result = submit("echo x", mem="64G")
    assert result[0] == "error"
    assert result[1]["code"] == "mem_too_large"


def test_cancel_queued_job(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    k1, j1, _ = submit("sleep 2")            # occupies the only GPU
    k2, j2, _ = submit("echo never")         # stays queued
    from gpuq import protocol
    assert wait_until(lambda: _status_by_id(daemon, j2) is not None
                      and _status_by_id(daemon, j2)["status"] == "queued")
    reply = daemon.request(protocol.cancel(j2, "tester"))
    assert reply["type"] == "cancel_response" and reply["cancelled"] == 1
    assert wait_until(lambda: _db_status(daemon, j2)[0] == "cancelled")


def test_cancel_running_without_force_rejected(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    k1, j1, _ = submit("sleep 2")
    from gpuq import protocol
    assert wait_until(lambda: _db_status(daemon, j1)[0] == "running")
    reply = daemon.request(protocol.cancel(j1, "tester"))
    assert reply["type"] == "error" and reply["code"] == "cancel_requires_force"
    # The job must be untouched by the refused cancel.
    assert _db_status(daemon, j1)[0] == "running"


def test_cancel_running_with_force_kills_job(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    k1, j1, _ = submit("sleep 60")
    from gpuq import protocol
    assert wait_until(lambda: _db_status(daemon, j1)[0] == "running")
    reply = daemon.request(protocol.cancel(j1, "tester", force=True))
    assert reply["type"] == "cancel_response" and reply["cancelled"] == 1
    # The waiter kills the command and reports back through the normal done
    # path, well before the 60s sleep would have ended on its own. The job is
    # 'cancelled', not 'failed': the non-zero exit came from our own SIGTERM.
    assert wait_until(lambda: _db_status(daemon, j1)[0] == "cancelled", timeout=15)


def test_cancel_running_frees_gpu_for_next_job(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    k1, j1, _ = submit("sleep 60")          # holds the only GPU
    from gpuq import protocol
    assert wait_until(lambda: _db_status(daemon, j1)[0] == "running")
    k2, j2, _ = submit("echo second")       # queued behind it
    assert wait_until(lambda: _status_by_id(daemon, j2) is not None
                      and _status_by_id(daemon, j2)["status"] == "queued")
    daemon.request(protocol.cancel(j1, "tester", force=True))
    assert wait_until(lambda: _db_status(daemon, j2)[0] == "done", timeout=15)


def test_cancel_running_kills_grandchildren(daemon, submit, tmp_path):
    """shell=True means the direct child is /bin/sh; the real workload is a
    grandchild. Killing the process group must reap it too, or it would keep
    holding GPU memory after gpuq frees the reservation."""
    daemon.set_gpus([(0, 0, 24576, 24576)])
    marker = tmp_path / "grandchild.pid"
    # The inner sleep writes its pid, then outlives its parent shell if only
    # the direct child is signalled.
    k1, j1, _ = submit(f"sh -c 'echo $$ > {marker}; sleep 60'")
    from gpuq import protocol
    assert wait_until(lambda: _db_status(daemon, j1)[0] == "running")
    assert wait_until(lambda: marker.exists() and marker.read_text().strip())
    inner_pid = int(marker.read_text().strip())

    daemon.request(protocol.cancel(j1, "tester", force=True))
    assert wait_until(lambda: _db_status(daemon, j1)[0] == "cancelled", timeout=15)

    def inner_gone():
        try:
            os.kill(inner_pid, 0)
        except ProcessLookupError:
            return True
        return False

    assert wait_until(inner_gone, timeout=15), f"pid {inner_pid} survived cancel"


def test_cancel_finished_job_rejected(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    k1, j1, _ = submit("echo quick")
    from gpuq import protocol
    assert wait_until(lambda: _db_status(daemon, j1)[0] == "done")
    reply = daemon.request(protocol.cancel(j1, "tester", force=True))
    assert reply["type"] == "error" and reply["code"] == "not_cancellable"


def test_cross_user_cancel_running_rejected(daemon, submit):
    """Force must not become a way around the ownership check."""
    daemon.set_gpus([(0, 0, 24576, 24576)])
    k1, j1, _ = submit("sleep 60", user="alice")
    from gpuq import protocol
    assert wait_until(lambda: _db_status(daemon, j1)[0] == "running")
    reply = daemon.request(protocol.cancel(j1, "mallory", force=True))
    assert reply["type"] == "error" and reply["code"] == "permission_denied"
    assert _db_status(daemon, j1)[0] == "running"


def test_cross_user_cancel_rejected(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    k1, j1, _ = submit("sleep 2")
    k2, j2, _ = submit("echo q", user="alice")
    from gpuq import protocol
    assert wait_until(lambda: _status_by_id(daemon, j2) is not None
                      and _status_by_id(daemon, j2)["status"] == "queued")
    reply = daemon.request(protocol.cancel(j2, "mallory"))
    assert reply["type"] == "error" and reply["code"] == "permission_denied"


def test_fairness_bob_runs_alongside_alice(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576), (1, 0, 24576, 24576)])  # 2 GPUs
    # Alice submits 3 first, Bob 1 last; Bob (fewer running) should still get a GPU.
    ka1, ja1, _ = submit("sleep 2", user="alice")
    ka2, ja2, _ = submit("sleep 2", user="alice")
    ka3, ja3, _ = submit("sleep 2", user="alice")
    kb, jb, _ = submit("sleep 2", user="bob")

    def bob_and_one_alice_running():
        s = {j["job_id"]: j["status"] for j in daemon.status()}
        running = [jid for jid, st in s.items() if st == "running"]
        return jb in running and len(running) == 2

    assert wait_until(bob_and_one_alice_running, timeout=10)
