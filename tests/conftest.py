"""Shared fixtures for integration tests.

The ``daemon`` fixture launches a real ``gpuqd`` in a subprocess against a temp
Unix socket and a temp SQLite DB, with ``nvidia-smi`` replaced by the fake shim
in ``tests/fakebin/nvidia-smi``. The shim reads its GPU state from a file the
test controls via ``GPUQ_FAKE_GPUS``, so a test can simulate any GPU layout
without real hardware.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gpuq import protocol

REPO = Path(__file__).resolve().parent.parent
FAKE_SMI = REPO / "tests" / "fakebin" / "nvidia-smi"
GB = 1024  # the shim file uses MiB; 1024 MiB = 1 GiB


def write_gpus(path, gpus):
    """gpus: list of (index, used_mib, total_mib, free_mib)."""
    lines = [f"{i}, {u}, {t}, {f}" for (i, u, t, f) in gpus]
    Path(path).write_text("\n".join(lines) + "\n")


@pytest.fixture
def gpu_state_file(tmp_path):
    p = tmp_path / "gpus.txt"
    # Default: two 24 GiB GPUs, idle.
    write_gpus(p, [(0, 0, 24576, 24576), (1, 0, 24576, 24576)])
    return p


class DaemonHandle:
    def __init__(self, socket_path, gpu_state_file, proc, db_path):
        self.socket_path = socket_path
        self.gpu_state_file = gpu_state_file
        self.proc = proc
        self.db_path = db_path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.socket_path)
        return s

    def request(self, msg):
        s = self.connect()
        protocol.send_msg(s, msg)
        reply = protocol.recv_one(s)
        s.close()
        return reply

    def status(self):
        return self.request(protocol.status()).get("jobs", [])

    def set_gpus(self, gpus):
        write_gpus(self.gpu_state_file, gpus)


def _wait_for_socket(path, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(path)
                s.close()
                return True
            except OSError:
                pass
        time.sleep(0.02)
    return False


@pytest.fixture
def daemon(tmp_path, gpu_state_file):
    if not hasattr(os, "fork"):
        pytest.skip("requires fork()")
    socket_path = str(tmp_path / "gpuq.sock")
    db_path = str(tmp_path / "queue.db")
    env = dict(os.environ)
    env["GPUQ_SOCKET"] = socket_path
    env["GPUQ_DB"] = db_path
    env["GPUQ_NVIDIA_SMI"] = str(FAKE_SMI)
    env["GPUQ_FAKE_GPUS"] = str(gpu_state_file)
    # Fast poll so external GPU changes / rescheduling are picked up quickly.
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [sys.executable, "-m", "gpuq.daemon", "--socket", socket_path, "--db", db_path],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if not _wait_for_socket(socket_path):
        out = proc.stdout.read().decode() if proc.stdout else ""
        proc.kill()
        pytest.fail(f"daemon did not start. output:\n{out}")
    handle = DaemonHandle(socket_path, gpu_state_file, proc, db_path)
    yield handle
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def submit(daemon):
    """Submit a job through the real waiter path (fork+setsid in this process).

    Returns the job_id. Mirrors gpuq.cli.cmd_submit's handshake + fork, but
    drives it directly so tests stay in-process.
    """
    import datetime
    import uuid

    forked = []

    def _submit(cmd, *, name=None, gpus=1, mem=None, share=False, user="tester"):
        from gpuq import cli
        from gpuq.sizeutil import parse_size

        job_id = str(uuid.uuid4())
        mem_bytes = parse_size(mem) if mem else None
        log_file = str(Path(daemon.socket_path).parent / f"{job_id}.log")
        submitted_at = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        reg = protocol.register(
            job_id=job_id, user=user, cmd=cmd, name=name, gpus_requested=gpus,
            mem_requested=mem_bytes, shareable=share, log_file=log_file,
            submitted_at=submitted_at,
        )
        conn = daemon.connect()
        protocol.send_msg(conn, reg)
        # Read the ack through an iterator and hand that same iterator to the
        # waiter, exactly like cmd_submit: a second reader on this socket would
        # drop any message the first had already buffered (e.g. an early 'go').
        msgs = protocol.iter_msgs(conn)
        reply = next(msgs, None)
        if reply.get("type") == protocol.ERROR:
            conn.close()
            return ("error", reply)
        # Fork the waiter exactly like the CLI does.
        pid = os.fork()
        if pid == 0:
            os.setsid()
            args = type("A", (), {"name": name, "share": share})()
            cli._run_waiter(conn, job_id, cmd, log_file, user, args, mem_bytes,
                            msgs=msgs)
            os._exit(0)
        forked.append(pid)
        conn.close()
        return ("queued", job_id, log_file)

    yield _submit
    for pid in forked:
        try:
            os.waitpid(pid, os.WNOHANG)
        except OSError:
            pass


def wait_until(predicate, timeout=8.0, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False
