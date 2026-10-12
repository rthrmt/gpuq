"""Shared fixtures for integration tests.

The ``daemon`` fixture launches a real ``gpuqd`` in a subprocess against a temp
Unix socket and a temp SQLite DB, with ``nvidia-smi`` replaced by the fake shim
in ``tests/fakebin/nvidia-smi``. The shim reads its GPU state from a file the
test controls via ``GPUQ_FAKE_GPUS`` (see ``DaemonHandle.set_gpus``), and any
fake per-GPU compute-process list via ``GPUQ_FAKE_PROCS_<index>`` (see
``DaemonHandle.set_procs``), so a test can simulate any GPU layout without
real hardware.
"""

import os
import signal
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
    """gpus: list of (index, used_mib, total_mib, free_mib) or
    (index, used_mib, total_mib, free_mib, utilization_pct). The 4-tuple form
    defaults utilization_pct to 0."""
    lines = []
    for entry in gpus:
        if len(entry) == 4:
            i, u, t, f = entry
            util = 0
        else:
            i, u, t, f, util = entry
        lines.append(f"{i}, {u}, {t}, {f}, {util}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_procs(path, procs):
    """procs: list of (pid, process_name, used_mib)."""
    lines = [f"{pid}, {name}, {mib}" for (pid, name, mib) in procs]
    Path(path).write_text("\n".join(lines) + "\n")


@pytest.fixture
def gpu_state_file(tmp_path):
    p = tmp_path / "gpus.txt"
    # Default: two 24 GiB GPUs, idle.
    write_gpus(p, [(0, 0, 24576, 24576), (1, 0, 24576, 24576)])
    return p


class DaemonHandle:
    def __init__(self, socket_path, gpu_state_file, db_path, tmp_path, env):
        self.socket_path = socket_path
        self.gpu_state_file = gpu_state_file
        self.proc = None
        self.db_path = db_path
        self._tmp_path = tmp_path
        self._env = env
        self._starts = 0

    def start(self):
        """Launch gpuqd; fails the test if it does not come up."""
        self._starts += 1
        self.log_path = self._tmp_path / f"daemon_{self._starts}.log"
        with open(self.log_path, "wb") as log:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "gpuq.daemon", "--socket", self.socket_path,
                 "--db", self.db_path],
                env=self._env, stdout=log, stderr=subprocess.STDOUT,
            )
        if not _wait_for_socket(self.socket_path):
            self.proc.kill()
            pytest.fail(f"daemon did not start. output:\n{self.log_path.read_text()}")

    def stop(self, sig=signal.SIGTERM):
        """Stop gpuqd the way `systemctl stop` (SIGTERM) or a crash (SIGKILL)
        would, and wait until it is gone."""
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.send_signal(sig)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def restart(self, sig=signal.SIGTERM):
        self.stop(sig)
        self.start()

    def log(self):
        """Output of the currently running (most recently started) gpuqd."""
        return self.log_path.read_text()

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

    def set_procs(self, index, procs):
        """Set the fake compute-process list nvidia-smi reports for GPU
        ``index``. The daemon subprocess has GPUQ_FAKE_PROCS_<index> pointed
        at a fixed path per index (see the `daemon` fixture), so this just
        (re)writes that file's contents."""
        write_procs(self._tmp_path / f"procs_{index}.txt", procs)


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
    # Fixed per-index process-list paths (a real nvidia-smi is queried once
    # per GPU via --id=<index>); tests write to these via DaemonHandle.set_procs.
    # A path that doesn't exist yet is fine -- the shim treats that as "no
    # processes on this GPU", the common case.
    for i in range(8):
        env[f"GPUQ_FAKE_PROCS_{i}"] = str(tmp_path / f"procs_{i}.txt")
    # Short grace period: after a restart, jobs whose waiter does not reattach
    # are given up on after this long (waiters retry within a second or two).
    env.setdefault("GPUQ_REATTACH_GRACE", "4")
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    handle = DaemonHandle(socket_path, gpu_state_file, db_path, tmp_path, env)
    handle.start()
    yield handle
    handle.stop()


@pytest.fixture
def submit(daemon):
    """Submit a job through the real waiter path (fork+setsid in this process).

    Returns the job_id. Mirrors gpuq.cli.cmd_submit's handshake + fork, but
    drives it directly so tests stay in-process.
    """
    import datetime
    import uuid

    forked = []

    def _submit(cmd, *, name=None, gpus=1, mem=None, share=False, user="tester",
                priority=0, top=False, submitted_at=None):
        from gpuq import cli
        from gpuq.sizeutil import parse_size

        job_id = str(uuid.uuid4())
        mem_bytes = parse_size(mem) if mem else None
        log_file = str(Path(daemon.socket_path).parent / f"{job_id}.log")
        # submitted_at has 1s resolution; tests that depend on FIFO order pass
        # explicit, distinct stamps instead of sleeping between submissions.
        submitted_at = submitted_at or datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        reg = protocol.register(
            job_id=job_id, user=user, cmd=cmd, name=name, gpus_requested=gpus,
            mem_requested=mem_bytes, shareable=share, log_file=log_file,
            submitted_at=submitted_at, priority=priority, top=top,
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
                            msgs=msgs, sock_path=daemon.socket_path)
            os._exit(0)
        forked.append(pid)
        conn.close()
        return ("queued", job_id, log_file)

    _submit.waiter_pids = forked
    yield _submit
    kill_waiters(forked)


def kill_waiters(pids):
    """SIGKILL waiter processes and whatever command each is running.

    Waiters outlive the daemon by design (they keep trying to reattach), so
    tests have to stop them explicitly. The command sits in its own process
    group (the waiter starts it with start_new_session), hence killpg.
    """
    for pid in pids:
        children = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True,
                                  text=True).stdout.split()
        for target, kill in [(pid, os.kill)] + [(int(c), os.killpg) for c in children]:
            try:
                kill(target, signal.SIGKILL)
            except OSError:
                pass
        try:
            os.waitpid(pid, 0)
        except OSError:
            pass


def wait_until(predicate, timeout=8.0, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False
