"""Crash-recovery: a DB 'running' job is marked failed when the daemon starts."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gpuq import protocol
from gpuq.db import Database

REPO = Path(__file__).resolve().parent.parent.parent
FAKE_SMI = REPO / "tests" / "fakebin" / "nvidia-smi"

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork()")


def _wait_for_socket(path, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(path)
            s.close()
            return True
        except OSError:
            time.sleep(0.02)
    return False


def test_startup_fails_inflight_jobs(tmp_path):
    db_path = str(tmp_path / "queue.db")
    socket_path = str(tmp_path / "gpuq.sock")
    gpus = tmp_path / "gpus.txt"
    gpus.write_text("0, 0, 24576, 24576\n")

    # Seed a 'running' job as if the daemon had crashed mid-flight.
    db = Database(db_path)
    db.insert_queued(job_id="ghost", user="alice", name="g", cmd="sleep 999",
                     gpus_requested=1, mem_requested=None, shareable=False,
                     log_file="/tmp/g.log", submitted_at="2026-01-01T00:00:00Z")
    db.mark_running("ghost", [0], "2026-01-01T00:00:01Z")
    db.close()

    env = dict(os.environ)
    env.update({
        "GPUQ_SOCKET": socket_path, "GPUQ_DB": db_path,
        "GPUQ_NVIDIA_SMI": str(FAKE_SMI), "GPUQ_FAKE_GPUS": str(gpus),
        "PYTHONPATH": str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", ""),
    })
    proc = subprocess.Popen(
        [sys.executable, "-m", "gpuq.daemon", "--socket", socket_path, "--db", db_path],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        assert _wait_for_socket(socket_path), "daemon failed to start"
        # The ghost job should now be 'failed'.
        db2 = Database(db_path)
        job = db2.get_job("ghost")
        db2.close()
        assert job.status == "failed"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
