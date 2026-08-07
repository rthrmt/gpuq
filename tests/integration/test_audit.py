"""Integration tests for the placement audit trail: gpuq records what
nvidia-smi showed on every managed GPU at the moment each job was placed,
and exposes it via `gpuq audit <job_id>`.
"""

import json
import os
import sqlite3

import pytest

from gpuq import protocol
from tests.conftest import wait_until

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork()")


def _placement_rows(daemon, job_id):
    conn = sqlite3.connect(daemon.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM placement_events WHERE job_id=? ORDER BY id", (job_id,)
    ).fetchall()
    conn.close()
    return rows


def _db_status(daemon, job_id):
    conn = sqlite3.connect(daemon.db_path)
    row = conn.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    conn.close()
    return row[0] if row else None


def test_placement_event_recorded_on_run(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    kind, job_id, _ = submit("echo hi")
    assert wait_until(lambda: _db_status(daemon, job_id) in ("done", "failed"))
    rows = _placement_rows(daemon, job_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["user"] == "tester"
    assert row["cmd"] == "echo hi"
    assert json.loads(row["gpu_ids"]) == [0]
    snapshot = json.loads(row["gpu_snapshot"])
    assert [g["index"] for g in snapshot] == [0]


def test_placement_event_covers_all_managed_gpus_not_just_chosen(daemon, submit):
    """The snapshot must show every managed GPU's state, not just the one(s)
    picked, so a later 'why did job X avoid GPU Y' question is answerable."""
    daemon.set_gpus([(0, 0, 24576, 24576), (1, 0, 24576, 24576)])
    kind, job_id, _ = submit("echo hi", gpus=1)
    assert wait_until(lambda: _db_status(daemon, job_id) in ("done", "failed"))
    rows = _placement_rows(daemon, job_id)
    snapshot = json.loads(rows[0]["gpu_snapshot"])
    assert {g["index"] for g in snapshot} == {0, 1}


def test_placement_event_incident_regression(daemon, submit):
    """Regression test for the motivating incident: a GPU with a pre-existing
    external process (under the idle threshold) sits alongside a fully free
    GPU. Prove the audit row would make the eligibility decision diagnosable:
    the busy GPU's pre-existing mem_used/process shows up in the snapshot
    even when it wasn't the one chosen."""
    # GPU 0: mostly free but has a few hundred MB in use by an external
    # process (under the default 500 MiB idle threshold, so still eligible
    # for an exclusive job) plus a fake compute process recorded against it.
    # GPU 1: fully idle 46 GiB card.
    daemon.set_gpus([(0, 100, 24576, 24476), (1, 0, 47104, 47104)])
    daemon.set_procs(0, [(4242, "stray_notebook", 100)])

    kind, job_id, _ = submit("echo hi", gpus=1)
    assert wait_until(lambda: _db_status(daemon, job_id) in ("done", "failed"))
    rows = _placement_rows(daemon, job_id)
    assert len(rows) == 1
    snapshot = json.loads(rows[0]["gpu_snapshot"])

    gpu0 = next(g for g in snapshot if g["index"] == 0)
    assert gpu0["mem_used"] == 100 * 1024 * 1024
    assert gpu0["processes"] == [
        {"pid": 4242, "process_name": "stray_notebook", "used_memory": 100 * 1024 * 1024}
    ]
    gpu1 = next(g for g in snapshot if g["index"] == 1)
    assert gpu1["mem_used"] == 0


def test_no_placement_event_for_queued_job(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    k1, j1, _ = submit("sleep 2")             # occupies the only GPU
    k2, j2, _ = submit("echo never")          # stays queued, never placed
    assert wait_until(lambda: _db_status(daemon, j1) == "running")
    assert _placement_rows(daemon, j2) == []


def test_audit_protocol_returns_events_for_job(daemon, submit):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    kind, job_id, _ = submit("echo hi")
    assert wait_until(lambda: _db_status(daemon, job_id) in ("done", "failed"))
    reply = daemon.request(protocol.audit(job_id))
    assert reply["type"] == "audit_response"
    events = reply["events"]
    assert len(events) == 1
    assert events[0]["job_id"] == job_id
    assert events[0]["gpu_ids"] == [0]
    assert len(events[0]["gpu_snapshot"]) == 1


def test_audit_protocol_empty_for_unknown_job(daemon):
    reply = daemon.request(protocol.audit("no-such-job"))
    assert reply["type"] == "audit_response"
    assert reply["events"] == []


def test_gpuq_audit_cli_prints_snapshot(daemon, submit, monkeypatch, capsys):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    kind, job_id, _ = submit("echo hi")
    assert wait_until(lambda: _db_status(daemon, job_id) in ("done", "failed"))

    from gpuq import cli

    monkeypatch.setenv("GPUQ_SOCKET", daemon.socket_path)
    args = cli.build_parser().parse_args(["audit", job_id])
    rc = cli.cmd_audit(args, daemon.socket_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert job_id[:4] in out
    assert "echo hi" in out
    assert "GPU" in out


def test_gpuq_audit_cli_resolves_short_prefix(daemon, submit, capsys):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    kind, job_id, _ = submit("echo hi")
    assert wait_until(lambda: _db_status(daemon, job_id) in ("done", "failed"))

    from gpuq import cli

    args = cli.build_parser().parse_args(["audit", job_id[:4]])
    rc = cli.cmd_audit(args, daemon.socket_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert job_id[:4] in out
