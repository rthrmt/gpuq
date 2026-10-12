"""Unit tests for the SQLite layer: transitions, ordering, fairness queries."""

import pytest

from gpuq.db import CANCELLED, DONE, FAILED, RUNNING, Database


@pytest.fixture
def db(tmp_path):
    d = Database(str(tmp_path / "q.db"))
    yield d
    d.close()


def _add(db, job_id, user="alice", ts="2026-01-01T00:00:00Z", **kw):
    db.insert_queued(
        job_id=job_id, user=user, name=kw.get("name"),
        cmd=kw.get("cmd", "echo hi"), gpus_requested=kw.get("gpus", 1),
        mem_requested=kw.get("mem"), shareable=kw.get("share", False),
        log_file=kw.get("log", "/tmp/x.log"), submitted_at=ts,
        priority=kw.get("priority", 0),
    )


def test_insert_and_get(db):
    _add(db, "j1", mem=8 * 1024 ** 3, share=True)
    job = db.get_job("j1")
    assert job.user == "alice"
    assert job.shareable is True
    assert job.mem_requested == 8 * 1024 ** 3
    assert job.status == "queued"
    assert job.gpu_ids is None


def test_queued_fifo_order(db):
    _add(db, "late", ts="2026-01-01T00:00:09Z")
    _add(db, "early", ts="2026-01-01T00:00:01Z")
    ids = [j.job_id for j in db.queued_jobs_ordered()]
    assert ids == ["early", "late"]


def test_mark_running_records_gpu_ids(db):
    _add(db, "j1")
    db.mark_running("j1", [0, 2], "2026-01-01T00:01:00Z")
    job = db.get_job("j1")
    assert job.status == RUNNING
    assert job.gpu_ids == [0, 2]
    assert job.started_at == "2026-01-01T00:01:00Z"


def test_mark_done_success_vs_failure(db):
    _add(db, "ok")
    _add(db, "bad")
    db.mark_done("ok", 0, "t")
    db.mark_done("bad", 3, "t")
    assert db.get_job("ok").status == DONE
    assert db.get_job("bad").status == FAILED
    assert db.get_job("bad").exit_code == 3


def test_running_counts_by_user(db):
    _add(db, "a1", "alice")
    _add(db, "a2", "alice")
    _add(db, "b1", "bob")
    db.mark_running("a1", [0], "t")
    db.mark_running("b1", [1], "t")
    counts = db.running_counts_by_user()
    assert counts == {"alice": 1, "bob": 1}


def test_history_filters_and_limits(db):
    for i in range(5):
        _add(db, f"d{i}", ts=f"2026-01-01T00:00:0{i}Z")
        db.mark_done(f"d{i}", 0, f"2026-01-01T01:00:0{i}Z")
    _add(db, "running")
    db.mark_running("running", [0], "t")
    hist = db.history(limit=3)
    assert len(hist) == 3
    assert all(j.status in (DONE, FAILED, CANCELLED) for j in hist)


def test_find_by_prefix(db):
    _add(db, "abcd1234")
    matches = db.find_by_prefix("abcd")
    assert len(matches) == 1 and matches[0].job_id == "abcd1234"


# --- priority ---------------------------------------------------------------

def test_priority_defaults_to_zero_and_round_trips(db):
    _add(db, "plain")
    _add(db, "urgent", priority=7)
    assert db.get_job("plain").priority == 0
    assert db.get_job("urgent").priority == 7


def test_set_priority_on_queued_job(db):
    _add(db, "j1")
    assert db.set_priority("j1", -3) is True
    assert db.get_job("j1").priority == -3


def test_set_priority_refuses_non_queued_jobs(db):
    _add(db, "run")
    _add(db, "fin")
    db.mark_running("run", [0], "t")
    db.mark_done("fin", 0, "t")
    assert db.set_priority("run", 5) is False
    assert db.set_priority("fin", 5) is False
    assert db.set_priority("missing", 5) is False
    assert db.get_job("run").priority == 0


def test_max_queued_priority_is_per_user_and_queued_only(db):
    _add(db, "a1", "alice", priority=2)
    _add(db, "a2", "alice", priority=9)
    _add(db, "a3", "alice", priority=50)
    _add(db, "b1", "bob", priority=100)
    db.mark_running("a3", [0], "t")          # no longer queued -> ignored
    assert db.max_queued_priority("alice") == 9
    assert db.max_queued_priority("alice", exclude_job_id="a2") == 2
    assert db.max_queued_priority("carol") is None


def test_database_from_before_priorities_is_migrated(tmp_path):
    """Upgrading gpuq must not need a fresh database: opening one created by a
    version without the priority column adds it and keeps the existing rows."""
    import sqlite3
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, user TEXT NOT NULL, name TEXT,
            cmd TEXT NOT NULL, gpus_requested INTEGER NOT NULL DEFAULT 1,
            mem_requested INTEGER, shareable INTEGER NOT NULL DEFAULT 0,
            gpu_ids TEXT, status TEXT NOT NULL, log_file TEXT NOT NULL,
            submitted_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
            exit_code INTEGER
        );
        INSERT INTO jobs (job_id, user, cmd, status, log_file, submitted_at)
        VALUES ('old1', 'alice', 'echo hi', 'queued', '/tmp/o.log', 't');
    """)
    conn.commit()
    conn.close()

    d = Database(path)
    try:
        assert d.get_job("old1").priority == 0
        assert d.set_priority("old1", 4) is True
        _add(d, "new1", priority=1)
        assert d.max_queued_priority("alice") == 4
    finally:
        d.close()
    # Re-opening an already migrated database is a no-op.
    d = Database(path)
    try:
        assert d.get_job("old1").priority == 4
    finally:
        d.close()


# --- placement_events -------------------------------------------------------

_SNAPSHOT = [
    {"index": 0, "mem_used": 512 * 1024 * 1024, "mem_free": 24 * 1024 ** 3,
     "mem_total": 24 * 1024 ** 3, "utilization_pct": 5,
     "processes": [{"pid": 1234, "process_name": "python", "used_memory": 512 * 1024 * 1024}]},
    {"index": 1, "mem_used": 0, "mem_free": 46 * 1024 ** 3,
     "mem_total": 46 * 1024 ** 3, "utilization_pct": 0, "processes": []},
]


def _add_placement_event(db, job_id, **kw):
    db.insert_placement_event(
        job_id=job_id, user=kw.get("user", "alice"), cmd=kw.get("cmd", "echo hi"),
        gpus_requested=kw.get("gpus", 1), mem_requested=kw.get("mem"),
        shareable=kw.get("share", False), gpu_ids=kw.get("gpu_ids", [1]),
        placed_at=kw.get("ts", "2026-01-01T00:01:00Z"),
        gpu_snapshot=kw.get("snapshot", _SNAPSHOT),
    )


def test_insert_placement_event_and_query_by_job(db):
    _add(db, "j1")
    _add_placement_event(db, "j1", gpu_ids=[1], mem=24 * 1024 ** 3)
    events = db.placement_events_for_job("j1")
    assert len(events) == 1
    ev = events[0]
    assert ev.job_id == "j1"
    assert ev.user == "alice"
    assert ev.cmd == "echo hi"
    assert ev.gpu_ids == [1]
    assert ev.mem_requested == 24 * 1024 ** 3
    assert ev.placed_at == "2026-01-01T00:01:00Z"


def test_placement_events_store_full_gpu_snapshot_json(db):
    """The snapshot round-trips through JSON, including the non-chosen GPU's
    state and nested process list -- this is the data that would have made
    the incident (job placed on a GPU another process was already using)
    diagnosable after the fact."""
    _add(db, "j1")
    _add_placement_event(db, "j1", gpu_ids=[1])
    ev = db.placement_events_for_job("j1")[0]
    assert ev.gpu_snapshot == _SNAPSHOT
    # GPU 0 (not chosen) still shows the pre-existing process.
    gpu0 = next(g for g in ev.gpu_snapshot if g["index"] == 0)
    assert gpu0["mem_used"] == 512 * 1024 * 1024
    assert gpu0["processes"][0]["pid"] == 1234


def test_placement_events_for_job_empty_when_none_recorded(db):
    _add(db, "j1")
    assert db.placement_events_for_job("j1") == []


def test_placement_events_for_job_only_matching_job(db):
    _add(db, "j1")
    _add(db, "j2")
    _add_placement_event(db, "j1")
    _add_placement_event(db, "j2")
    events = db.placement_events_for_job("j1")
    assert len(events) == 1
    assert events[0].job_id == "j1"


def test_placement_events_ordered_by_insertion(db):
    _add(db, "j1")
    _add_placement_event(db, "j1", ts="2026-01-01T00:01:00Z", gpu_ids=[0])
    _add_placement_event(db, "j1", ts="2026-01-01T00:02:00Z", gpu_ids=[1])
    events = db.placement_events_for_job("j1")
    assert [e.gpu_ids for e in events] == [[0], [1]]
