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


def test_fail_all_running_crash_recovery(db):
    _add(db, "r1")
    _add(db, "q1")
    db.mark_running("r1", [0], "t")
    failed = db.fail_all_running("2026-01-01T00:05:00Z")
    assert failed == ["r1"]
    assert db.get_job("r1").status == FAILED
    assert db.get_job("q1").status == "queued"  # queued untouched


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
