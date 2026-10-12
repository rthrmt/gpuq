"""SQLite persistence layer for the job queue.

A single :class:`Database` instance owns one connection guarded by a lock, so
it is safe to call from the daemon's several threads. Rows are returned as
:class:`Job` dataclasses. The daemon treats the DB as the durable record of
truth for *what jobs exist and their status*; the live GPU reservation model
(:mod:`gpuq.gpustate`) is rebuilt from ``running`` rows on startup, and the
waiters of the ``queued``/``running`` rows are expected to reattach.
"""

import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

# Job statuses.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

ACTIVE_STATUSES = (QUEUED, RUNNING)
TERMINAL_STATUSES = (DONE, FAILED, CANCELLED)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id         TEXT PRIMARY KEY,
    user           TEXT NOT NULL,
    name           TEXT,
    cmd            TEXT NOT NULL,
    gpus_requested INTEGER NOT NULL DEFAULT 1,
    mem_requested  INTEGER,
    shareable      INTEGER NOT NULL DEFAULT 0,
    gpu_ids        TEXT,
    status         TEXT NOT NULL,
    log_file       TEXT NOT NULL,
    submitted_at   TEXT NOT NULL,
    started_at     TEXT,
    finished_at    TEXT,
    exit_code      INTEGER,
    priority       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_status    ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_user      ON jobs(user);
CREATE INDEX IF NOT EXISTS idx_submitted ON jobs(submitted_at);

CREATE TABLE IF NOT EXISTS placement_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id         TEXT NOT NULL,
    user           TEXT NOT NULL,
    cmd            TEXT NOT NULL,
    gpus_requested INTEGER NOT NULL,
    mem_requested  INTEGER,
    shareable      INTEGER NOT NULL DEFAULT 0,
    gpu_ids        TEXT NOT NULL,
    placed_at      TEXT NOT NULL,
    gpu_snapshot   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_placement_job  ON placement_events(job_id);
CREATE INDEX IF NOT EXISTS idx_placement_time ON placement_events(placed_at);
"""


@dataclass
class Job:
    job_id: str
    user: str
    name: Optional[str]
    cmd: str
    gpus_requested: int
    mem_requested: Optional[int]   # bytes per GPU; None for exclusive jobs w/o --mem
    shareable: bool
    gpu_ids: Optional[List[int]]
    status: str
    log_file: str
    submitted_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    exit_code: Optional[int]
    priority: int = 0              # higher runs first among the owner's queued jobs

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        gpu_ids = json.loads(row["gpu_ids"]) if row["gpu_ids"] else None
        return cls(
            job_id=row["job_id"],
            user=row["user"],
            name=row["name"],
            cmd=row["cmd"],
            gpus_requested=row["gpus_requested"],
            mem_requested=row["mem_requested"],
            shareable=bool(row["shareable"]),
            gpu_ids=gpu_ids,
            status=row["status"],
            log_file=row["log_file"],
            submitted_at=row["submitted_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            exit_code=row["exit_code"],
            priority=row["priority"],
        )


@dataclass
class PlacementEvent:
    id: int
    job_id: str
    user: str
    cmd: str
    gpus_requested: int
    mem_requested: Optional[int]
    shareable: bool
    gpu_ids: List[int]
    placed_at: str
    gpu_snapshot: list   # list of per-GPU dicts (index, mem_used, mem_free,
                         # mem_total, utilization_pct, processes)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "PlacementEvent":
        return cls(
            id=row["id"],
            job_id=row["job_id"],
            user=row["user"],
            cmd=row["cmd"],
            gpus_requested=row["gpus_requested"],
            mem_requested=row["mem_requested"],
            shareable=bool(row["shareable"]),
            gpu_ids=json.loads(row["gpu_ids"]),
            placed_at=row["placed_at"],
            gpu_snapshot=json.loads(row["gpu_snapshot"]),
        )


class Database:
    def __init__(self, path: str):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        self.init_schema()

    def init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created.

        ``CREATE TABLE IF NOT EXISTS`` leaves an existing table untouched, so a
        database written by an older gpuq lacks them. Caller holds the lock.
        """
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(jobs)")}
        if "priority" not in cols:
            self._conn.execute(
                "ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 0")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- writes -----------------------------------------------------------

    def insert_queued(
        self,
        *,
        job_id: str,
        user: str,
        name: Optional[str],
        cmd: str,
        gpus_requested: int,
        mem_requested: Optional[int],
        shareable: bool,
        log_file: str,
        submitted_at: str,
        priority: int = 0,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO jobs (job_id, user, name, cmd, gpus_requested,
                       mem_requested, shareable, gpu_ids, status, log_file,
                       submitted_at, started_at, finished_at, exit_code, priority)
                   VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL, NULL, NULL, ?)""",
                (job_id, user, name, cmd, gpus_requested, mem_requested,
                 1 if shareable else 0, QUEUED, log_file, submitted_at, priority),
            )
            self._conn.commit()

    def set_priority(self, job_id: str, priority: int) -> bool:
        """Set a *queued* job's priority. Returns False if it is no longer
        queued (the status check and the write are one statement, so a job the
        scheduler just started cannot be reprioritized after the fact)."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET priority=? WHERE job_id=? AND status=?",
                (priority, job_id, QUEUED),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def mark_running(self, job_id: str, gpu_ids: List[int], started_at: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, gpu_ids=?, started_at=? WHERE job_id=?",
                (RUNNING, json.dumps(gpu_ids), started_at, job_id),
            )
            self._conn.commit()

    def insert_placement_event(
        self,
        *,
        job_id: str,
        user: str,
        cmd: str,
        gpus_requested: int,
        mem_requested: Optional[int],
        shareable: bool,
        gpu_ids: List[int],
        placed_at: str,
        gpu_snapshot: list,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO placement_events (job_id, user, cmd, gpus_requested,
                       mem_requested, shareable, gpu_ids, placed_at, gpu_snapshot)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, user, cmd, gpus_requested, mem_requested,
                 1 if shareable else 0, json.dumps(gpu_ids), placed_at,
                 json.dumps(gpu_snapshot)),
            )
            self._conn.commit()

    def mark_done(self, job_id: str, exit_code: int, finished_at: str) -> None:
        status = DONE if exit_code == 0 else FAILED
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, exit_code=?, finished_at=? WHERE job_id=?",
                (status, exit_code, finished_at, job_id),
            )
            self._conn.commit()

    def mark_failed(self, job_id: str, finished_at: str, exit_code: Optional[int] = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, exit_code=?, finished_at=? WHERE job_id=?",
                (FAILED, exit_code, finished_at, job_id),
            )
            self._conn.commit()

    def mark_cancelled(self, job_id: str, finished_at: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, finished_at=? WHERE job_id=?",
                (CANCELLED, finished_at, job_id),
            )
            self._conn.commit()

    # --- reads ------------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return Job.from_row(row) if row else None

    def find_by_prefix(self, prefix: str) -> List[Job]:
        """Resolve a short id prefix (the CLI prints/accepts 4-char prefixes)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id LIKE ? ORDER BY submitted_at",
                (prefix + "%",),
            ).fetchall()
        return [Job.from_row(r) for r in rows]

    def queued_jobs_ordered(self) -> List[Job]:
        """Queued jobs in FIFO order (submission time, then id as a stable tie)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY submitted_at, job_id",
                (QUEUED,),
            ).fetchall()
        return [Job.from_row(r) for r in rows]

    def max_queued_priority(self, user: str,
                            exclude_job_id: Optional[str] = None) -> Optional[int]:
        """Highest priority among ``user``'s queued jobs (optionally ignoring
        one of them), or ``None`` if there are none."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(priority) AS p FROM jobs "
                "WHERE status=? AND user=? AND job_id IS NOT ?",
                (QUEUED, user, exclude_job_id),
            ).fetchone()
        return row["p"]

    def running_jobs(self) -> List[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY started_at", (RUNNING,)
            ).fetchall()
        return [Job.from_row(r) for r in rows]

    def active_jobs(self) -> List[Job]:
        """Queued + running, ordered for display (running first, then queue)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status IN (?, ?) "
                "ORDER BY (status=?) DESC, submitted_at, job_id",
                (QUEUED, RUNNING, RUNNING),
            ).fetchall()
        return [Job.from_row(r) for r in rows]

    def running_counts_by_user(self) -> Dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT user, COUNT(*) AS n FROM jobs WHERE status=? GROUP BY user",
                (RUNNING,),
            ).fetchall()
        return {r["user"]: r["n"] for r in rows}

    def history(self, limit: int = 20, user: Optional[str] = None) -> List[Job]:
        q = "SELECT * FROM jobs WHERE status IN (?, ?, ?)"
        params: list = [DONE, FAILED, CANCELLED]
        if user is not None:
            q += " AND user=?"
            params.append(user)
        q += " ORDER BY finished_at DESC, submitted_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        return [Job.from_row(r) for r in rows]

    def placement_events_for_job(self, job_id: str) -> List[PlacementEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM placement_events WHERE job_id=? ORDER BY id",
                (job_id,),
            ).fetchall()
        return [PlacementEvent.from_row(r) for r in rows]
