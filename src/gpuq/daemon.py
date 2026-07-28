"""``gpuqd`` — the root scheduler daemon.

Architecture (see CLAUDE.md for the full contract):

* The **acceptor** loop accepts connections and hands each to a thread.
* A **connection thread** for a ``register`` validates, inserts a queued row,
  records the connection in ``_pending``, signals the scheduler, then blocks
  reading for the ``done`` message (the socket stays open for the job's life).
  Short queries (status/cancel/history) are answered and the socket closed.
* A single **scheduler thread** waits on a condition (woken by register/done/
  cancel or the poll timer), refreshes live GPU state, and runs a scheduling
  pass. It sends ``go`` directly on the stored connection.

The daemon NEVER spawns or owns user processes; it only tells already-running
waiter processes when and where to run.

Invariant: all access to ``GpuState`` and ``_pending`` is guarded by
``_state_lock``; ``_wake`` (a Condition built on that lock) signals the
scheduler.
"""

import datetime
import logging
import os
import signal
import socket
import sys
import threading
from typing import Dict, List, Optional

from . import errors, protocol
from .config import Config, load_config
from .db import Database, Job
from .gpu import NvidiaSmiError, query_gpus
from .gpustate import GpuState
from .scheduler import SchedJob, schedule_pass

logger = logging.getLogger("gpuq.daemon")


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Pending:
    """A registered-but-not-yet-finished job and its live connection."""

    __slots__ = ("job", "conn", "write_lock")

    def __init__(self, job: Job, conn: socket.socket):
        self.job = job
        self.conn = conn
        self.write_lock = threading.Lock()

    def send(self, msg) -> None:
        with self.write_lock:
            protocol.send_msg(self.conn, msg)


class Daemon:
    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config.db_path)
        self._state_lock = threading.RLock()
        self._wake = threading.Condition(self._state_lock)
        self._pending: Dict[str, _Pending] = {}
        # Jobs we asked a waiter to kill; their non-zero exit is a cancellation,
        # not a failure. Guarded by _state_lock.
        self._terminating: set = set()
        self._server: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None

        managed = self._initial_managed_indices()
        self.state = GpuState(managed, config.idle_memory_threshold_bytes)
        logger.info("managing GPUs: %s", managed if managed else "(none detected)")

    # --- setup ------------------------------------------------------------

    def _initial_managed_indices(self) -> List[int]:
        if self.config.managed_gpus is not None:
            return list(self.config.managed_gpus)
        try:
            return [g.index for g in query_gpus()]
        except NvidiaSmiError as exc:
            logger.error("nvidia-smi unavailable at startup: %s", exc)
            return []

    def _refresh_live(self) -> None:
        """Refresh GpuState from nvidia-smi. Caller holds _state_lock."""
        try:
            live = query_gpus(self.state.managed_indices or None)
        except NvidiaSmiError as exc:
            logger.warning("nvidia-smi query failed: %s; treating GPUs as unavailable", exc)
            live = []
        self.state.reconcile(live)

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._reconcile_startup()
        self._open_socket()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, name="scheduler", daemon=True
        )
        self._scheduler_thread.start()
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        logger.info("gpuqd listening on %s", self.config.socket_path)
        self._accept_loop()

    def _reconcile_startup(self) -> None:
        """Crash recovery: any DB 'running' job lost its waiter connection when
        the daemon stopped, so we can no longer receive its 'done'. Mark such
        jobs failed and free their GPUs (the underlying processes may keep
        running until they exit on their own)."""
        failed = self.db.fail_all_running(_now_iso())
        if failed:
            logger.warning(
                "startup: marked %d in-flight job(s) failed (no reconnect "
                "protocol; their processes may still be running): %s",
                len(failed), ", ".join(s[:8] for s in failed),
            )
        with self._state_lock:
            self.state.clear()

    def _open_socket(self) -> None:
        path = self.config.socket_path
        sock_dir = os.path.dirname(path)
        os.makedirs(sock_dir, exist_ok=True)
        # The runtime directory must be world-traversable or non-root users
        # cannot reach the socket inside it, regardless of the socket's own mode.
        try:
            os.chmod(sock_dir, 0o755)
        except OSError as exc:
            logger.warning("could not chmod socket dir %s: %s", sock_dir, exc)
        if os.path.exists(path):
            os.unlink(path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(64)
        self._apply_socket_perms(path)
        self._server = srv

    def _apply_socket_perms(self, path: str) -> None:
        # No Unix group gating: any local user may connect (mode 0666 by
        # default). os.chmod overrides the process umask, so the configured
        # mode is applied exactly.
        try:
            os.chmod(path, self.config.socket_mode)
        except OSError as exc:
            logger.warning("could not chmod socket: %s", exc)

    def _handle_signal(self, signum, frame):
        logger.info("received signal %s, shutting down", signum)
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        with self._wake:
            self._wake.notify_all()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        try:
            if os.path.exists(self.config.socket_path):
                os.unlink(self.config.socket_path)
        except OSError:
            pass

    # --- accept loop ------------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                break  # socket closed during shutdown
            threading.Thread(
                target=self._handle_conn, args=(conn,), daemon=True
            ).start()

    # --- connection handling ----------------------------------------------

    def _handle_conn(self, conn: socket.socket) -> None:
        try:
            msg = protocol.recv_one(conn)
            if msg is None:
                conn.close()
                return
            mtype = protocol.get_type(msg)
            if mtype == protocol.REGISTER:
                self._handle_register(conn, msg)   # owns conn lifetime
            elif mtype == protocol.STATUS:
                self._reply_status(conn)
                conn.close()
            elif mtype == protocol.HISTORY:
                self._reply_history(conn, msg)
                conn.close()
            elif mtype == protocol.CANCEL:
                self._reply_cancel(conn, msg)
                conn.close()
            elif mtype == protocol.CANCEL_ALL:
                self._reply_cancel_all(conn, msg)
                conn.close()
            else:
                protocol.send_msg(conn, protocol.error(
                    errors.ERR_BAD_REQUEST, f"unknown message type: {mtype}"))
                conn.close()
        except errors.ProtocolError as exc:
            try:
                protocol.send_msg(conn, protocol.error(errors.ERR_BAD_REQUEST, str(exc)))
            except OSError:
                pass
            conn.close()
        except Exception:
            logger.exception("error handling connection")
            try:
                conn.close()
            except OSError:
                pass

    # --- register / job lifecycle -----------------------------------------

    def _validate_register(self, msg) -> None:
        gpus = msg.get("gpus_requested", 1)
        mem = msg.get("mem_requested")
        shareable = bool(msg.get("shareable", False))
        if not isinstance(gpus, int) or gpus < 1:
            raise errors.ValidationError(errors.ERR_BAD_GPUS, "gpus_requested must be >= 1")
        if shareable and mem is None:
            raise errors.ValidationError(
                errors.ERR_SHARE_REQUIRES_MEM, "--share requires --mem (e.g. --mem 8G)")
        with self._state_lock:
            # Refresh live GPU data so largest_gpu_memory() reflects reality
            # even before the first scheduler poll.
            self._refresh_live()
            num_managed = self.state.num_managed()
            largest = self.state.largest_gpu_memory()
        if num_managed == 0:
            raise errors.ValidationError(
                errors.ERR_TOO_MANY_GPUS, "no GPUs are currently managed by gpuqd")
        if gpus > num_managed:
            raise errors.ValidationError(
                errors.ERR_TOO_MANY_GPUS,
                f"requested {gpus} GPUs but only {num_managed} are managed")
        if mem is not None and largest and mem > largest:
            from .sizeutil import format_size
            raise errors.ValidationError(
                errors.ERR_MEM_TOO_LARGE,
                f"--mem exceeds the largest managed GPU ({format_size(largest)})")

    def _handle_register(self, conn: socket.socket, msg) -> None:
        try:
            self._validate_register(msg)
        except errors.ValidationError as exc:
            protocol.send_msg(conn, protocol.error(exc.code, exc.message))
            conn.close()
            return

        job = Job(
            job_id=msg["job_id"], user=msg["user"], name=msg.get("name"),
            cmd=msg["cmd"], gpus_requested=msg.get("gpus_requested", 1),
            mem_requested=msg.get("mem_requested"),
            shareable=bool(msg.get("shareable", False)), gpu_ids=None,
            status="queued", log_file=msg["log_file"],
            submitted_at=msg.get("submitted_at") or _now_iso(),
            started_at=None, finished_at=None, exit_code=None,
        )
        self.db.insert_queued(
            job_id=job.job_id, user=job.user, name=job.name, cmd=job.cmd,
            gpus_requested=job.gpus_requested, mem_requested=job.mem_requested,
            shareable=job.shareable, log_file=job.log_file,
            submitted_at=job.submitted_at,
        )
        position = self._queue_position(job.job_id)
        pending = _Pending(job, conn)
        with self._state_lock:
            self._pending[job.job_id] = pending
        try:
            pending.send(protocol.queued(position))
        except OSError:
            self._on_waiter_lost(job.job_id, still_queued=True)
            conn.close()
            return

        self._signal_scheduler()

        # Block reading on this connection until the waiter reports done or the
        # connection drops (EOF). The scheduler thread sends 'go' meanwhile.
        try:
            done = protocol.recv_one(conn)
        except errors.ProtocolError:
            done = None
        if done is not None and protocol.get_type(done) == protocol.DONE:
            self._on_done(job.job_id, int(done.get("exit_code", 1)))
        else:
            self._on_waiter_lost(job.job_id, still_queued=None)
        conn.close()

    def _on_done(self, job_id: str, exit_code: int) -> None:
        with self._state_lock:
            cancelled = job_id in self._terminating
            self._terminating.discard(job_id)
        # A job we killed exits non-zero by construction (SIGTERM/SIGKILL);
        # recording that as 'failed' would blame the user's command for our
        # signal, so cancellation wins over the exit code.
        if cancelled:
            self.db.mark_cancelled(job_id, _now_iso())
        else:
            self.db.mark_done(job_id, exit_code, _now_iso())
        with self._state_lock:
            self.state.release_job(job_id)
            self._pending.pop(job_id, None)
        logger.info("job %s %s (exit %d)", job_id[:8],
                    "cancelled" if cancelled else "done", exit_code)
        self._signal_scheduler()

    def _on_waiter_lost(self, job_id: str, still_queued) -> None:
        """The waiter connection dropped without a clean 'done'."""
        job = self.db.get_job(job_id)
        with self._state_lock:
            self.state.release_job(job_id)
            self._pending.pop(job_id, None)
            cancelled = job_id in self._terminating
            self._terminating.discard(job_id)
        if job is None:
            return
        if job.status == "queued":
            self.db.mark_cancelled(job_id, _now_iso())
            logger.info("job %s: waiter gone before start -> cancelled", job_id[:8])
        elif job.status == "running":
            if cancelled:
                # We killed it and the waiter died before reporting; still a
                # cancellation, not a failure.
                self.db.mark_cancelled(job_id, _now_iso())
                logger.info("job %s: waiter gone after terminate -> cancelled",
                            job_id[:8])
            else:
                self.db.mark_failed(job_id, _now_iso())
                logger.warning("job %s: waiter died while running -> failed", job_id[:8])
        self._signal_scheduler()

    # --- scheduler --------------------------------------------------------

    def _signal_scheduler(self) -> None:
        with self._wake:
            self._wake.notify_all()

    def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            with self._wake:
                self._wake.wait(timeout=self.config.poll_interval_seconds)
                if self._stop.is_set():
                    break
                self._run_schedule_pass_locked()

    def _run_schedule_pass_locked(self) -> None:
        """Run one scheduling pass. Caller holds _state_lock (via _wake)."""
        self._refresh_live()
        queued_rows = self.db.queued_jobs_ordered()
        # Only consider jobs whose waiter is still connected.
        candidates = [
            SchedJob(j.job_id, j.user, j.gpus_requested, j.mem_requested,
                     j.shareable, j.submitted_at)
            for j in queued_rows if j.job_id in self._pending
        ]
        if not candidates:
            return
        running_counts = self.db.running_counts_by_user()
        held = self._gpus_held_by_user()

        def commit(job: SchedJob, gpu_ids: List[int]) -> None:
            for idx in gpu_ids:
                self.state.reserve(idx, job.job_id, job.mem_requested or 0, job.shareable)
            self.db.mark_running(job.job_id, gpu_ids, _now_iso())
            pending = self._pending.get(job.job_id)
            if pending is None:
                self.state.release_job(job.job_id)
                self.db.mark_failed(job.job_id, _now_iso())
                return
            try:
                pending.send(protocol.go(gpu_ids))
                logger.info("job %s -> GPUs %s (%s)", job.job_id[:8], gpu_ids,
                            "shared" if job.shareable else "exclusive")
            except OSError:
                # Waiter died between register and go.
                self.state.release_job(job.job_id)
                self.db.mark_failed(job.job_id, _now_iso())
                self._pending.pop(job.job_id, None)

        schedule_pass(
            candidates, self.state, running_counts, commit=commit,
            max_gpus_per_user=self.config.max_gpus_per_user,
            gpus_held_by_user=held,
        )

    def _gpus_held_by_user(self) -> Dict[str, int]:
        by_user: Dict[str, set] = {}
        for j in self.db.running_jobs():
            by_user.setdefault(j.user, set()).add(j.job_id)
        return self.state.gpus_held_by_user_jobs(by_user)

    def _queue_position(self, job_id: str) -> int:
        for i, j in enumerate(self.db.queued_jobs_ordered(), start=1):
            if j.job_id == job_id:
                return i
        return 0

    # --- queries ----------------------------------------------------------

    def _job_to_dict(self, job: Job, queue_position: Optional[int]) -> dict:
        from .sizeutil import format_size
        return {
            "job_id": job.job_id, "user": job.user, "name": job.name,
            "cmd": job.cmd, "status": job.status, "gpu_ids": job.gpu_ids,
            "gpus_requested": job.gpus_requested,
            "mem_requested": job.mem_requested,
            "mem_requested_human": format_size(job.mem_requested) if job.mem_requested else None,
            "shareable": job.shareable, "queue_position": queue_position,
            "submitted_at": job.submitted_at, "started_at": job.started_at,
            "finished_at": job.finished_at, "exit_code": job.exit_code,
            "log_file": job.log_file,
        }

    def _reply_status(self, conn: socket.socket) -> None:
        active = self.db.active_jobs()
        queue_pos = {}
        pos = 0
        for j in active:
            if j.status == "queued":
                pos += 1
                queue_pos[j.job_id] = pos
        jobs = [self._job_to_dict(j, queue_pos.get(j.job_id)) for j in active]
        protocol.send_msg(conn, protocol.status_response(jobs))

    def _reply_history(self, conn: socket.socket, msg) -> None:
        limit = int(msg.get("limit") or 20)
        user = msg.get("user") if msg.get("mine") else None
        jobs = [self._job_to_dict(j, None) for j in self.db.history(limit, user)]
        protocol.send_msg(conn, protocol.history_response(jobs))

    def _reply_cancel(self, conn: socket.socket, msg) -> None:
        job_id = msg.get("job_id")
        requester = msg.get("requesting_user")
        job = self.db.get_job(job_id)
        if job is None:
            protocol.send_msg(conn, protocol.error(errors.ERR_NOT_FOUND, "no such job"))
            return
        if requester != "root" and job.user != requester:
            protocol.send_msg(conn, protocol.error(
                errors.ERR_PERMISSION, "you may only cancel your own jobs"))
            return
        if job.status == "queued":
            self._cancel_queued_job(job.job_id)
            protocol.send_msg(conn, protocol.cancel_response(1, job.name))
            return
        if job.status == "running":
            if not msg.get("force"):
                protocol.send_msg(conn, protocol.error(
                    errors.ERR_CANCEL_REQUIRES_FORCE,
                    "job is running; confirm to kill it"))
                return
            if self._terminate_running_job(job.job_id):
                protocol.send_msg(conn, protocol.cancel_response(1, job.name))
            else:
                protocol.send_msg(conn, protocol.error(
                    errors.ERR_NOT_CANCELLABLE,
                    "job is running but its waiter is unreachable"))
            return
        protocol.send_msg(conn, protocol.error(
            errors.ERR_NOT_CANCELLABLE,
            f"job is {job.status}; only queued or running jobs can be cancelled"))

    def _reply_cancel_all(self, conn: socket.socket, msg) -> None:
        requester = msg.get("requesting_user")
        count = 0
        for j in self.db.queued_jobs_ordered():
            if requester == "root" or j.user == requester:
                self._cancel_queued_job(j.job_id)
                count += 1
        protocol.send_msg(conn, protocol.cancel_all_response(count))

    def _terminate_running_job(self, job_id: str) -> bool:
        """Ask a running job's waiter to kill its command's process group.

        The waiter reports the killed command's exit code back through the
        normal ``done`` path, which is what marks the job finished and releases
        its GPUs — so this deliberately does not touch the DB or GpuState, and
        cannot race with a ``done`` already in flight for a job that finished
        on its own a moment ago. Returns False if the waiter is gone.
        """
        with self._state_lock:
            pending = self._pending.get(job_id)
            if pending is None:
                return False
            self._terminating.add(job_id)
        try:
            pending.send(protocol.terminate(job_id))
        except OSError:
            with self._state_lock:
                self._terminating.discard(job_id)
            return False
        logger.info("job %s: sent terminate to waiter", job_id[:8])
        return True

    def _cancel_queued_job(self, job_id: str) -> None:
        self.db.mark_cancelled(job_id, _now_iso())
        with self._state_lock:
            pending = self._pending.pop(job_id, None)
        if pending is not None:
            # Tell the waiter to give up, then drop the connection.
            try:
                pending.send(protocol.error(errors.ERR_NOT_CANCELLABLE, "cancelled"))
                pending.conn.close()
            except OSError:
                pass
        self._signal_scheduler()


def _setup_logging(config: Config) -> None:
    handlers = [logging.StreamHandler(sys.stderr)]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="gpuqd", description="gpuq scheduler daemon")
    parser.add_argument("--config", help="path to gpuq.conf")
    parser.add_argument("--socket", help="override socket path")
    parser.add_argument("--db", help="override database path")
    args = parser.parse_args(argv)

    config = load_config(conf_path=args.config, socket_path=args.socket, db_path=args.db)
    _setup_logging(config)
    daemon = Daemon(config)
    try:
        daemon.start()
    except KeyboardInterrupt:
        daemon.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
