"""``gpuq`` — the user CLI and the per-job *waiter* process.

``submit`` is the interesting path. To make submit-time errors visible in the
user's shell, we do a **pre-fork validation handshake**: connect, send
``register``, and wait for either a ``queued`` ack or an ``error``. Only after a
successful ack do we ``fork`` the long-lived waiter (which ``setsid``s to detach
from the terminal) and return the shell to the user. The waiter then blocks for
``go``, redirects output to the log file, ``exec``s the command as the user, and
finally reports ``done``.

The daemon never spawns the user's process — the waiter is forked from the
user's own shell, so it already runs as the right user.
"""

import argparse
import datetime
import getpass
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from typing import List, Optional

from . import protocol
from .config import client_socket_path
from .errors import ProtocolError
from .logfile import compute_log_path, ensure_logs_dir, write_header
from .sizeutil import format_size, parse_size


def _connect(sock_path: str) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    return s


def _die(msg: str, code: int = 1):
    print(f"gpuq: error: {msg}", file=sys.stderr)
    sys.exit(code)


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", str(os.getuid()))


def _short(job_id: str) -> str:
    return job_id[:4]


# --- submit ----------------------------------------------------------------

def cmd_submit(args, sock_path: str) -> int:
    if args.gpus < 1:
        _die("--gpus must be >= 1")
    mem_bytes: Optional[int] = None
    if args.mem is not None:
        try:
            mem_bytes = parse_size(args.mem)
        except ValueError as exc:
            _die(str(exc))
    if args.share and mem_bytes is None:
        _die("--share requires --mem (e.g. --mem 8G)")

    cmd = args.command
    user = _current_user()
    job_id = str(uuid.uuid4())
    when = datetime.datetime.now()
    log_file = compute_log_path(args.name, cmd, when=when)
    submitted_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    register = protocol.register(
        job_id=job_id, user=user, cmd=cmd, name=args.name,
        gpus_requested=args.gpus, mem_requested=mem_bytes,
        shareable=bool(args.share), log_file=log_file, submitted_at=submitted_at,
    )

    # --- pre-fork handshake: connect & confirm acceptance synchronously ----
    try:
        conn = _connect(sock_path)
    except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
        _die(f"cannot reach gpuqd at {sock_path}: {exc}")

    protocol.send_msg(conn, register)
    # This iterator owns the connection's read buffer and is handed to the
    # waiter across the fork; a second reader would drop anything it buffered.
    msgs = protocol.iter_msgs(conn)
    reply = next(msgs, None)
    if reply is None:
        _die("daemon closed the connection without responding")
    rtype = reply.get("type")
    if rtype == protocol.ERROR:
        conn.close()
        _die(reply.get("message", "submission rejected"))
    if rtype != protocol.QUEUED:
        conn.close()
        _die(f"unexpected reply from daemon: {reply!r}")

    position = reply.get("queue_position", 0)
    print(f"Job {_short(job_id)} queued (position {position})")
    extra = []
    if args.gpus > 1:
        extra.append(f"{args.gpus} GPUs")
    if mem_bytes:
        extra.append(format_size(mem_bytes))
    if args.share:
        extra.append("shared")
    if extra:
        print("  " + ", ".join(extra))
    print(f"Logs: {log_file}")

    # --- fork the long-lived waiter; parent returns to the shell -----------
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid > 0:
        # Parent: hand the socket to the child and return immediately.
        conn.close()
        return 0

    # Child (waiter): detach fully from the terminal so it survives the shell.
    os.setsid()
    _run_waiter(conn, job_id, cmd, log_file, user, args, mem_bytes, msgs=msgs)
    os._exit(0)


TERMINATE_GRACE_SECONDS = 10


def _watch_for_terminate(msgs, proc, logf) -> None:
    """Kill ``proc``'s process group when the daemon sends 'terminate'.

    Runs in a thread because the waiter's main thread is blocked in
    ``proc.wait()`` and cannot read the socket. ``msgs`` must be the *same*
    iterator the waiter used to read ``go``: a fresh ``recv_one`` would start a
    new generator and silently drop whatever that one had already buffered.
    On a clean exit the daemon closes the connection, the iterator ends, and
    this thread returns without touching the process.
    """
    try:
        msg = next(msgs, None)
    except (OSError, ProtocolError):
        return
    if msg is None or msg.get("type") != protocol.TERMINATE:
        return

    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return  # already reaped

    def note(text: str) -> None:
        try:
            logf.write(f"\n# gpuq: {text}\n")
        except (OSError, ValueError):
            pass

    note("cancelled; sending SIGTERM to the job's process group")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return
    try:
        proc.wait(timeout=TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    note(f"still running after {TERMINATE_GRACE_SECONDS}s; sending SIGKILL")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


def _run_waiter(conn, job_id, cmd, log_file, user, args, mem_bytes, msgs=None) -> None:
    # One iterator for the whole connection: it owns the read buffer, so
    # handing it to the terminate watcher later cannot lose a message that
    # arrived in the same packet as 'go'. Callers that already read from this
    # socket (the pre-fork handshake) must pass their iterator in, for the same
    # reason -- a fresh one would not see what theirs had buffered.
    if msgs is None:
        msgs = protocol.iter_msgs(conn)

    # Wait for 'go' (or a cancellation/error closing the connection).
    go = next(msgs, None)
    if go is None or go.get("type") != protocol.GO:
        # Cancelled or daemon went away before granting a slot.
        os._exit(0)
    gpu_ids: List[int] = go.get("gpu_ids", [])

    ensure_logs_dir(os.path.dirname(log_file))
    started_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    # Run the command as ourselves, stdout+stderr merged into the log file,
    # which opens with the metadata header first.
    exit_code = 1
    try:
        with open(log_file, "a", buffering=1) as logf:
            write_header(logf, {
                "job_id": job_id, "user": user, "cmd": cmd, "name": args.name,
                "gpu_ids": ",".join(str(g) for g in gpu_ids),
                "mem_requested_human": format_size(mem_bytes) if mem_bytes else None,
                "shareable": args.share, "started_at": started_at,
            })
            # start_new_session puts the command in its own process group, so a
            # cancel can signal the whole tree (shell=True means the direct
            # child is /bin/sh, not the user's real workload).
            proc = subprocess.Popen(
                cmd, shell=True, stdout=logf, stderr=subprocess.STDOUT,
                env=env, start_new_session=True,
            )
            watcher = threading.Thread(
                target=_watch_for_terminate, args=(msgs, proc, logf), daemon=True
            )
            watcher.start()
            exit_code = proc.wait()
    except Exception as exc:  # noqa: BLE001
        try:
            with open(log_file, "a") as logf:
                logf.write(f"\n# gpuq: failed to launch command: {exc}\n")
        except OSError:
            pass
        exit_code = 127

    try:
        protocol.send_msg(conn, protocol.done(job_id, exit_code))
    except OSError:
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


# --- status / mine ---------------------------------------------------------

def _fmt_time(iso: Optional[str]) -> str:
    if not iso:
        return "-"
    try:
        dt = datetime.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso


def _name_or_cmd(job: dict) -> str:
    return job.get("name") or job.get("cmd", "")


def _query_status(sock_path: str) -> list:
    conn = _connect(sock_path)
    protocol.send_msg(conn, protocol.status())
    reply = protocol.recv_one(conn)
    conn.close()
    if reply is None:
        _die("no response from daemon")
    return reply.get("jobs", [])


def _gpu_cell(job: dict) -> str:
    ids = job.get("gpu_ids")
    if ids:
        return ",".join(str(g) for g in ids)
    return "-"


def cmd_status(args, sock_path: str) -> int:
    try:
        jobs = _query_status(sock_path)
    except OSError as exc:
        _die(f"cannot reach gpuqd at {sock_path}: {exc}")
    me = _current_user()
    if args.mine:
        jobs = [j for j in jobs if j.get("user") == me]
    if not jobs:
        print("No active jobs." if not args.mine else "You have no active jobs.")
        return 0
    if args.mine:
        running = sum(1 for j in jobs if j.get("status") == "running")
        queued = sum(1 for j in jobs if j.get("status") == "queued")
        print(f"Your jobs: {running} running, {queued} queued\n")
        header = f"{'ID':<7}{'STATUS':<10}{'POS':<5}{'GPU':<6}{'SUBMITTED':<18}{'STARTED':<18}NAME/CMD"
    else:
        header = f"{'ID':<7}{'USER':<9}{'STATUS':<10}{'POS':<5}{'GPU':<6}{'SUBMITTED':<18}{'STARTED':<18}NAME/CMD"
    print(header)
    for j in jobs:
        pos = j.get("queue_position")
        pos_s = str(pos) if pos else "-"
        row = (f"{_short(j['job_id']):<7}"
               + ("" if args.mine else f"{j.get('user',''):<9}")
               + f"{j.get('status',''):<10}{pos_s:<5}{_gpu_cell(j):<6}"
               + f"{_fmt_time(j.get('submitted_at')):<18}{_fmt_time(j.get('started_at')):<18}"
               + _name_or_cmd(j))
        print(row)
    return 0


def cmd_mine(args, sock_path: str) -> int:
    args.mine = True
    return cmd_status(args, sock_path)


# --- logs ------------------------------------------------------------------

def cmd_logs(args, sock_path: str) -> int:
    try:
        jobs = _query_status(sock_path)
    except OSError:
        jobs = []
    log_file = _resolve_log_file(args.job_id, jobs, sock_path)
    if log_file is None:
        _die(f"no log file found for job {args.job_id}")
    if not os.path.exists(log_file):
        _die(f"log file does not exist yet: {log_file}")
    if args.follow:
        _tail_follow(log_file)
    else:
        with open(log_file) as f:
            sys.stdout.write(f.read())
    return 0


def _resolve_log_file(prefix: str, active_jobs: list, sock_path: str) -> Optional[str]:
    for j in active_jobs:
        if j["job_id"].startswith(prefix):
            return j.get("log_file")
    # Fall back to history for finished jobs.
    try:
        conn = _connect(sock_path)
        protocol.send_msg(conn, protocol.history(200, False, None))
        reply = protocol.recv_one(conn)
        conn.close()
        for j in (reply or {}).get("jobs", []):
            if j["job_id"].startswith(prefix):
                return j.get("log_file")
    except OSError:
        pass
    return None


def _tail_follow(path: str) -> None:
    with open(path) as f:
        sys.stdout.write(f.read())
        sys.stdout.flush()
        try:
            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass


# --- cancel ----------------------------------------------------------------

def cmd_cancel(args, sock_path: str) -> int:
    user = _current_user()
    try:
        conn = _connect(sock_path)
    except OSError as exc:
        _die(f"cannot reach gpuqd at {sock_path}: {exc}")
    if args.all:
        protocol.send_msg(conn, protocol.cancel_all(user))
        reply = protocol.recv_one(conn)
        conn.close()
        n = (reply or {}).get("cancelled", 0)
        print(f"Cancelled {n} queued job(s).")
        return 0
    if not args.job_id:
        _die("specify a job id or --all")
    job = _resolve_active_job(args.job_id, sock_path)
    if job is None:
        conn.close()
        _die(f"no active job matching {args.job_id}")

    force = bool(args.force)
    if job.get("status") == "running" and not force:
        if not _confirm_kill(job):
            conn.close()
            print("Not cancelled.")
            return 0
        force = True

    protocol.send_msg(conn, protocol.cancel(job["job_id"], user, force=force))
    reply = protocol.recv_one(conn)
    conn.close()
    if reply and reply.get("type") == protocol.ERROR:
        _die(reply.get("message", "cancel failed"))
    name = (reply or {}).get("name")
    label = f" ({name})" if name else ""
    if job.get("status") == "running":
        print(f"Killed job {args.job_id}{label}")
    else:
        print(f"Cancelled job {args.job_id}{label}")
    return 0


def _resolve_active_job(prefix: str, sock_path: str) -> Optional[dict]:
    try:
        for j in _query_status(sock_path):
            if j["job_id"].startswith(prefix):
                return j
    except OSError:
        pass
    return None


def _confirm_kill(job: dict) -> bool:
    """Prompt before killing a running job. Requires a tty; --force skips it."""
    if not sys.stdin.isatty():
        _die(f"job {_short(job['job_id'])} is running; "
             f"pass --force to kill it non-interactively")
    label = f" ({job['name']})" if job.get("name") else ""
    gpus = job.get("gpu_ids")
    if isinstance(gpus, (list, tuple)):
        gpus = ",".join(str(g) for g in gpus)
    where = f" on GPU {gpus}" if gpus else ""
    print(f"Job {_short(job['job_id'])}{label} is running{where}:")
    print(f"  {job.get('cmd', '')}")
    try:
        reply = input("Kill it? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return reply.strip().lower() in ("y", "yes")


# --- history ---------------------------------------------------------------

def cmd_history(args, sock_path: str) -> int:
    user = _current_user() if args.mine else None
    try:
        conn = _connect(sock_path)
    except OSError as exc:
        _die(f"cannot reach gpuqd at {sock_path}: {exc}")
    protocol.send_msg(conn, protocol.history(args.limit, args.mine, user))
    reply = protocol.recv_one(conn)
    conn.close()
    jobs = (reply or {}).get("jobs", [])
    if not jobs:
        print("No finished jobs.")
        return 0
    print(f"{'ID':<7}{'STATUS':<10}{'GPU':<6}{'SUBMITTED':<18}{'STARTED':<18}"
          f"{'FINISHED':<18}{'EXIT':<6}NAME/CMD")
    for j in jobs:
        exit_s = "-" if j.get("exit_code") is None else str(j["exit_code"])
        print(f"{_short(j['job_id']):<7}{j.get('status',''):<10}{_gpu_cell(j):<6}"
              f"{_fmt_time(j.get('submitted_at')):<18}{_fmt_time(j.get('started_at')):<18}"
              f"{_fmt_time(j.get('finished_at')):<18}{exit_s:<6}{_name_or_cmd(j)}")
    return 0


# --- audit -------------------------------------------------------------

def _resolve_job_id(prefix: str, sock_path: str) -> Optional[str]:
    """Resolve a (possibly short) job id prefix to a full job id, checking
    the active queue first and falling back to history for finished jobs."""
    try:
        for j in _query_status(sock_path):
            if j["job_id"].startswith(prefix):
                return j["job_id"]
    except OSError:
        pass
    try:
        conn = _connect(sock_path)
        protocol.send_msg(conn, protocol.history(200, False, None))
        reply = protocol.recv_one(conn)
        conn.close()
        for j in (reply or {}).get("jobs", []):
            if j["job_id"].startswith(prefix):
                return j["job_id"]
    except OSError:
        pass
    return None


def cmd_audit(args, sock_path: str) -> int:
    job_id = _resolve_job_id(args.job_id, sock_path)
    if job_id is None:
        _die(f"no job found matching {args.job_id}")
    try:
        conn = _connect(sock_path)
    except OSError as exc:
        _die(f"cannot reach gpuqd at {sock_path}: {exc}")
    protocol.send_msg(conn, protocol.audit(job_id))
    reply = protocol.recv_one(conn)
    conn.close()
    events = (reply or {}).get("events", [])
    if not events:
        print(f"No placement events recorded for job {_short(job_id)}.")
        return 0
    for ev in events:
        gpu_ids = ",".join(str(g) for g in ev.get("gpu_ids", []))
        print(f"Job {_short(job_id)} placed at {ev.get('placed_at')} on GPU(s) {gpu_ids}")
        print(f"  cmd: {ev.get('cmd')}")
        mem_h = ev.get("mem_requested_human")
        if mem_h:
            print(f"  requested: {mem_h}"
                  f"{' (shared)' if ev.get('shareable') else ''}")
        print(f"  {'GPU':<5}{'USED':<10}{'FREE':<10}{'TOTAL':<10}{'UTIL':<6}PROCESSES")
        for g in ev.get("gpu_snapshot", []):
            procs = g.get("processes") or []
            proc_s = ", ".join(
                f"{p.get('pid')}:{p.get('process_name')}"
                f"({format_size(p.get('used_memory', 0))})"
                for p in procs
            ) or "-"
            print(f"  {g.get('index'):<5}"
                  f"{format_size(g.get('mem_used', 0)):<10}"
                  f"{format_size(g.get('mem_free', 0)):<10}"
                  f"{format_size(g.get('mem_total', 0)):<10}"
                  f"{str(g.get('utilization_pct', 0)) + '%':<6}"
                  f"{proc_s}")
        print()
    return 0


# --- argparse --------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpuq", description="GPU job queue client")
    p.add_argument("--socket", help="path to the gpuqd socket")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="queue a command to run on a GPU")
    s.add_argument("command", help="the command to run (quote it)")
    s.add_argument("--name", help="a label for the job and its log file")
    s.add_argument("--gpus", type=int, default=1, help="number of GPUs (default 1)")
    s.add_argument("--mem", help="per-GPU memory hint/requirement, e.g. 8G, 512M")
    s.add_argument("--share", action="store_true",
                   help="allow co-locating with other --share jobs (requires --mem)")
    s.set_defaults(func=cmd_submit)

    st = sub.add_parser("status", help="show queued and running jobs")
    st.add_argument("--mine", action="store_true", help="only your jobs")
    st.set_defaults(func=cmd_status)

    mn = sub.add_parser("mine", help="show your jobs with a summary")
    mn.set_defaults(func=cmd_mine, mine=True)

    lg = sub.add_parser("logs", help="print or follow a job's log")
    lg.add_argument("job_id", help="job id (prefix ok)")
    lg.add_argument("-f", "--follow", action="store_true", help="follow the log")
    lg.set_defaults(func=cmd_logs)

    cl = sub.add_parser("cancel", help="cancel a queued or running job")
    cl.add_argument("job_id", nargs="?", help="job id (prefix ok)")
    cl.add_argument("--all", action="store_true", help="cancel all your queued jobs")
    cl.add_argument("-f", "--force", action="store_true",
                    help="kill a running job without confirming")
    cl.set_defaults(func=cmd_cancel)

    hi = sub.add_parser("history", help="show recently finished jobs")
    hi.add_argument("--limit", type=int, default=20, help="max rows (default 20)")
    hi.add_argument("--mine", action="store_true", help="only your jobs")
    hi.set_defaults(func=cmd_history)

    au = sub.add_parser("audit", help="show GPU state at the moment a job was placed")
    au.add_argument("job_id", help="job id (prefix ok)")
    au.set_defaults(func=cmd_audit)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sock_path = client_socket_path(args.socket)
    return args.func(args, sock_path)


if __name__ == "__main__":
    raise SystemExit(main())
