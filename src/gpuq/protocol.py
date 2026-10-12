"""Newline-delimited JSON (NDJSON) IPC over the Unix domain socket.

Each connection carries one job lifecycle (register -> queued -> go -> done),
the rest of one after a daemon restart (reattach -> reattached -> ...), or
one short query (status / cancel / cancel_all / prioritize / history). This module owns the
*wire* concern: framing bytes on/off the socket and building/validating the
message dicts, so the daemon and CLI never duplicate that logic.

Message types
-------------
register      waiter  -> daemon   submit a job and wait for a slot
queued        daemon  -> waiter   ack: accepted, here's your queue position
go            daemon  -> waiter   run now on these GPU ids
done          waiter  -> daemon   the job finished with this exit code
done_ack      daemon  -> waiter   the exit code was recorded
reattach      waiter  -> daemon   resume a job after the connection was lost
reattached    daemon  -> waiter   ack: the job is tracked again
error         daemon  -> client   request rejected (carries a code)
status        client  -> daemon   request the active queue
status_response daemon-> client   the active/queued jobs
history       client  -> daemon   request finished jobs
history_response daemon->client   finished jobs
cancel        client  -> daemon   cancel one queued or running job
terminate     daemon  -> waiter   kill the running command's process group
cancel_all    client  -> daemon   cancel all of a user's queued jobs
cancel_response / cancel_all_response  daemon -> client
prioritize    client  -> daemon   change one queued job's priority
prioritize_response daemon->client the new priority and queue position
audit         client  -> daemon   request placement events for one job
audit_response daemon -> client   the placement event(s) for that job
"""

import json
import socket
from typing import Any, Dict, Iterator, Optional

from .errors import ProtocolError

# --- message type constants ----------------------------------------------
REGISTER = "register"
QUEUED = "queued"
GO = "go"
DONE = "done"
ERROR = "error"
STATUS = "status"
STATUS_RESPONSE = "status_response"
HISTORY = "history"
HISTORY_RESPONSE = "history_response"
CANCEL = "cancel"
CANCEL_RESPONSE = "cancel_response"
TERMINATE = "terminate"
CANCEL_ALL = "cancel_all"
CANCEL_ALL_RESPONSE = "cancel_all_response"
DONE_ACK = "done_ack"
REATTACH = "reattach"
REATTACHED = "reattached"
PRIORITIZE = "prioritize"
PRIORITIZE_RESPONSE = "prioritize_response"
AUDIT = "audit"
AUDIT_RESPONSE = "audit_response"

# Version of the wire protocol this code speaks. A waiter is a long-lived
# process that keeps running the code it was started with, across any number of
# daemon upgrades, so the daemon must keep understanding every version still in
# the field: change messages additively and bump this when a peer needs to tell.
#   1  original protocol (no version field anywhere)
#   2  waiters survive a daemon restart: reattach / reattached / done_ack
PROTOCOL_VERSION = 2

# What a waiter tells the daemon about its job when it reattaches.
STATE_QUEUED = "queued"        # still waiting for 'go'
STATE_RUNNING = "running"      # the command is running on the GPUs it was given
STATE_FINISHED = "finished"    # the command ended; exit_code is set

# Priorities are bounded so they always fit SQLite's integer column and stay
# readable in `gpuq status`.
PRIORITY_LIMIT = 1_000_000


# --- framing --------------------------------------------------------------

def send_msg(sock: socket.socket, msg: Dict[str, Any]) -> None:
    """Serialize ``msg`` as one NDJSON line and send it on ``sock``."""
    data = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
    sock.sendall(data)


def iter_msgs(sock: socket.socket, *, bufsize: int = 65536) -> Iterator[Dict[str, Any]]:
    """Yield decoded messages from ``sock`` as newline-delimited JSON.

    Reassembles partial reads and may yield several messages buffered in a
    single ``recv``. Stops cleanly when the peer closes the connection.
    Raises :class:`ProtocolError` on a malformed JSON frame.
    """
    buf = b""
    while True:
        try:
            chunk = sock.recv(bufsize)
        except (ConnectionResetError, OSError):
            break
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ProtocolError(f"malformed message: {exc}") from exc


def recv_one(sock: socket.socket, *, bufsize: int = 65536) -> Optional[Dict[str, Any]]:
    """Read and return exactly one message, or ``None`` if the peer closed.

    Only safe when this is the *last* message you will read from ``sock``. A
    single ``recv`` can pull several messages off the wire; this abandons the
    generator holding the ones it did not return, so they are lost. To read
    more than one message from a connection, keep one :func:`iter_msgs`
    iterator and pull from it (see ``cli.py:_DaemonLink``, which reads ``go``,
    ``terminate`` and ``done_ack`` from the same iterator).
    """
    for msg in iter_msgs(sock, bufsize=bufsize):
        return msg
    return None


def get_type(msg: Dict[str, Any]) -> str:
    t = msg.get("type")
    if not isinstance(t, str):
        raise ProtocolError(f"message missing string 'type': {msg!r}")
    return t


# --- message builders ------------------------------------------------------

def register(
    *,
    job_id: str,
    user: str,
    cmd: str,
    name: Optional[str],
    gpus_requested: int,
    mem_requested: Optional[int],
    shareable: bool,
    log_file: str,
    submitted_at: str,
    priority: int = 0,
    top: bool = False,
) -> Dict[str, Any]:
    """Submit a job.

    ``priority`` orders the job among the *same user's* queued jobs (higher
    first). ``top`` asks the daemon to pick a priority above all of that user's
    queued jobs instead, and overrides ``priority``.
    """
    return {
        "type": REGISTER,
        "version": PROTOCOL_VERSION,
        "job_id": job_id,
        "user": user,
        "cmd": cmd,
        "name": name,
        "gpus_requested": gpus_requested,
        "mem_requested": mem_requested,
        "shareable": shareable,
        "log_file": log_file,
        "submitted_at": submitted_at,
        "priority": priority,
        "top": bool(top),
    }


def queued(queue_position: int, priority: int = 0) -> Dict[str, Any]:
    return {"type": QUEUED, "queue_position": queue_position, "priority": priority}


def go(gpu_ids) -> Dict[str, Any]:
    return {"type": GO, "gpu_ids": list(gpu_ids)}


def done(job_id: str, exit_code: int) -> Dict[str, Any]:
    return {"type": DONE, "job_id": job_id, "exit_code": exit_code}


def done_ack() -> Dict[str, Any]:
    return {"type": DONE_ACK}


def reattach(job_id: str, user: str, state: str,
             exit_code: Optional[int] = None) -> Dict[str, Any]:
    """Resume tracking a job whose connection was lost (the daemon restarted).

    ``state`` is one of the ``STATE_*`` constants and describes the job as the
    waiter sees it; ``exit_code`` accompanies ``STATE_FINISHED``.
    """
    return {
        "type": REATTACH,
        "version": PROTOCOL_VERSION,
        "job_id": job_id,
        "user": user,
        "state": state,
        "exit_code": exit_code,
    }


def reattached() -> Dict[str, Any]:
    return {"type": REATTACHED}


def error(code: str, message: str) -> Dict[str, Any]:
    return {"type": ERROR, "code": code, "message": message}


def status() -> Dict[str, Any]:
    return {"type": STATUS}


def status_response(jobs) -> Dict[str, Any]:
    # protocol_version lets a client tell what the *running* daemon supports
    # (see `gpuqd --restart-check`); daemons before version 2 omit it.
    return {"type": STATUS_RESPONSE, "jobs": list(jobs),
            "protocol_version": PROTOCOL_VERSION}


def history(limit: int, mine: bool, user: Optional[str]) -> Dict[str, Any]:
    return {"type": HISTORY, "limit": limit, "mine": mine, "user": user}


def history_response(jobs) -> Dict[str, Any]:
    return {"type": HISTORY_RESPONSE, "jobs": list(jobs)}


def cancel(job_id: str, requesting_user: str, force: bool = False) -> Dict[str, Any]:
    """Cancel one job.

    ``force`` must be true to cancel a *running* job; the daemon rejects an
    unforced running-job cancel so the confirmation cannot be bypassed by a
    client that skips the prompt.
    """
    return {
        "type": CANCEL,
        "job_id": job_id,
        "requesting_user": requesting_user,
        "force": bool(force),
    }


def terminate(job_id: str) -> Dict[str, Any]:
    return {"type": TERMINATE, "job_id": job_id}


def cancel_response(cancelled: int, name: Optional[str] = None) -> Dict[str, Any]:
    return {"type": CANCEL_RESPONSE, "cancelled": cancelled, "name": name}


def cancel_all(requesting_user: str) -> Dict[str, Any]:
    return {"type": CANCEL_ALL, "requesting_user": requesting_user}


def cancel_all_response(cancelled: int) -> Dict[str, Any]:
    return {"type": CANCEL_ALL_RESPONSE, "cancelled": cancelled}


def prioritize(job_id: str, requesting_user: str, priority: Optional[int] = None,
               top: bool = False) -> Dict[str, Any]:
    """Change one queued job's priority: to ``priority``, or (``top``) to just
    above every other queued job of the same owner."""
    return {
        "type": PRIORITIZE,
        "job_id": job_id,
        "requesting_user": requesting_user,
        "priority": priority,
        "top": bool(top),
    }


def prioritize_response(priority: int, previous: int, queue_position: int,
                        name: Optional[str] = None) -> Dict[str, Any]:
    return {
        "type": PRIORITIZE_RESPONSE,
        "priority": priority,
        "previous": previous,
        "queue_position": queue_position,
        "name": name,
    }


def audit(job_id: str) -> Dict[str, Any]:
    return {"type": AUDIT, "job_id": job_id}


def audit_response(events) -> Dict[str, Any]:
    return {"type": AUDIT_RESPONSE, "events": list(events)}
