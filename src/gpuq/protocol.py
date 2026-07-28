"""Newline-delimited JSON (NDJSON) IPC over the Unix domain socket.

Each connection carries one job lifecycle (register -> queued -> go -> done) or
one short query (status / cancel / cancel_all / history). This module owns the
*wire* concern: framing bytes on/off the socket and building/validating the
message dicts, so the daemon and CLI never duplicate that logic.

Message types
-------------
register      waiter  -> daemon   submit a job and wait for a slot
queued        daemon  -> waiter   ack: accepted, here's your queue position
go            daemon  -> waiter   run now on these GPU ids
done          waiter  -> daemon   the job finished with this exit code
error         daemon  -> client   request rejected (carries a code)
status        client  -> daemon   request the active queue
status_response daemon-> client   the active/queued jobs
history       client  -> daemon   request finished jobs
history_response daemon->client   finished jobs
cancel        client  -> daemon   cancel one queued or running job
terminate     daemon  -> waiter   kill the running command's process group
cancel_all    client  -> daemon   cancel all of a user's queued jobs
cancel_response / cancel_all_response  daemon -> client
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
    iterator and pull from it (see ``cli.py:_run_waiter``, which reads ``go``
    and then ``terminate`` from the same iterator).
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
) -> Dict[str, Any]:
    return {
        "type": REGISTER,
        "job_id": job_id,
        "user": user,
        "cmd": cmd,
        "name": name,
        "gpus_requested": gpus_requested,
        "mem_requested": mem_requested,
        "shareable": shareable,
        "log_file": log_file,
        "submitted_at": submitted_at,
    }


def queued(queue_position: int) -> Dict[str, Any]:
    return {"type": QUEUED, "queue_position": queue_position}


def go(gpu_ids) -> Dict[str, Any]:
    return {"type": GO, "gpu_ids": list(gpu_ids)}


def done(job_id: str, exit_code: int) -> Dict[str, Any]:
    return {"type": DONE, "job_id": job_id, "exit_code": exit_code}


def error(code: str, message: str) -> Dict[str, Any]:
    return {"type": ERROR, "code": code, "message": message}


def status() -> Dict[str, Any]:
    return {"type": STATUS}


def status_response(jobs) -> Dict[str, Any]:
    return {"type": STATUS_RESPONSE, "jobs": list(jobs)}


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
