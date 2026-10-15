"""Job-completion emails, sent by the waiter once its command has ended.

The waiter sends the mail itself: it already runs as the submitting user and
is the one that knows the exit code, so the daemon stays out of it and no mail
goes out with root's identity. Delivery is handed to the local
sendmail-compatible binary (postfix, exim, msmtp-mta, nullmailer, ...), which
keeps the runtime stdlib-only and leaves relays and credentials to the admin.

Everything except :func:`tail_log` and :func:`send` is pure.
"""

import datetime
import re
import signal
import subprocess
from email.message import EmailMessage
from typing import List, Optional, Sequence

from .logfile import HEADER_END, HEADER_START

SENDMAIL_TIMEOUT_SECONDS = 60
LOG_TAIL_LINES = 20
_LOG_TAIL_BYTES = 16 * 1024

# Deliberately loose -- the MTA is the real judge -- but strict enough to keep
# header-breaking characters (CR/LF, commas, angle brackets) out of To:.
_ADDRESS_RE = re.compile(r"^[^@\s,;<>\"()]+@[^@\s,;<>\"()]+\.[^@\s,;<>\"()]+$")


def parse_addresses(raw: str) -> List[str]:
    """Split a comma-separated address list and validate each entry.

    Raises ValueError naming the first bad address.
    """
    addrs = [a.strip() for a in raw.split(",") if a.strip()]
    if not addrs:
        raise ValueError("empty email address")
    for a in addrs:
        if not _ADDRESS_RE.match(a):
            raise ValueError(f"invalid email address {a!r}")
    return addrs


def job_outcome(exit_code: int, cancelled: bool) -> str:
    """The status the daemon records for this ending: done, failed or cancelled."""
    if cancelled:
        return "cancelled"
    return "done" if exit_code == 0 else "failed"


def describe_exit(exit_code: int) -> str:
    if exit_code < 0:
        try:
            return f"killed by {signal.Signals(-exit_code).name}"
        except ValueError:
            return f"killed by signal {-exit_code}"
    return f"exit code {exit_code}"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_message(
    *,
    to: Sequence[str],
    job_id: str,
    name: Optional[str],
    cmd: str,
    host: str,
    gpu_ids: Sequence[int],
    exit_code: int,
    cancelled: bool,
    started_at: datetime.datetime,
    finished_at: datetime.datetime,
    log_file: str,
    log_tail: Optional[List[str]],
) -> EmailMessage:
    """The notification for a job that has ended. Times are aware datetimes."""
    outcome = job_outcome(exit_code, cancelled)
    label = f"job {job_id[:4]} ({_one_line(name or cmd, 60)})"
    if outcome == "done":
        subject = f"[gpuq] {label} finished on {host}"
    elif outcome == "cancelled":
        subject = f"[gpuq] {label} cancelled on {host}"
    else:
        subject = f"[gpuq] {label} failed ({describe_exit(exit_code)}) on {host}"

    def local(dt: datetime.datetime) -> str:
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    rows = [
        ("Job", job_id),
        ("Name", name or "-"),
        ("Command", cmd),
        ("Host", host),
        ("GPUs", ",".join(str(g) for g in gpu_ids) or "-"),
        ("Status", f"{outcome} ({describe_exit(exit_code)})"),
        ("Started", local(started_at)),
        ("Finished", local(finished_at)),
        ("Runtime", format_duration((finished_at - started_at).total_seconds())),
        ("Log", log_file),
    ]
    body = "\n".join(f"{k + ':':<10}{v}" for k, v in rows) + "\n"
    if log_tail:
        body += f"\nLast {len(log_tail)} lines of the log:\n\n"
        body += "\n".join("    " + line for line in log_tail) + "\n"

    msg = EmailMessage()
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject
    # RFC 3834: keep vacation responders from answering a robot.
    msg["Auto-Submitted"] = "auto-generated"
    msg["X-Gpuq-Job-Id"] = job_id
    msg.set_content(body)
    return msg


def tail_log(path: str, lines: int = LOG_TAIL_LINES) -> Optional[List[str]]:
    """The last ``lines`` lines of the job's output as a terminal would show
    them (a ``\\r``-redrawn progress bar collapses to its final state), or None
    if the log cannot be read. gpuq's own header is left out: the mail already
    says all of it."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - _LOG_TAIL_BYTES))
            data = f.read()
    except OSError:
        return None
    text = data.decode("utf-8", errors="replace")
    out = [line.rstrip("\r").rsplit("\r", 1)[-1] for line in text.split("\n")]
    if size > _LOG_TAIL_BYTES:
        out = out[1:]  # the first line was cut off by the seek
    elif out and out[0] == HEADER_START and HEADER_END in out:
        out = out[out.index(HEADER_END) + 1:]
    while out and not out[-1].strip():
        out.pop()
    return out[-lines:]


def send(msg: EmailMessage, sendmail: str) -> None:
    """Hand ``msg`` to the sendmail binary; raises OSError if that fails.

    ``-t`` takes the recipients from the headers; From is left to the MTA,
    which fills in the submitting user.
    """
    try:
        proc = subprocess.run(
            [sendmail, "-t", "-oi"], input=msg.as_bytes(),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=SENDMAIL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise OSError(f"{sendmail} timed out after {SENDMAIL_TIMEOUT_SECONDS}s") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise OSError(f"{sendmail} exited with {proc.returncode}" + (f": {err}" if err else ""))
