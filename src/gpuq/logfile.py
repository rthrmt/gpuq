"""Pure helpers for per-job log file naming and header writing.

Log files live in ~/.gpuq/logs/ by default.
"""

import os
import re
from datetime import datetime
from typing import Optional

_DEFAULT_LOGS_DIR = os.path.join("~", ".gpuq", "logs")


def sanitize_cmd(cmd: str) -> str:
    """Sanitize a command string for use as a filename component.

    Rules (applied in order):
    1. Lowercase the entire string.
    2. Replace every run of non-alphanumeric characters with a single '_'.
    3. Strip leading and trailing underscores.
    4. Truncate to 60 characters.
    5. Strip a trailing underscore that may have been left by truncation.

    The '.' character is treated as non-alphanumeric (replaced with '_').

    Examples:
        "python train.py --lr 0.001" -> "python_train_py_lr_0_001"
        "baseline_run"               -> "baseline_run"
    """
    s = cmd.lower()
    # Replace every run of non-alphanumeric characters with a single '_'
    s = re.sub(r"[^a-z0-9]+", "_", s)
    # Strip leading/trailing underscores
    s = s.strip("_")
    # Truncate to 60 chars
    s = s[:60]
    # Strip trailing underscore introduced by truncation
    s = s.rstrip("_")
    return s


def timestamp(when: Optional[datetime] = None) -> str:
    """Return a local-time timestamp string in YYYYMMDD_HHMMSS format.

    Args:
        when: Optional datetime for testability; defaults to datetime.now().
    """
    if when is None:
        when = datetime.now()
    return when.strftime("%Y%m%d_%H%M%S")


def compute_log_path(
    name: Optional[str],
    cmd: str,
    when: Optional[datetime] = None,
    logs_dir: Optional[str] = None,
) -> str:
    """Compute the full path for a job log file (pure, no I/O).

    Args:
        name:     Optional job name (from --name). If non-empty, used as the
                  base instead of cmd.
        cmd:      The command string; used as base when name is absent.
        when:     Optional datetime for the timestamp (default: now).
        logs_dir: Override for the log directory; defaults to
                  ~/.gpuq/logs (expanded at call time).

    Returns:
        Absolute path string: <logs_dir>/<base>_<YYYYMMDD_HHMMSS>.log
        Does NOT create any directories.
    """
    if logs_dir is None:
        logs_dir = os.path.expanduser(_DEFAULT_LOGS_DIR)

    if name and name.strip():
        base = sanitize_cmd(name)
    else:
        base = sanitize_cmd(cmd)

    filename = f"{base}_{timestamp(when)}.log"
    return os.path.join(logs_dir, filename)


def ensure_logs_dir(logs_dir: Optional[str] = None) -> str:
    """Create the logs directory if it does not exist and return its path.

    This is the only function in this module that touches the filesystem.

    Args:
        logs_dir: Override path; defaults to ~/.gpuq/logs.

    Returns:
        The (possibly just-created) directory path.
    """
    if logs_dir is None:
        logs_dir = os.path.expanduser(_DEFAULT_LOGS_DIR)
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def write_header(fileobj, meta: dict) -> None:
    """Write a commented metadata header to an open text file object.

    Format:
        # gpuq job log
        # gpuq <key>: <value>
        ...
        # ------------------------------------------------------------

    Keys whose value is None are skipped.
    The file is flushed after writing.

    Args:
        fileobj: An open, writable text file (or any object with write/flush).
        meta:    Dict with optional keys: job_id, user, cmd, name, gpu_ids,
                 mem_requested_human, shareable, submitted_at, started_at.
    """
    fileobj.write("# gpuq job log\n")
    for key, value in meta.items():
        if value is None:
            continue
        fileobj.write(f"# gpuq {key}: {value}\n")
    fileobj.write("# " + "-" * 60 + "\n")
    fileobj.flush()
