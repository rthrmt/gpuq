"""Configuration: defaults, ``/etc/gpuq/gpuq.conf`` parsing, env/flag overrides.

Override precedence (highest first):

    explicit CLI flag  >  environment variable  >  config file  >  built-in default

The environment seams (``GPUQ_SOCKET``, ``GPUQ_DB``, ``GPUQ_LOG``, ``GPUQ_CONF``,
``GPUQ_NVIDIA_SMI``, ``GPUQ_REATTACH_GRACE``) make it possible to run a daemon + client entirely in user
space for development and tests, without touching ``/etc`` or ``/var`` and
without root.

The client also reads a per-user file (``~/.config/gpuq/config``, see
:func:`user_conf_path`) for personal submit defaults such as the notification
email address. ``GPUQ_USER_CONF`` and ``GPUQ_SENDMAIL`` are its test seams.
"""

import configparser
import os
from dataclasses import dataclass
from typing import List, Optional

from .errors import ConfigError

DEFAULT_CONF_PATH = "/etc/gpuq/gpuq.conf"
DEFAULT_SOCKET_PATH = "/run/gpuq/gpuq.sock"
DEFAULT_DB_PATH = "/var/lib/gpuq/queue.db"
DEFAULT_LOG_PATH = "/var/log/gpuq/daemon.log"
DEFAULT_SENDMAIL_PATH = "/usr/sbin/sendmail"
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_IDLE_THRESHOLD_MB = 500
# How long a freshly started daemon waits for the waiters of the jobs it finds
# in the database to reattach before giving up on them. Waiters retry at most
# every 5 seconds (cli.py), so this leaves a wide margin.
DEFAULT_REATTACH_GRACE = 30.0
# Open to all local users by default. gpuq is a single-node, trusted-box tool;
# access is not gated behind a Unix group (see also the daemon's socket setup).
DEFAULT_SOCKET_MODE = 0o666

_MB = 1024 * 1024


@dataclass(frozen=True)
class Config:
    socket_path: str
    db_path: str
    log_path: str
    poll_interval_seconds: float
    managed_gpus: Optional[List[int]]          # None => all GPUs nvidia-smi reports
    idle_memory_threshold_bytes: int
    max_gpus_per_user: int                     # 0 => unlimited
    socket_mode: int                           # e.g. 0o666 (open to all local users)
    reattach_grace_seconds: float = DEFAULT_REATTACH_GRACE


def _conf_path(explicit: Optional[str]) -> Optional[str]:
    """Resolve which config file to read, honouring GPUQ_CONF."""
    path = explicit or os.environ.get("GPUQ_CONF") or DEFAULT_CONF_PATH
    return path if path and os.path.exists(path) else None


def _parse_gpu_list(raw: str) -> List[int]:
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError as exc:
            raise ConfigError(f"invalid GPU index in managed_gpus: {part!r}") from exc
    return out


def _parse_mode(raw: str) -> int:
    raw = raw.strip()
    try:
        # Octal by convention (e.g. "0660"); accept a leading 0o too.
        return int(raw, 8)
    except ValueError as exc:
        raise ConfigError(f"invalid socket_mode {raw!r}: use octal like 0660") from exc


def load_config(
    *,
    conf_path: Optional[str] = None,
    socket_path: Optional[str] = None,
    db_path: Optional[str] = None,
    log_path: Optional[str] = None,
) -> Config:
    """Build a :class:`Config`, applying the documented override precedence.

    The ``*_path`` keyword arguments correspond to CLI flags and win over
    everything else when provided.
    """
    parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    found = _conf_path(conf_path)
    if found:
        try:
            parser.read(found)
        except configparser.Error as exc:
            raise ConfigError(f"failed to parse {found}: {exc}") from exc

    daemon = parser["daemon"] if parser.has_section("daemon") else {}
    gpus = parser["gpus"] if parser.has_section("gpus") else {}
    fairness = parser["fairness"] if parser.has_section("fairness") else {}

    def pick(flag, env, section, key, default):
        if flag is not None:
            return flag
        if env and os.environ.get(env):
            return os.environ[env]
        if key in section:
            return section[key]
        return default

    resolved_socket = pick(socket_path, "GPUQ_SOCKET", daemon, "socket_path", DEFAULT_SOCKET_PATH)
    resolved_db = pick(db_path, "GPUQ_DB", daemon, "db_path", DEFAULT_DB_PATH)
    resolved_log = pick(log_path, "GPUQ_LOG", daemon, "log_path", DEFAULT_LOG_PATH)

    poll_raw = pick(None, None, daemon, "poll_interval_seconds", DEFAULT_POLL_INTERVAL)
    try:
        poll = float(poll_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid poll_interval_seconds {poll_raw!r}") from exc

    grace_raw = pick(None, "GPUQ_REATTACH_GRACE", daemon, "reattach_grace_seconds",
                     DEFAULT_REATTACH_GRACE)
    try:
        grace = float(grace_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid reattach_grace_seconds {grace_raw!r}") from exc
    if grace < 0:
        raise ConfigError(f"invalid reattach_grace_seconds {grace_raw!r}: must be >= 0")

    managed: Optional[List[int]] = None
    if "managed_gpus" in gpus and gpus["managed_gpus"].strip():
        managed = _parse_gpu_list(gpus["managed_gpus"])

    idle_mb_raw = gpus.get("idle_memory_threshold_mb", str(DEFAULT_IDLE_THRESHOLD_MB))
    try:
        idle_bytes = int(float(idle_mb_raw)) * _MB
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid idle_memory_threshold_mb {idle_mb_raw!r}") from exc

    max_raw = fairness.get("max_gpus_per_user", "0")
    try:
        max_per_user = int(max_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid max_gpus_per_user {max_raw!r}") from exc

    mode_raw = daemon.get("socket_mode")
    socket_mode = _parse_mode(mode_raw) if mode_raw else DEFAULT_SOCKET_MODE

    return Config(
        socket_path=resolved_socket,
        db_path=resolved_db,
        log_path=resolved_log,
        poll_interval_seconds=poll,
        managed_gpus=managed,
        idle_memory_threshold_bytes=idle_bytes,
        max_gpus_per_user=max_per_user,
        socket_mode=socket_mode,
        reattach_grace_seconds=grace,
    )


def _system_option(section: str, key: str) -> Optional[str]:
    """One value from the system config file, or None. Errors are ignored:
    the client must keep working when it cannot make sense of the file."""
    found = _conf_path(None)
    if found:
        parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        try:
            parser.read(found)
            if parser.has_option(section, key):
                return parser.get(section, key)
        except configparser.Error:
            pass
    return None


def client_socket_path(explicit: Optional[str] = None) -> str:
    """Resolve just the socket path for the CLI client (no full config needed)."""
    if explicit:
        return explicit
    if os.environ.get("GPUQ_SOCKET"):
        return os.environ["GPUQ_SOCKET"]
    return _system_option("daemon", "socket_path") or DEFAULT_SOCKET_PATH


def client_sendmail_path() -> str:
    """The sendmail-compatible binary a waiter hands notification emails to:
    ``GPUQ_SENDMAIL`` (test seam), else ``[mail] sendmail`` in the system
    config, else ``/usr/sbin/sendmail``."""
    if os.environ.get("GPUQ_SENDMAIL"):
        return os.environ["GPUQ_SENDMAIL"]
    return (_system_option("mail", "sendmail") or "").strip() or DEFAULT_SENDMAIL_PATH


def user_conf_path() -> str:
    """The per-user config file: ``GPUQ_USER_CONF``, else
    ``$XDG_CONFIG_HOME/gpuq/config`` (``~/.config/gpuq/config``)."""
    if os.environ.get("GPUQ_USER_CONF"):
        return os.environ["GPUQ_USER_CONF"]
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join("~", ".config")
    return os.path.expanduser(os.path.join(base, "gpuq", "config"))


def user_default_email(path: Optional[str] = None) -> Optional[str]:
    """``[notify] email`` from the per-user config file, or None if the file
    or the setting is absent. A file that does not parse is a ConfigError, so
    a typo is reported instead of silently disabling notifications."""
    path = path or user_conf_path()
    if not os.path.exists(path):
        return None
    parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    try:
        parser.read(path)
    except configparser.Error as exc:
        raise ConfigError(f"failed to parse {path}: {exc}") from exc
    value = parser.get("notify", "email", fallback="").strip()
    return value or None
