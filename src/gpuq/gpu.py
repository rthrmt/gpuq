"""GPU detection via nvidia-smi.

Provides :func:`query_gpus` which shells out to ``nvidia-smi`` and returns a
list of :class:`GpuInfo` dataclasses with memory figures in **bytes**.

The binary path is overridable for testing via the ``GPUQ_NVIDIA_SMI``
environment variable (default ``"nvidia-smi"``).
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Optional

logger = logging.getLogger("gpuq.gpu")

_MIB = 1024 * 1024


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NvidiaSmiError(Exception):
    """Raised when nvidia-smi cannot be run or exits with a non-zero status."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    process_name: str
    used_memory: int  # bytes


@dataclass(frozen=True)
class GpuInfo:
    index: int      # physical GPU index
    mem_total: int  # bytes
    mem_used: int   # bytes
    mem_free: int   # bytes
    utilization_pct: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def nvidia_smi_path() -> str:
    """Return the path to the nvidia-smi binary (overridable via env var)."""
    return os.environ.get("GPUQ_NVIDIA_SMI", "nvidia-smi")


def _parse_field(raw: str, field_name: str, gpu_index_hint: str) -> int:
    """Parse a single numeric field from nvidia-smi CSV output.

    Returns 0 and logs a warning for ``[N/A]`` / ``N/A`` values.
    """
    value = raw.strip()
    if value in ("[N/A]", "N/A"):
        logger.warning(
            "nvidia-smi reported N/A for field %s on GPU %s; treating as 0",
            field_name,
            gpu_index_hint,
        )
        return 0
    return int(value)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_csv(
    text: str,
    managed: Optional[Iterable[int]] = None,
) -> List[GpuInfo]:
    """Parse the CSV output of ``nvidia-smi --format=csv,noheader,nounits``.

    Each non-empty line must be in the form::

        index, used_MiB, total_MiB, free_MiB, utilization_pct

    Memory values reported by nvidia-smi are in **MiB**; they are converted to
    bytes (MiB * 1 048 576) before being stored in the returned
    :class:`GpuInfo` objects. ``utilization_pct`` is nvidia-smi's
    ``utilization.gpu`` percentage, already unitless.

    Parameters
    ----------
    text:
        Raw stdout captured from nvidia-smi.
    managed:
        Optional iterable of GPU indices to keep.  When provided, only GPUs
        whose index is in this set are included in the result.  The returned
        list is always sorted ascending by index.
    """
    managed_set: Optional[set] = None if managed is None else set(managed)

    gpus: List[GpuInfo] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            logger.warning("Skipping malformed nvidia-smi line: %r", line)
            continue

        raw_index, raw_used, raw_total, raw_free, raw_util = parts

        # Skip the whole line if the index is N/A
        if raw_index in ("[N/A]", "N/A"):
            logger.warning("Skipping nvidia-smi line with N/A index: %r", line)
            continue

        index = int(raw_index)

        if managed_set is not None and index not in managed_set:
            continue

        mem_used = _parse_field(raw_used, "memory.used", str(index)) * _MIB
        mem_total = _parse_field(raw_total, "memory.total", str(index)) * _MIB
        mem_free = _parse_field(raw_free, "memory.free", str(index)) * _MIB
        utilization_pct = _parse_field(raw_util, "utilization.gpu", str(index))

        gpus.append(GpuInfo(index=index, mem_total=mem_total, mem_used=mem_used,
                             mem_free=mem_free, utilization_pct=utilization_pct))

    gpus.sort(key=lambda g: g.index)
    return gpus


def parse_compute_apps_csv(text: str) -> List[GpuProcess]:
    """Parse the CSV output of
    ``nvidia-smi --query-compute-apps=pid,process_name,used_memory
    --format=csv,noheader,nounits`` for a single GPU (scoped via ``--id=``).

    Each non-empty line is ``pid, process_name, used_memory_MiB``. An empty
    ``text`` (no compute processes on the GPU) yields an empty list — that is
    the common case, not an error.
    """
    processes: List[GpuProcess] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            logger.warning("Skipping malformed nvidia-smi compute-apps line: %r", line)
            continue

        raw_pid, process_name, raw_used = parts
        if raw_pid in ("[N/A]", "N/A"):
            logger.warning("Skipping compute-apps line with N/A pid: %r", line)
            continue

        pid = int(raw_pid)
        used_memory = _parse_field(raw_used, "used_memory", str(pid)) * _MIB
        processes.append(GpuProcess(pid=pid, process_name=process_name, used_memory=used_memory))

    return processes


def query_gpus(
    managed: Optional[Iterable[int]] = None,
    *,
    timeout: float = 10.0,
) -> List[GpuInfo]:
    """Query per-GPU memory by running nvidia-smi as a subprocess.

    Parameters
    ----------
    managed:
        Passed through to :func:`parse_csv`; keeps only the listed indices.
    timeout:
        Seconds to wait for nvidia-smi before raising :exc:`NvidiaSmiError`.

    Raises
    ------
    NvidiaSmiError
        When the binary is not found, exits non-zero, or times out.
    """
    binary = nvidia_smi_path()
    cmd = [
        binary,
        "--query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        msg = f"nvidia-smi binary not found at {binary!r}"
        logger.error(msg)
        raise NvidiaSmiError(msg) from None
    except subprocess.TimeoutExpired:
        msg = f"nvidia-smi timed out after {timeout}s"
        logger.error(msg)
        raise NvidiaSmiError(msg) from None

    if result.returncode != 0:
        msg = (
            f"nvidia-smi exited with code {result.returncode}: "
            f"{result.stderr.strip()!r}"
        )
        logger.error(msg)
        raise NvidiaSmiError(msg)

    return parse_csv(result.stdout, managed)


def query_gpu_processes(index: int, *, timeout: float = 10.0) -> List[GpuProcess]:
    """Query the compute processes running on a single GPU.

    Scoped to one GPU via ``--id=<index>``, since
    ``--query-compute-apps`` has no per-row GPU index column of its own.
    Returns an empty list (not an error) when nvidia-smi reports no compute
    processes on the GPU.

    Raises
    ------
    NvidiaSmiError
        When the binary is not found, exits non-zero, or times out.
    """
    binary = nvidia_smi_path()
    cmd = [
        binary,
        f"--id={index}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        msg = f"nvidia-smi binary not found at {binary!r}"
        logger.error(msg)
        raise NvidiaSmiError(msg) from None
    except subprocess.TimeoutExpired:
        msg = f"nvidia-smi timed out after {timeout}s"
        logger.error(msg)
        raise NvidiaSmiError(msg) from None

    if result.returncode != 0:
        msg = (
            f"nvidia-smi exited with code {result.returncode}: "
            f"{result.stderr.strip()!r}"
        )
        logger.error(msg)
        raise NvidiaSmiError(msg)

    return parse_compute_apps_csv(result.stdout)
