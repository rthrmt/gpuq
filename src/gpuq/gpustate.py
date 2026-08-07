"""In-memory GPU reservation model and the per-GPU fit/eligibility rules.

This bridges gpuq's *intent* (which jobs we've committed to which GPUs) with
*reality* (live ``nvidia-smi`` numbers). Reservations are the source of truth
for placement decisions — a freshly-placed job may not have allocated its VRAM
yet, so we must not infer "free" purely from live usage.

Placement model:

* An **exclusive** job (no ``--share``) owns each of its GPUs alone.
* A **shareable** job (``--share``, which requires ``--mem``) may co-locate on a
  GPU *only* with other shareable jobs, and only if it fits by BOTH:
  - the sum of gpuq ``--mem`` reservations leaves room, AND
  - live ``nvidia-smi`` free memory is sufficient (guards against OOM from
    external processes / not-yet-accounted usage).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .gpu import GpuInfo

logger = logging.getLogger("gpuq.gpustate")

# GPU occupancy modes.
FREE = "free"
EXCLUSIVE = "exclusive"
SHARED = "shared"


@dataclass
class Reservation:
    job_id: str
    mem_bytes: int          # bytes reserved on THIS gpu (0 = unknown -> whole card)
    shareable: bool


@dataclass
class _GpuSlot:
    index: int
    reservations: List[Reservation] = field(default_factory=list)

    @property
    def is_free(self) -> bool:
        return not self.reservations

    @property
    def mode(self) -> str:
        if not self.reservations:
            return FREE
        if any(not r.shareable for r in self.reservations):
            return EXCLUSIVE
        return SHARED

    @property
    def reserved_mem(self) -> int:
        return sum(r.mem_bytes for r in self.reservations)


class GpuState:
    """Tracks reservations per managed GPU and answers placement queries.

    ``live`` is refreshed from :func:`gpuq.gpu.query_gpus` each scheduler pass
    via :meth:`reconcile`.
    """

    def __init__(self, managed_indices: List[int], idle_threshold_bytes: int):
        self._slots: Dict[int, _GpuSlot] = {i: _GpuSlot(i) for i in managed_indices}
        self._live: Dict[int, GpuInfo] = {}
        self.idle_threshold_bytes = idle_threshold_bytes

    # --- live state -------------------------------------------------------

    def reconcile(self, live: List[GpuInfo]) -> None:
        """Refresh the live nvidia-smi snapshot. Does not evict reservations."""
        self._live = {g.index: g for g in live if g.index in self._slots}
        for idx, slot in self._slots.items():
            if idx not in self._live:
                continue
            g = self._live[idx]
            if slot.mode == EXCLUSIVE:
                # Best-effort warning if an exclusively-held card shows large
                # memory we can't attribute to our (single) reservation.
                expected = max((r.mem_bytes for r in slot.reservations), default=0)
                if expected == 0 and g.mem_used > self.idle_threshold_bytes:
                    pass  # exclusive job legitimately uses the card; no warning

    def snapshot(self) -> List[GpuInfo]:
        """Live nvidia-smi state as of the most recent reconcile(), across
        all managed GPUs. Used to record what the scheduler saw at the
        moment of a placement decision."""
        return list(self._live.values())

    @property
    def managed_indices(self) -> List[int]:
        return sorted(self._slots)

    def total_memory(self, index: int) -> int:
        g = self._live.get(index)
        return g.mem_total if g else 0

    def largest_gpu_memory(self) -> int:
        return max((g.mem_total for g in self._live.values()), default=0)

    def num_managed(self) -> int:
        return len(self._slots)

    # --- reservations -----------------------------------------------------

    def reserve(self, index: int, job_id: str, mem_bytes: int, shareable: bool) -> None:
        self._slots[index].reservations.append(
            Reservation(job_id=job_id, mem_bytes=mem_bytes, shareable=shareable)
        )

    def release_job(self, job_id: str) -> List[int]:
        """Drop all reservations for ``job_id``. Returns freed GPU indices."""
        freed = []
        for idx, slot in self._slots.items():
            before = len(slot.reservations)
            slot.reservations = [r for r in slot.reservations if r.job_id != job_id]
            if len(slot.reservations) != before:
                freed.append(idx)
        return freed

    def clear(self) -> None:
        for slot in self._slots.values():
            slot.reservations = []

    def gpus_held_by_user_jobs(self, job_ids_by_user: Dict[str, set]) -> Dict[str, int]:
        """Count GPUs currently reserved per user, given a user->job_ids map."""
        counts: Dict[str, int] = {}
        for idx, slot in self._slots.items():
            for r in slot.reservations:
                for user, ids in job_ids_by_user.items():
                    if r.job_id in ids:
                        counts[user] = counts.get(user, 0) + 1
        return counts

    # --- eligibility & fit ------------------------------------------------

    def eligible(self, index: int, *, shareable: bool, mem_bytes: Optional[int]) -> bool:
        """Whether a candidate job may use GPU ``index``.

        ``mem_bytes`` is the per-GPU request (None allowed only for exclusive
        jobs). See module docstring for the rules.
        """
        slot = self._slots.get(index)
        if slot is None:
            return False
        live = self._live.get(index)
        if live is None:
            return False  # no live data -> don't risk placing here

        if not shareable:
            # Exclusive: the card must be wholly free of gpuq reservations and
            # not occupied by external processes above the idle threshold.
            if not slot.is_free:
                return False
            if live.mem_used > self.idle_threshold_bytes:
                return False
            if mem_bytes is not None and mem_bytes > live.mem_total:
                return False
            return True

        # Shareable requires a memory request.
        if mem_bytes is None:
            return False
        if slot.mode == EXCLUSIVE:
            return False
        # Dual fit check: reservation budget AND live free memory.
        if slot.reserved_mem + mem_bytes > live.mem_total:
            return False
        if mem_bytes > live.mem_free:
            return False
        return True

    def fit_metric(self, index: int, *, shareable: bool, mem_bytes: Optional[int]):
        """Best-fit sort key for an *eligible* GPU (smaller = preferred).

        Returns ``(leftover, total_memory, index)``. For a shareable job we
        prefer the GPU left with the least spare capacity (tight pack). For an
        exclusive job we prefer the smallest card that fits, keeping big cards
        free for big jobs.
        """
        slot = self._slots[index]
        live = self._live[index]
        if shareable:
            leftover = live.mem_total - slot.reserved_mem - (mem_bytes or 0)
        else:
            # Exclusive owns the whole card; "leftover" is the card's total so
            # smaller cards sort first.
            leftover = live.mem_total
        return (leftover, live.mem_total, index)
