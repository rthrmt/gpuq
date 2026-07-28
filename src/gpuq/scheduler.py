"""Pure scheduling logic: fairness + multi-GPU best-fit + opt-in packing.

This module performs *no* I/O. It decides, given the current queue, the live
GPU state, and per-user running counts, which job (if any) to place next and on
which GPUs. The daemon owns the side effects (DB writes, sending ``go``).

Scheduling contract
--------------------
1. **Fairness selects WHICH job.** Order queued jobs by ``(running_count[user],
   submitted_at, job_id)`` — the user with the fewest running jobs goes first,
   ties broken FIFO.
2. **Placement decides IF/WHERE it fits.** :func:`try_place` returns a list of
   GPU ids (exactly ``gpus_requested`` of them) or ``None``.
3. **Skip, don't block.** If the fairest job can't be placed, try the next
   candidate rather than head-of-line blocking. (Trade-off: a large/exclusive
   job can be starved by a stream of small jobs; aging is a future extension.)
4. Multi-GPU is **atomic**: a job gets all N GPUs or none.
"""

from typing import Callable, Dict, List, Optional, Sequence

from .gpustate import GpuState


class SchedJob:
    """Minimal view of a queued job the scheduler needs. Keeps it decoupled
    from the DB ``Job`` dataclass so the algorithm is easy to test."""

    __slots__ = ("job_id", "user", "gpus_requested", "mem_requested",
                 "shareable", "submitted_at")

    def __init__(self, job_id, user, gpus_requested, mem_requested,
                 shareable, submitted_at):
        self.job_id = job_id
        self.user = user
        self.gpus_requested = gpus_requested
        self.mem_requested = mem_requested
        self.shareable = shareable
        self.submitted_at = submitted_at


def fair_order(jobs: Sequence[SchedJob], running_counts: Dict[str, int]) -> List[SchedJob]:
    """Return jobs ordered fairest-first (fewest running, then FIFO)."""
    return sorted(
        jobs,
        key=lambda j: (running_counts.get(j.user, 0), j.submitted_at, j.job_id),
    )


def try_place(job: SchedJob, state: GpuState) -> Optional[List[int]]:
    """Return the GPU ids to assign ``job``, or ``None`` if it cannot be placed.

    Uses best-fit: among eligible GPUs, pick the ``gpus_requested`` with the
    smallest fit metric (see :meth:`GpuState.fit_metric`).
    """
    eligible = [
        idx for idx in state.managed_indices
        if state.eligible(idx, shareable=job.shareable, mem_bytes=job.mem_requested)
    ]
    if len(eligible) < job.gpus_requested:
        return None
    eligible.sort(
        key=lambda idx: state.fit_metric(
            idx, shareable=job.shareable, mem_bytes=job.mem_requested
        )
    )
    return eligible[: job.gpus_requested]


def schedule_pass(
    queued: Sequence[SchedJob],
    state: GpuState,
    running_counts: Dict[str, int],
    *,
    commit: Callable[[SchedJob, List[int]], None],
    max_gpus_per_user: int = 0,
    gpus_held_by_user: Optional[Dict[str, int]] = None,
) -> int:
    """Place as many jobs as fit in one pass. Returns the number placed.

    ``commit(job, gpu_ids)`` is invoked for each placement and is responsible
    for reserving the GPUs in ``state``, persisting, and notifying the waiter.
    ``running_counts`` and the held-GPU tally are updated locally so a single
    pass stays fair across multiple placements.
    """
    counts = dict(running_counts)
    held = dict(gpus_held_by_user or {})
    pending = list(queued)
    placed = 0

    progress = True
    while progress and pending:
        progress = False
        for job in fair_order(pending, counts):
            if max_gpus_per_user:
                if held.get(job.user, 0) + job.gpus_requested > max_gpus_per_user:
                    continue
            gpu_ids = try_place(job, state)
            if gpu_ids is None:
                continue
            commit(job, gpu_ids)              # reserves in `state` as a side effect
            counts[job.user] = counts.get(job.user, 0) + 1
            held[job.user] = held.get(job.user, 0) + len(gpu_ids)
            pending.remove(job)
            placed += 1
            progress = True
            break                              # counts changed -> re-evaluate fairness
    return placed
