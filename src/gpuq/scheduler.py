"""Pure scheduling logic: fairness + multi-GPU best-fit + opt-in packing.

This module performs *no* I/O. It decides, given the current queue, the live
GPU state, and per-user running counts, which job (if any) to place next and on
which GPUs. The daemon owns the side effects (DB writes, sending ``go``).

Scheduling contract
--------------------
1. **Fairness selects WHICH job.** Order queued jobs by ``(running_count[user],
   queue slot)`` — the user with the fewest running jobs goes first, ties
   broken FIFO.
   **Priority only reorders a user's own jobs.** Each queued job contributes
   one FIFO slot ``(submitted_at, job_id)`` to its owner; the owner's jobs fill
   those slots highest-priority first (see :func:`queue_slots`). A user's place
   relative to *other* users is therefore untouched by any priority they set.
   :func:`start_order` forecasts the whole queue with the same ranking, for
   display.
2. **Placement decides IF/WHERE it fits.** :func:`try_place` returns a list of
   GPU ids (exactly ``gpus_requested`` of them) or ``None``.
3. **Skip, don't block.** If the fairest job can't be placed, try the next
   candidate rather than head-of-line blocking. (Trade-off: a large/exclusive
   job can be starved by a stream of small jobs; aging is a future extension.)
4. Multi-GPU is **atomic**: a job gets all N GPUs or none.
"""

from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .gpustate import GpuState


class SchedJob:
    """Minimal view of a queued job the scheduler needs. Keeps it decoupled
    from the DB ``Job`` dataclass so the algorithm is easy to test."""

    __slots__ = ("job_id", "user", "gpus_requested", "mem_requested",
                 "shareable", "submitted_at", "priority")

    def __init__(self, job_id, user, gpus_requested, mem_requested,
                 shareable, submitted_at, priority=0):
        self.job_id = job_id
        self.user = user
        self.gpus_requested = gpus_requested
        self.mem_requested = mem_requested
        self.shareable = shareable
        self.submitted_at = submitted_at
        self.priority = priority


def queue_slots(jobs: Sequence) -> Dict[str, Tuple[str, str]]:
    """Map each job id to the FIFO slot it competes with.

    A slot is the ``(submitted_at, job_id)`` of one of the owner's queued jobs.
    Per user, the slots are handed out earliest-first to that user's jobs
    ranked by ``(-priority, submitted_at, job_id)``. With equal priorities this
    is the identity, i.e. plain FIFO. Raising a priority swaps slots among the
    owner's own jobs and never creates an earlier slot, so it cannot move a job
    ahead of another user's.

    Works on anything with ``job_id``/``user``/``submitted_at``/``priority``
    (``SchedJob`` or the DB ``Job``).
    """
    by_user: Dict[str, list] = {}
    for j in jobs:
        by_user.setdefault(j.user, []).append(j)
    slots: Dict[str, Tuple[str, str]] = {}
    for user_jobs in by_user.values():
        fifo = sorted((j.submitted_at, j.job_id) for j in user_jobs)
        ranked = sorted(
            user_jobs, key=lambda j: (-j.priority, j.submitted_at, j.job_id))
        for j, slot in zip(ranked, fifo):
            slots[j.job_id] = slot
    return slots


def fair_order(jobs: Sequence[SchedJob], running_counts: Dict[str, int]) -> List[SchedJob]:
    """Return jobs ordered fairest-first (fewest running, then queue slot)."""
    slots = queue_slots(jobs)
    return sorted(
        jobs,
        key=lambda j: (running_counts.get(j.user, 0),) + slots[j.job_id],
    )


def start_order(jobs: Sequence, running_users: Sequence[str]) -> list:
    """Return queued jobs in the order they are expected to start.

    This is what users are shown as the queue (``POS`` in ``gpuq status``).
    A queued job starts when a running one ends, and that lowers its owner's
    running count *before* fairness picks the successor. So the forecast
    replays exactly that: the running jobs (``running_users``: their owners,
    oldest-started first) are assumed to end in the order they started, and
    each time one does, :func:`fair_order` picks the next job, which then
    joins the back of the running list.

    It is a forecast, not a promise: jobs do not really finish in start order,
    and a job may not fit the GPU that frees up.
    """
    counts: Dict[str, int] = {}
    for user in running_users:
        counts[user] = counts.get(user, 0) + 1
    finishing = deque(running_users)
    pending = list(jobs)
    order = []
    while pending:
        if finishing:
            counts[finishing.popleft()] -= 1
        job = fair_order(pending, counts)[0]
        order.append(job)
        pending.remove(job)
        counts[job.user] = counts.get(job.user, 0) + 1
        finishing.append(job.user)
    return order


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
