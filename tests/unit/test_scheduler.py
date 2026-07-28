"""Unit tests for the pure scheduling logic and the GpuState fit rules."""

import pytest

from gpuq.gpu import GpuInfo
from gpuq.gpustate import GpuState
from gpuq.scheduler import SchedJob, fair_order, schedule_pass, try_place

GB = 1024 ** 3


def make_state(gpus, idle_threshold=500 * 1024 * 1024):
    """gpus: list of (index, total_gb, used_gb). free = total-used."""
    state = GpuState([g[0] for g in gpus], idle_threshold)
    live = [
        GpuInfo(index=i, mem_total=t * GB, mem_used=u * GB, mem_free=(t - u) * GB)
        for (i, t, u) in gpus
    ]
    state.reconcile(live)
    return state


def job(jid, user="alice", gpus=1, mem=None, share=False, ts="2026-01-01T00:00:00Z"):
    return SchedJob(jid, user, gpus, mem * GB if mem else None, share, ts)


# --- fair_order ------------------------------------------------------------

def test_fair_order_fewest_running_first():
    jobs = [job("a", user="alice"), job("b", user="bob")]
    order = fair_order(jobs, {"alice": 2, "bob": 0})
    assert [j.job_id for j in order] == ["b", "a"]


def test_fair_order_fifo_tiebreak():
    jobs = [
        job("late", user="alice", ts="2026-01-01T00:00:05Z"),
        job("early", user="bob", ts="2026-01-01T00:00:01Z"),
    ]
    order = fair_order(jobs, {})
    assert [j.job_id for j in order] == ["early", "late"]


# --- try_place: exclusive --------------------------------------------------

def test_exclusive_needs_free_card():
    state = make_state([(0, 24, 0)])
    assert try_place(job("a"), state) == [0]


def test_exclusive_blocked_by_external_usage():
    # Card has 2 GB used by a non-gpuq process; above idle threshold -> ineligible.
    state = make_state([(0, 24, 2)])
    assert try_place(job("a"), state) is None


def test_exclusive_best_fit_prefers_smaller_card():
    # Two free cards; exclusive job should take the smaller one (16 GB).
    state = make_state([(0, 80, 0), (1, 16, 0)])
    assert try_place(job("a"), state) == [1]


# --- try_place: multi-GPU --------------------------------------------------

def test_multigpu_atomic_all_or_none():
    state = make_state([(0, 24, 0), (1, 24, 0)])
    assert sorted(try_place(job("a", gpus=2), state)) == [0, 1]


def test_multigpu_not_enough_returns_none():
    state = make_state([(0, 24, 0)])
    assert try_place(job("a", gpus=2), state) is None


# --- try_place: memory fit -------------------------------------------------

def test_mem_too_large_for_any_card():
    state = make_state([(0, 16, 0)])
    assert try_place(job("a", mem=32), state) is None


def test_mem_best_fit_smallest_sufficient_card():
    # 4 GB job: cards 8/24/80 free -> best fit is the 8 GB card (exclusive uses total).
    state = make_state([(0, 80, 0), (1, 24, 0), (2, 8, 0)])
    assert try_place(job("a", mem=4), state) == [2]


# --- packing (--share) -----------------------------------------------------

def test_share_requires_mem():
    state = make_state([(0, 24, 0)])
    assert try_place(job("a", share=True, mem=None), state) is None


def test_two_shareable_jobs_colocate():
    state = make_state([(0, 24, 0)])
    placed = []

    def commit(j, ids):
        placed.append((j.job_id, ids))
        for i in ids:
            state.reserve(i, j.job_id, j.mem_requested or 0, j.shareable)

    jobs = [job("a", share=True, mem=8), job("b", share=True, mem=8)]
    n = schedule_pass(jobs, state, {}, commit=commit)
    assert n == 2
    assert placed[0][1] == [0] and placed[1][1] == [0]


def test_packing_third_job_held_when_full():
    state = make_state([(0, 24, 0)])

    def commit(j, ids):
        for i in ids:
            state.reserve(i, j.job_id, j.mem_requested or 0, j.shareable)

    jobs = [job("a", share=True, mem=10), job("b", share=True, mem=10),
            job("c", share=True, mem=10)]
    n = schedule_pass(jobs, state, {}, commit=commit)
    assert n == 2  # third (30 GB total > 24) cannot fit


def test_exclusive_cannot_join_shared_gpu():
    state = make_state([(0, 24, 0)])
    state.reserve(0, "shared-job", 8 * GB, True)
    state.reconcile([GpuInfo(0, 24 * GB, 8 * GB, 16 * GB)])
    assert try_place(job("excl"), state) is None


def test_shareable_cannot_join_exclusive_gpu():
    state = make_state([(0, 24, 0)])
    state.reserve(0, "excl-job", 0, False)
    state.reconcile([GpuInfo(0, 24 * GB, 0, 24 * GB)])
    assert try_place(job("s", share=True, mem=4), state) is None


# --- dual VRAM check -------------------------------------------------------

def test_dual_check_live_free_too_low():
    # Reservation budget fits (0 reserved + 8 <= 24) but live free is only 4 GB.
    state = make_state([(0, 24, 20)])  # 4 GB free live
    assert try_place(job("s", share=True, mem=8), state) is None


def test_dual_check_reservation_budget_exceeded():
    state = make_state([(0, 24, 0)])
    state.reserve(0, "x", 20 * GB, True)
    # Live shows full free (job hasn't allocated yet) but reservation budget is tight.
    state.reconcile([GpuInfo(0, 24 * GB, 0, 24 * GB)])
    assert try_place(job("s", share=True, mem=8), state) is None  # 20+8 > 24


# --- fairness across a pass + max_gpus_per_user ----------------------------

def test_schedule_pass_fairness_balances_users():
    state = make_state([(0, 24, 0), (1, 24, 0)])
    placed = []

    def commit(j, ids):
        placed.append(j.user)
        for i in ids:
            state.reserve(i, j.job_id, 0, False)

    # Alice has 3 queued, Bob 1; with 2 GPUs, Bob (fewer running) should get one.
    jobs = [job("a1", "alice", ts="t1"), job("a2", "alice", ts="t2"),
            job("a3", "alice", ts="t3"), job("b1", "bob", ts="t4")]
    schedule_pass(jobs, state, {}, commit=commit)
    assert "bob" in placed
    assert len(placed) == 2


def test_max_gpus_per_user_enforced():
    state = make_state([(0, 24, 0), (1, 24, 0)])

    def commit(j, ids):
        for i in ids:
            state.reserve(i, j.job_id, 0, False)

    jobs = [job("a1", "alice", ts="t1"), job("a2", "alice", ts="t2")]
    n = schedule_pass(jobs, state, {}, commit=commit, max_gpus_per_user=1)
    assert n == 1  # alice capped at 1 GPU
