"""Unit tests for the pure scheduling logic and the GpuState fit rules."""

import pytest

from gpuq.gpu import GpuInfo
from gpuq.gpustate import GpuState
from gpuq.scheduler import (SchedJob, fair_order, queue_order, schedule_pass,
                            try_place)

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


def job(jid, user="alice", gpus=1, mem=None, share=False, ts="2026-01-01T00:00:00Z",
        prio=0):
    return SchedJob(jid, user, gpus, mem * GB if mem else None, share, ts, prio)


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


# --- priority: reorders a user's own jobs only -----------------------------

def _ids(jobs):
    return [j.job_id for j in jobs]


def test_priority_moves_job_ahead_of_own_older_jobs():
    jobs = [job("a1", ts="t1"), job("a2", ts="t2"), job("a3", ts="t3", prio=5)]
    assert _ids(fair_order(jobs, {})) == ["a3", "a1", "a2"]


def test_equal_priorities_are_plain_fifo():
    jobs = [job("a2", "alice", ts="t2", prio=3), job("b1", "bob", ts="t3", prio=3),
            job("a1", "alice", ts="t1", prio=3)]
    assert _ids(fair_order(jobs, {})) == ["a1", "a2", "b1"]


def test_priority_cannot_jump_another_users_job():
    # Bob submitted later; no priority he sets may put him ahead of Alice.
    jobs = [job("a1", "alice", ts="t1"), job("b1", "bob", ts="t2", prio=1000)]
    assert _ids(fair_order(jobs, {})) == ["a1", "b1"]


def test_priority_swaps_only_within_the_owners_slots():
    # Queue slots by submission: alice, bob, alice, bob. Bob promotes his
    # later job: his two jobs trade places, Alice's positions are unchanged.
    jobs = [job("a1", "alice", ts="t1"), job("b1", "bob", ts="t2"),
            job("a2", "alice", ts="t3"), job("b2", "bob", ts="t4", prio=1)]
    assert _ids(fair_order(jobs, {})) == ["a1", "b2", "a2", "b1"]


def test_priority_respected_when_submitted_in_the_same_second():
    # submitted_at has 1s resolution, so the job_id tie-break must not be able
    # to put a lower-priority job first.
    jobs = [job("a", ts="t1"), job("b", ts="t1"), job("z", ts="t1", prio=1)]
    assert _ids(fair_order(jobs, {}))[0] == "z"


def test_negative_priority_goes_to_the_back_of_own_queue():
    jobs = [job("a1", ts="t1", prio=-1), job("a2", ts="t2"), job("a3", ts="t3")]
    assert _ids(fair_order(jobs, {})) == ["a2", "a3", "a1"]


def test_fairness_still_outranks_priority():
    # Alice already has a job running; her high priority does not beat Bob.
    jobs = [job("a1", "alice", ts="t1", prio=99), job("b1", "bob", ts="t2")]
    assert _ids(fair_order(jobs, {"alice": 1})) == ["b1", "a1"]


def test_queue_order_ignores_running_counts():
    jobs = [job("a1", "alice", ts="t1"), job("b1", "bob", ts="t2"),
            job("a2", "alice", ts="t3", prio=1)]
    assert _ids(queue_order(jobs)) == ["a2", "b1", "a1"]


def test_schedule_pass_places_high_priority_job_first():
    state = make_state([(0, 24, 0)])
    placed = []

    def commit(j, ids):
        placed.append(j.job_id)
        for i in ids:
            state.reserve(i, j.job_id, 0, False)

    jobs = [job("a1", ts="t1"), job("a2", ts="t2"), job("a3", ts="t3", prio=1)]
    assert schedule_pass(jobs, state, {}, commit=commit) == 1
    assert placed == ["a3"]


def test_unplaceable_high_priority_job_does_not_block_own_jobs():
    # Skip-don't-block still applies: the promoted 2-GPU job cannot fit on the
    # single GPU, so the owner's ordinary job runs instead.
    state = make_state([(0, 24, 0)])
    placed = []

    def commit(j, ids):
        placed.append(j.job_id)
        for i in ids:
            state.reserve(i, j.job_id, 0, False)

    jobs = [job("small", ts="t1"), job("big", ts="t2", gpus=2, prio=5)]
    assert schedule_pass(jobs, state, {}, commit=commit) == 1
    assert placed == ["small"]


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
