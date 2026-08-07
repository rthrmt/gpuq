"""Unit tests for gpuq.gpustate.GpuState, focused on the snapshot() accessor
used to build placement_events audit rows."""

from gpuq.gpu import GpuInfo
from gpuq.gpustate import GpuState

_GB = 1024 ** 3


def _state(managed=(0, 1), idle_threshold=500 * 1024 * 1024):
    return GpuState(list(managed), idle_threshold)


def test_snapshot_empty_before_any_reconcile():
    state = _state()
    assert state.snapshot() == []


def test_snapshot_reflects_most_recent_reconcile():
    state = _state()
    live = [
        GpuInfo(index=0, mem_used=0, mem_total=24 * _GB, mem_free=24 * _GB, utilization_pct=0),
        GpuInfo(index=1, mem_used=2 * _GB, mem_total=46 * _GB, mem_free=44 * _GB, utilization_pct=10),
    ]
    state.reconcile(live)
    snap = state.snapshot()
    assert {g.index for g in snap} == {0, 1}
    gpu1 = next(g for g in snap if g.index == 1)
    assert gpu1.mem_used == 2 * _GB
    assert gpu1.utilization_pct == 10


def test_snapshot_updates_after_second_reconcile():
    state = _state()
    state.reconcile([GpuInfo(index=0, mem_used=0, mem_total=24 * _GB, mem_free=24 * _GB)])
    state.reconcile([GpuInfo(index=0, mem_used=5 * _GB, mem_total=24 * _GB, mem_free=19 * _GB)])
    snap = state.snapshot()
    assert len(snap) == 1
    assert snap[0].mem_used == 5 * _GB


def test_snapshot_returns_a_copy_not_live_reference():
    state = _state()
    state.reconcile([GpuInfo(index=0, mem_used=0, mem_total=24 * _GB, mem_free=24 * _GB)])
    snap = state.snapshot()
    snap.append(GpuInfo(index=99, mem_used=0, mem_total=0, mem_free=0))
    # Mutating the returned list must not affect internal state.
    assert len(state.snapshot()) == 1


def test_snapshot_only_includes_managed_gpus():
    state = _state(managed=(0,))
    state.reconcile([
        GpuInfo(index=0, mem_used=0, mem_total=24 * _GB, mem_free=24 * _GB),
        GpuInfo(index=1, mem_used=0, mem_total=24 * _GB, mem_free=24 * _GB),
    ])
    snap = state.snapshot()
    assert [g.index for g in snap] == [0]


# --- a light check that eligible()'s idle-threshold rule behaves as
# documented in CLAUDE.md, since the audit feature exists specifically to
# make this rule's behavior diagnosable after the fact.

def test_eligible_exclusive_rejects_gpu_above_idle_threshold():
    state = _state(idle_threshold=500 * 1024 * 1024)
    state.reconcile([GpuInfo(index=0, mem_used=2 * _GB, mem_total=24 * _GB, mem_free=22 * _GB)])
    assert state.eligible(0, shareable=False, mem_bytes=None) is False


def test_eligible_exclusive_accepts_gpu_below_idle_threshold():
    state = _state(idle_threshold=500 * 1024 * 1024)
    state.reconcile([GpuInfo(index=0, mem_used=0, mem_total=24 * _GB, mem_free=24 * _GB)])
    assert state.eligible(0, shareable=False, mem_bytes=None) is True
