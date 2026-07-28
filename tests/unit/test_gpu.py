"""Unit tests for gpuq.gpu.parse_csv and integration tests for query_gpus."""

from __future__ import annotations

import os
import pytest

from gpuq.gpu import GpuInfo, NvidiaSmiError, parse_csv, query_gpus

_MIB = 1024 * 1024

# Path to the fake nvidia-smi shim (relative to repo root, resolved at runtime).
_FAKEBIN = os.path.join(os.path.dirname(__file__), "..", "fakebin", "nvidia-smi")


# ---------------------------------------------------------------------------
# parse_csv — normal operation
# ---------------------------------------------------------------------------


def test_parse_csv_single_gpu():
    text = "0, 512, 24576, 24064\n"
    gpus = parse_csv(text)
    assert len(gpus) == 1
    g = gpus[0]
    assert g.index == 0
    assert g.mem_used == 512 * _MIB
    assert g.mem_total == 24576 * _MIB
    assert g.mem_free == 24064 * _MIB


def test_parse_csv_multi_gpu():
    text = (
        "0, 100, 8192, 8092\n"
        "1, 200, 16384, 16184\n"
        "2, 300, 24576, 24276\n"
    )
    gpus = parse_csv(text)
    assert len(gpus) == 3
    assert gpus[0].index == 0
    assert gpus[0].mem_used == 100 * _MIB
    assert gpus[1].index == 1
    assert gpus[1].mem_total == 16384 * _MIB
    assert gpus[2].index == 2
    assert gpus[2].mem_free == 24276 * _MIB


def test_parse_csv_byte_conversion():
    """1 MiB == 1 048 576 bytes."""
    text = "0, 1, 1, 0\n"
    gpus = parse_csv(text)
    assert gpus[0].mem_used == 1_048_576
    assert gpus[0].mem_total == 1_048_576
    assert gpus[0].mem_free == 0


# ---------------------------------------------------------------------------
# parse_csv — whitespace and blank lines
# ---------------------------------------------------------------------------


def test_parse_csv_whitespace_tolerance():
    """Extra spaces around fields must be stripped."""
    text = "  0 ,  256 ,  8192 ,  7936 \n"
    gpus = parse_csv(text)
    assert len(gpus) == 1
    assert gpus[0].mem_used == 256 * _MIB


def test_parse_csv_blank_lines_skipped():
    text = "\n0, 0, 8192, 8192\n\n1, 64, 8192, 8128\n\n"
    gpus = parse_csv(text)
    assert len(gpus) == 2
    assert gpus[0].index == 0
    assert gpus[1].index == 1


def test_parse_csv_empty_string():
    assert parse_csv("") == []


def test_parse_csv_only_blank_lines():
    assert parse_csv("\n\n   \n") == []


# ---------------------------------------------------------------------------
# parse_csv — N/A handling
# ---------------------------------------------------------------------------


def test_parse_csv_na_used_treated_as_zero():
    text = "0, N/A, 8192, 8192\n"
    gpus = parse_csv(text)
    assert len(gpus) == 1
    assert gpus[0].mem_used == 0


def test_parse_csv_bracketed_na_treated_as_zero():
    text = "0, [N/A], 8192, [N/A]\n"
    gpus = parse_csv(text)
    assert len(gpus) == 1
    assert gpus[0].mem_used == 0
    assert gpus[0].mem_free == 0


def test_parse_csv_na_index_line_skipped():
    text = "N/A, 0, 8192, 8192\n0, 64, 8192, 8128\n"
    gpus = parse_csv(text)
    assert len(gpus) == 1
    assert gpus[0].index == 0


def test_parse_csv_bracketed_na_index_skipped():
    text = "[N/A], 0, 8192, 8192\n"
    gpus = parse_csv(text)
    assert gpus == []


# ---------------------------------------------------------------------------
# parse_csv — managed filter
# ---------------------------------------------------------------------------


def test_parse_csv_managed_filter_keeps_only_selected():
    text = "0, 0, 8192, 8192\n1, 0, 8192, 8192\n2, 0, 8192, 8192\n3, 0, 8192, 8192\n"
    gpus = parse_csv(text, managed=[1, 3])
    assert [g.index for g in gpus] == [1, 3]


def test_parse_csv_managed_filter_ascending_order():
    """Result is sorted ascending even if managed is given in descending order."""
    text = "0, 0, 8192, 8192\n1, 0, 8192, 8192\n2, 0, 8192, 8192\n"
    gpus = parse_csv(text, managed=[2, 0])
    assert [g.index for g in gpus] == [0, 2]


def test_parse_csv_managed_filter_none_keeps_all():
    text = "0, 0, 8192, 8192\n1, 0, 8192, 8192\n"
    gpus = parse_csv(text, managed=None)
    assert len(gpus) == 2


def test_parse_csv_managed_filter_empty_set():
    text = "0, 0, 8192, 8192\n1, 0, 8192, 8192\n"
    gpus = parse_csv(text, managed=[])
    assert gpus == []


# ---------------------------------------------------------------------------
# parse_csv — result is always sorted by index
# ---------------------------------------------------------------------------


def test_parse_csv_result_sorted_by_index():
    """Even if nvidia-smi reports GPUs out of order, result is sorted."""
    text = "3, 0, 8192, 8192\n1, 0, 8192, 8192\n0, 0, 8192, 8192\n"
    gpus = parse_csv(text)
    assert [g.index for g in gpus] == [0, 1, 3]


# ---------------------------------------------------------------------------
# query_gpus — integration tests using the fake shim
# ---------------------------------------------------------------------------


def test_query_gpus_with_fake_shim(tmp_path, monkeypatch):
    """query_gpus returns correct GpuInfo when pointed at the fake shim."""
    state_file = tmp_path / "gpus.txt"
    state_file.write_text("0, 512, 24576, 24064\n1, 1024, 24576, 23552\n")

    monkeypatch.setenv("GPUQ_NVIDIA_SMI", os.path.abspath(_FAKEBIN))
    monkeypatch.setenv("GPUQ_FAKE_GPUS", str(state_file))

    gpus = query_gpus()

    assert len(gpus) == 2
    assert gpus[0] == GpuInfo(
        index=0,
        mem_used=512 * _MIB,
        mem_total=24576 * _MIB,
        mem_free=24064 * _MIB,
    )
    assert gpus[1] == GpuInfo(
        index=1,
        mem_used=1024 * _MIB,
        mem_total=24576 * _MIB,
        mem_free=23552 * _MIB,
    )


def test_query_gpus_missing_state_file_raises(tmp_path, monkeypatch):
    """When the fake shim exits non-zero (missing state file), NvidiaSmiError is raised."""
    missing = tmp_path / "does_not_exist.txt"

    monkeypatch.setenv("GPUQ_NVIDIA_SMI", os.path.abspath(_FAKEBIN))
    monkeypatch.setenv("GPUQ_FAKE_GPUS", str(missing))

    with pytest.raises(NvidiaSmiError):
        query_gpus()


def test_query_gpus_binary_not_found_raises(monkeypatch):
    """When the binary path is wrong, NvidiaSmiError is raised."""
    monkeypatch.setenv("GPUQ_NVIDIA_SMI", "/no/such/nvidia-smi")

    with pytest.raises(NvidiaSmiError):
        query_gpus()
