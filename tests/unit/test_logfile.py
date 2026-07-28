"""Unit tests for gpuq.logfile — sanitize_cmd, timestamp, compute_log_path,
write_header."""

import io
import os
from datetime import datetime

import pytest

from gpuq.logfile import (
    sanitize_cmd,
    timestamp,
    compute_log_path,
    write_header,
)


# ---------------------------------------------------------------------------
# sanitize_cmd
# ---------------------------------------------------------------------------

class TestSanitizeCmd:
    def test_simple_word(self):
        assert sanitize_cmd("python") == "python"

    def test_lowercase_conversion(self):
        assert sanitize_cmd("PYTHON") == "python"

    def test_dots_replaced(self):
        # '.' is non-alphanumeric -> replaced with '_'
        result = sanitize_cmd("train.py")
        assert "." not in result
        assert result == "train_py"

    def test_full_command_with_dashes_and_dots(self):
        # "python train.py --lr 0.001"
        result = sanitize_cmd("python train.py --lr 0.001")
        assert result == "python_train_py_lr_0_001"

    def test_dedup_underscores(self):
        # Multiple non-alnum chars in a row -> single underscore
        result = sanitize_cmd("a  --  b")
        assert "__" not in result
        assert result == "a_b"

    def test_strip_leading_underscore(self):
        # Leading non-alnum char produces leading underscore which is stripped
        result = sanitize_cmd("--flag")
        assert not result.startswith("_")

    def test_strip_trailing_underscore(self):
        result = sanitize_cmd("cmd--")
        assert not result.endswith("_")

    def test_60_char_truncation(self):
        long_cmd = "a" * 80
        result = sanitize_cmd(long_cmd)
        assert len(result) <= 60

    def test_truncation_strips_trailing_underscore(self):
        # Build a string where the 60th char falls in the middle of a
        # non-alnum run, so truncation would leave a trailing underscore.
        # "a" * 59 + "---rest": after sanitization -> "a"*59 + "_" + rest
        # Truncating at 60 leaves "a"*59 + "_"; trailing _ should be stripped.
        cmd = "a" * 59 + "---rest"
        result = sanitize_cmd(cmd)
        assert len(result) <= 60
        assert not result.endswith("_")

    def test_name_sanitization(self):
        # Names go through the same sanitize_cmd function
        assert sanitize_cmd("baseline_run") == "baseline_run"

    def test_run_experiment_sh(self):
        result = sanitize_cmd("./run_experiment.sh")
        assert result == "run_experiment_sh"

    def test_mixed_case_and_numbers(self):
        result = sanitize_cmd("Python3 Train.py")
        assert result == "python3_train_py"

    def test_single_alnum(self):
        assert sanitize_cmd("a") == "a"

    def test_only_non_alnum_gives_empty(self):
        # All non-alnum: after replacement and stripping -> empty string
        result = sanitize_cmd("---")
        assert result == ""

    def test_number_in_command(self):
        result = sanitize_cmd("train_v2")
        assert result == "train_v2"


# ---------------------------------------------------------------------------
# timestamp
# ---------------------------------------------------------------------------

class TestTimestamp:
    def test_fixed_datetime(self):
        dt = datetime(2026, 5, 16, 14, 30, 22)
        assert timestamp(dt) == "20260516_143022"

    def test_format_length(self):
        dt = datetime(2026, 1, 1, 0, 0, 0)
        ts = timestamp(dt)
        assert len(ts) == 15  # "YYYYMMDD_HHMMSS"

    def test_default_returns_string(self):
        ts = timestamp()
        assert isinstance(ts, str)
        assert len(ts) == 15

    def test_format_pattern(self):
        import re
        ts = timestamp(datetime(2026, 12, 31, 23, 59, 59))
        assert re.match(r"^\d{8}_\d{6}$", ts)

    def test_zero_padding(self):
        dt = datetime(2026, 1, 2, 3, 4, 5)
        assert timestamp(dt) == "20260102_030405"


# ---------------------------------------------------------------------------
# compute_log_path
# ---------------------------------------------------------------------------

class TestComputeLogPath:
    def test_uses_cmd_when_no_name(self):
        dt = datetime(2026, 5, 16, 14, 30, 22)
        path = compute_log_path(None, "python train.py", when=dt, logs_dir="/tmp/logs")
        assert path == "/tmp/logs/python_train_py_20260516_143022.log"

    def test_uses_name_when_provided(self):
        dt = datetime(2026, 5, 16, 14, 30, 22)
        path = compute_log_path("baseline_run", "python train.py", when=dt, logs_dir="/tmp/logs")
        assert path == "/tmp/logs/baseline_run_20260516_143022.log"

    def test_empty_name_falls_back_to_cmd(self):
        dt = datetime(2026, 5, 16, 14, 30, 22)
        path = compute_log_path("", "python train.py", when=dt, logs_dir="/tmp/logs")
        assert path == "/tmp/logs/python_train_py_20260516_143022.log"

    def test_whitespace_only_name_falls_back_to_cmd(self):
        dt = datetime(2026, 5, 16, 14, 30, 22)
        path = compute_log_path("   ", "python train.py", when=dt, logs_dir="/tmp/logs")
        assert path == "/tmp/logs/python_train_py_20260516_143022.log"

    def test_path_ends_with_dot_log(self):
        path = compute_log_path(None, "myscript.sh", logs_dir="/tmp/logs")
        assert path.endswith(".log")

    def test_logs_dir_joined_correctly(self):
        dt = datetime(2026, 1, 1, 0, 0, 0)
        path = compute_log_path("myjob", "cmd", when=dt, logs_dir="/custom/dir")
        assert path.startswith("/custom/dir/")

    def test_no_directory_created(self, tmp_path):
        # compute_log_path is pure — it must NOT create any directory
        nonexistent = str(tmp_path / "does_not_exist")
        assert not os.path.exists(nonexistent)
        compute_log_path(None, "cmd", logs_dir=nonexistent)
        assert not os.path.exists(nonexistent), "compute_log_path must not create dirs"

    def test_name_sanitized(self):
        dt = datetime(2026, 5, 16, 14, 30, 22)
        path = compute_log_path("My Job!", "cmd", when=dt, logs_dir="/tmp/logs")
        assert path == "/tmp/logs/my_job_20260516_143022.log"

    def test_default_logs_dir_contains_gpuq(self):
        # Without specifying logs_dir the default should include ".gpuq/logs"
        path = compute_log_path(None, "cmd")
        assert ".gpuq" in path
        assert "logs" in path


# ---------------------------------------------------------------------------
# write_header
# ---------------------------------------------------------------------------

class TestWriteHeader:
    def _capture_header(self, meta: dict) -> str:
        buf = io.StringIO()
        write_header(buf, meta)
        return buf.getvalue()

    def test_starts_with_job_log_line(self):
        output = self._capture_header({})
        assert output.startswith("# gpuq job log\n")

    def test_ends_with_separator(self):
        output = self._capture_header({})
        lines = output.splitlines()
        assert lines[-1] == "# " + "-" * 60

    def test_includes_job_id(self):
        output = self._capture_header({"job_id": "abc-123"})
        assert "# gpuq job_id: abc-123" in output

    def test_skips_none_values(self):
        output = self._capture_header({"job_id": "abc-123", "name": None, "user": "alice"})
        assert "name" not in output
        assert "# gpuq user: alice" in output

    def test_all_none_only_header_and_separator(self):
        output = self._capture_header({"job_id": None, "user": None})
        lines = output.strip().splitlines()
        assert lines[0] == "# gpuq job log"
        assert lines[-1] == "# " + "-" * 60
        assert len(lines) == 2

    def test_multiple_meta_keys(self):
        meta = {
            "job_id": "uuid-1",
            "user": "bob",
            "cmd": "python train.py",
            "gpu_ids": "[0, 1]",
        }
        output = self._capture_header(meta)
        assert "# gpuq job_id: uuid-1" in output
        assert "# gpuq user: bob" in output
        assert "# gpuq cmd: python train.py" in output
        assert "# gpuq gpu_ids: [0, 1]" in output

    def test_flush_called(self):
        """write_header must flush the file object."""
        class TrackingStringIO(io.StringIO):
            def __init__(self):
                super().__init__()
                self.flush_called = False

            def flush(self):
                self.flush_called = True
                super().flush()

        buf = TrackingStringIO()
        write_header(buf, {"job_id": "x"})
        assert buf.flush_called, "write_header did not call flush()"

    def test_format_each_line(self):
        output = self._capture_header({"submitted_at": "2026-05-16T14:30:22Z"})
        assert "# gpuq submitted_at: 2026-05-16T14:30:22Z" in output

    def test_comment_prefix_on_all_lines(self):
        output = self._capture_header({"user": "carol", "cmd": "run.sh"})
        for line in output.splitlines():
            assert line.startswith("#"), f"Line missing # prefix: {line!r}"
