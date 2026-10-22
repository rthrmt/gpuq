"""`gpuq logs` must print the log file byte for byte, carriage returns included.

tqdm redraws its progress bar with "\r"; reading the file in text mode's default newline
handling turned each "\r" into "\n", so every refresh printed as a new line.
"""
import argparse

import pytest

from gpuq import cli

LOG = "header\ntest:  10%|#  | 1/10\rtest:  20%|## | 2/10\rtest: 100%|###| 10/10\ndone\n"


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    path = tmp_path / "job.log"
    path.write_bytes(LOG.encode())
    monkeypatch.setattr(cli, "_query_status", lambda sock_path: [])
    monkeypatch.setattr(cli, "_resolve_log_file", lambda *a: str(path))
    return path


def test_logs_keeps_carriage_returns(log_file, capsys):
    assert cli.cmd_logs(argparse.Namespace(job_id="abcd", follow=False), "/nonexistent") == 0
    assert capsys.readouterr().out == LOG


def test_logs_follow_keeps_carriage_returns(log_file, monkeypatch, capsys):
    def stop(_seconds):
        raise KeyboardInterrupt  # _tail_follow's documented exit (Ctrl-C)

    monkeypatch.setattr(cli.time, "sleep", stop)
    assert cli.cmd_logs(argparse.Namespace(job_id="abcd", follow=True), "/nonexistent") == 0
    assert capsys.readouterr().out == LOG
