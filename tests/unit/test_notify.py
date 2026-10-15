"""Notification email building, address parsing, and the per-user default."""

import argparse
import datetime

import pytest

from gpuq import cli, config, notify

UTC = datetime.timezone.utc


# --- parse_addresses --------------------------------------------------------

def test_parse_single_and_list():
    assert notify.parse_addresses("a@example.com") == ["a@example.com"]
    assert notify.parse_addresses(" a@x.org, b@y.de ,") == ["a@x.org", "b@y.de"]


@pytest.mark.parametrize("raw", [
    "", " , ", "alice", "alice@localhost", "a b@x.org", "a@x.org\nBcc: c@y.org",
    "Alice <a@x.org>", "a@@x.org",
])
def test_parse_rejects(raw):
    with pytest.raises(ValueError):
        notify.parse_addresses(raw)


# --- small formatters -------------------------------------------------------

def test_outcome_and_exit_description():
    assert notify.job_outcome(0, False) == "done"
    assert notify.job_outcome(2, False) == "failed"
    assert notify.job_outcome(-15, True) == "cancelled"
    assert notify.describe_exit(0) == "exit code 0"
    assert notify.describe_exit(-9) == "killed by SIGKILL"


def test_format_duration():
    assert notify.format_duration(5) == "5s"
    assert notify.format_duration(125) == "2m 05s"
    assert notify.format_duration(3723) == "1h 02m 03s"


# --- build_message ----------------------------------------------------------

def _build(**kw):
    started = datetime.datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
    base = dict(
        to=["alice@example.com"], job_id="7a3f0000-1111-2222-3333-444455556666",
        name="baseline", cmd="python train.py", host="gpubox", gpu_ids=[0, 1],
        exit_code=0, cancelled=False, started_at=started,
        finished_at=started + datetime.timedelta(seconds=3723),
        log_file="/home/alice/.gpuq/logs/baseline.log", log_tail=["epoch 9", "done"],
    )
    base.update(kw)
    return notify.build_message(**base)


def test_message_success():
    msg = _build()
    assert msg["Subject"] == "[gpuq] job 7a3f (baseline) finished on gpubox"
    assert msg["To"] == "alice@example.com"
    assert msg["Auto-Submitted"] == "auto-generated"
    body = msg.get_content()
    assert "done (exit code 0)" in body
    assert "1h 02m 03s" in body
    assert "GPUs:     0,1" in body
    assert "Last 2 lines of the log" in body and "    epoch 9" in body


def test_message_failure_and_cancel_subjects():
    assert _build(exit_code=7)["Subject"] == \
        "[gpuq] job 7a3f (baseline) failed (exit code 7) on gpubox"
    assert _build(exit_code=-15, cancelled=True)["Subject"] == \
        "[gpuq] job 7a3f (baseline) cancelled on gpubox"


def test_message_multiline_command_stays_one_header_line():
    msg = _build(name=None, cmd="python a.py \\\n  --lr 0.1\n" + "x" * 200)
    assert "\n" not in msg["Subject"]
    assert msg["Subject"].startswith("[gpuq] job 7a3f (python a.py \\ --lr 0.1 x")
    assert "..." in msg["Subject"]


def test_message_without_log_tail():
    assert "Last" not in _build(log_tail=None).get_content()


# --- tail_log ---------------------------------------------------------------

def test_tail_log_collapses_progress_bars(tmp_path):
    p = tmp_path / "job.log"
    p.write_bytes(b"one\n 10%\r 50%\r100%\ntwo\n\n\n")
    assert notify.tail_log(str(p)) == ["one", "100%", "two"]


def test_tail_log_limits_lines_and_drops_partial_first_line(tmp_path):
    p = tmp_path / "job.log"
    p.write_text("".join(f"line {i:06d}\n" for i in range(5000)))
    tail = notify.tail_log(str(p), lines=3)
    assert tail == ["line 004997", "line 004998", "line 004999"]
    assert all(line.startswith("line ") for line in notify.tail_log(str(p), lines=10**6))


def test_tail_log_skips_gpuq_header(tmp_path):
    from gpuq.logfile import write_header
    p = tmp_path / "job.log"
    with open(p, "w") as f:
        write_header(f, {"job_id": "abcd", "cmd": "echo hi"})
        f.write("hi\n")
    assert notify.tail_log(str(p)) == ["hi"]
    with open(p, "w") as f:
        write_header(f, {"job_id": "abcd"})
    assert notify.tail_log(str(p)) == []


def test_tail_log_missing_file(tmp_path):
    assert notify.tail_log(str(tmp_path / "nope.log")) is None


# --- send -------------------------------------------------------------------

def test_send_reports_sendmail_failure(tmp_path):
    fake = tmp_path / "sendmail"
    fake.write_text("#!/bin/sh\ncat >/dev/null\necho 'no relay' >&2\nexit 75\n")
    fake.chmod(0o755)
    with pytest.raises(OSError, match="exited with 75: no relay"):
        notify.send(_build(), str(fake))


def test_send_missing_binary(tmp_path):
    with pytest.raises(OSError):
        notify.send(_build(), str(tmp_path / "missing"))


# --- per-user config --------------------------------------------------------

def test_user_conf_path_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("GPUQ_USER_CONF", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config.user_conf_path() == str(tmp_path / "gpuq" / "config")
    monkeypatch.setenv("GPUQ_USER_CONF", "/x/y")
    assert config.user_conf_path() == "/x/y"


def test_user_default_email(tmp_path):
    p = tmp_path / "config"
    assert config.user_default_email(str(p)) is None
    p.write_text("[notify]\nemail = alice@example.com  # mine\n")
    assert config.user_default_email(str(p)) == "alice@example.com"
    p.write_text("[other]\nx = 1\n")
    assert config.user_default_email(str(p)) is None
    p.write_text("email = no section\n")
    with pytest.raises(config.ConfigError):
        config.user_default_email(str(p))


def test_sendmail_path_resolution(monkeypatch, tmp_path):
    conf = tmp_path / "gpuq.conf"
    monkeypatch.setenv("GPUQ_CONF", str(conf))
    monkeypatch.delenv("GPUQ_SENDMAIL", raising=False)
    assert config.client_sendmail_path() == "/usr/sbin/sendmail"
    conf.write_text("[mail]\nsendmail = /usr/bin/msmtp\n")
    assert config.client_sendmail_path() == "/usr/bin/msmtp"
    monkeypatch.setenv("GPUQ_SENDMAIL", "/fake")
    assert config.client_sendmail_path() == "/fake"


# --- cli: which address a submission uses -----------------------------------

@pytest.fixture
def user_conf(tmp_path, monkeypatch):
    p = tmp_path / "config"
    monkeypatch.setenv("GPUQ_USER_CONF", str(p))
    return p


def _args(email=None, no_email=False):
    return argparse.Namespace(email=email, no_email=no_email)


def test_resolve_uses_flag_then_default(user_conf):
    assert cli._resolve_notify_to(_args()) is None
    user_conf.write_text("[notify]\nemail = me@example.com\n")
    assert cli._resolve_notify_to(_args()) == ["me@example.com"]
    assert cli._resolve_notify_to(_args(email="x@y.org")) == ["x@y.org"]
    assert cli._resolve_notify_to(_args(no_email=True)) is None


def test_resolve_bad_address_dies(user_conf, capsys):
    user_conf.write_text("[notify]\nemail = not-an-address\n")
    with pytest.raises(SystemExit):
        cli._resolve_notify_to(_args())
    assert "invalid email address 'not-an-address'" in capsys.readouterr().err


def test_submit_flags_are_exclusive():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["submit", "true", "--email", "a@b.org", "--no-email"])
