"""Job-completion emails, sent by the waiter through a fake sendmail."""

import os

import pytest

from gpuq import protocol
from tests.conftest import wait_until

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork()")


def test_email_on_success(daemon, submit, mailbox):
    _, job_id, log_file = submit("echo all-good", name="baseline",
                                 notify_to=["alice@example.com"])
    assert wait_until(lambda: len(mailbox()) == 1)
    msg = mailbox()[0]
    assert msg["To"] == "alice@example.com"
    assert msg["Subject"].startswith(f"[gpuq] job {job_id[:4]} (baseline) finished on ")
    assert msg["X-Gpuq-Job-Id"] == job_id
    body = msg.get_content()
    assert "done (exit code 0)" in body
    assert log_file in body
    assert "all-good" in body  # the log tail


def test_email_on_failure_names_exit_code(daemon, submit, mailbox):
    _, job_id, _ = submit("exit 3", notify_to=["a@example.com", "b@example.com"])
    assert wait_until(lambda: len(mailbox()) == 1)
    msg = mailbox()[0]
    assert msg["To"] == "a@example.com, b@example.com"
    assert "failed (exit code 3)" in msg["Subject"]


def test_email_on_cancelled_running_job(daemon, submit, mailbox):
    daemon.set_gpus([(0, 0, 24576, 24576)])
    _, job_id, _ = submit("sleep 60", notify_to=["alice@example.com"])
    assert wait_until(lambda: any(j["job_id"] == job_id and j["status"] == "running"
                                  for j in daemon.status()))
    daemon.request(protocol.cancel(job_id, "tester", force=True))
    assert wait_until(lambda: len(mailbox()) == 1, timeout=15)
    assert " cancelled on " in mailbox()[0]["Subject"]


def test_no_email_without_address(daemon, submit, mailbox):
    _, job_id, log_file = submit("echo quiet")
    assert wait_until(lambda: "quiet" in open(log_file).read()
                      if os.path.exists(log_file) else False)
    # The waiter has no address, so nothing should ever arrive.
    assert not wait_until(lambda: mailbox(), timeout=1.0)


def test_delivery_failure_is_noted_in_log(daemon, submit, mailbox, monkeypatch):
    monkeypatch.setenv("GPUQ_FAKE_SENDMAIL_EXIT", "75")
    _, job_id, log_file = submit("echo hi", notify_to=["alice@example.com"])
    assert wait_until(lambda: os.path.exists(log_file)
                      and "could not send email" in open(log_file).read())
    content = open(log_file).read()
    assert "exited with 75" in content and "delivery refused" in content
