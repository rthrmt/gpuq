"""Unit tests for the daemon's connection-level helpers: who a request comes
from, and the `gpuqd --restart-check` verdict."""

import os
import socket
import threading

import pytest

from gpuq import daemon as daemon_mod
from gpuq import protocol
from gpuq.daemon import _peer_uid, effective_user, restart_check


# --- identity ----------------------------------------------------------------

def test_peer_uid_is_the_connecting_process():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        assert _peer_uid(a) == os.getuid()
    finally:
        a.close()
        b.close()


@pytest.fixture
def passwd(monkeypatch):
    names = {1001: "alice", 1002: "bob"}

    class Entry:
        def __init__(self, name):
            self.pw_name = name

    def getpwuid(uid):
        return Entry(names[uid])      # KeyError for unknown uids, like pwd

    monkeypatch.setattr(daemon_mod.pwd, "getpwuid", getpwuid)


def test_other_users_claim_is_ignored(passwd):
    # alice (uid 1001) says she is bob: the kernel-reported uid wins.
    assert effective_user(1001, own_uid=0, claimed="bob") == "alice"
    assert effective_user(1001, own_uid=0, claimed="root") == "alice"
    assert effective_user(1001, own_uid=0, claimed="alice") == "alice"


def test_uid_without_passwd_entry_falls_back_to_the_number(passwd):
    assert effective_user(4242, own_uid=0, claimed="bob") == "4242"


@pytest.mark.parametrize("peer_uid, own_uid", [
    (0, 0),          # root talking to a root daemon
    (0, 1001),       # root talking to a user-space daemon
    (1001, 1001),    # the daemon's own user (dev runs, the test-suite)
    (None, 0),       # platform cannot identify the peer
])
def test_trusted_peers_may_act_for_any_user(passwd, peer_uid, own_uid):
    assert effective_user(peer_uid, own_uid, claimed="bob") == "bob"


# --- restart check -----------------------------------------------------------

@pytest.fixture
def fake_daemon(tmp_path):
    """A one-shot server answering 'status' with a canned reply."""
    path = str(tmp_path / "fake.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)

    def serve(reply):
        def run():
            conn, _ = srv.accept()
            assert protocol.recv_one(conn)["type"] == "status"
            protocol.send_msg(conn, reply)
            conn.close()
        threading.Thread(target=run, daemon=True).start()
        return path

    yield serve
    srv.close()


def test_restart_check_old_daemon_with_jobs_is_unsafe(fake_daemon, capsys):
    # A gpuqd from before reattach existed sends no protocol_version.
    path = fake_daemon({"type": "status_response", "jobs": [{}, {}]})
    assert restart_check(path) == 1
    assert "2 active job(s) would be lost" in capsys.readouterr().out


def test_restart_check_old_daemon_without_jobs_is_safe(fake_daemon, capsys):
    path = fake_daemon({"type": "status_response", "jobs": []})
    assert restart_check(path) == 0
    assert "no active jobs" in capsys.readouterr().out


def test_restart_check_current_daemon_with_jobs_is_safe(fake_daemon, capsys):
    path = fake_daemon(protocol.status_response([{}, {}, {}]))
    assert restart_check(path) == 0
    assert "3 active job(s); they will reattach" in capsys.readouterr().out


def test_restart_check_without_daemon(tmp_path, capsys):
    assert restart_check(str(tmp_path / "absent.sock")) == 2
    assert "not reachable" in capsys.readouterr().out
