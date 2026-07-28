"""Unit tests for NDJSON framing and message builders."""

import socket

import pytest

from gpuq import protocol
from gpuq.errors import ProtocolError


def socketpair():
    return socket.socketpair()


def test_round_trip_single_message():
    a, b = socketpair()
    try:
        protocol.send_msg(a, protocol.go([0, 1]))
        msg = protocol.recv_one(b)
        assert msg == {"type": "go", "gpu_ids": [0, 1]}
    finally:
        a.close()
        b.close()


def test_multiple_messages_in_one_buffer():
    a, b = socketpair()
    try:
        protocol.send_msg(a, protocol.status())
        protocol.send_msg(a, protocol.done("j1", 0))
        a.close()
        msgs = list(protocol.iter_msgs(b))
        assert protocol.get_type(msgs[0]) == "status"
        assert msgs[1] == {"type": "done", "job_id": "j1", "exit_code": 0}
    finally:
        b.close()


def test_partial_frame_reassembly():
    a, b = socketpair()
    try:
        payload = b'{"type":"sta'
        a.sendall(payload)
        # Send the rest plus a newline a moment later.
        a.sendall(b'tus"}\n')
        a.close()
        msg = protocol.recv_one(b)
        assert msg == {"type": "status"}
    finally:
        b.close()


def test_malformed_json_raises():
    a, b = socketpair()
    try:
        a.sendall(b"not json at all\n")
        a.close()
        with pytest.raises(ProtocolError):
            protocol.recv_one(b)
    finally:
        b.close()


def test_closed_peer_returns_none():
    a, b = socketpair()
    a.close()
    assert protocol.recv_one(b) is None
    b.close()


def test_register_builder_fields():
    msg = protocol.register(
        job_id="j", user="u", cmd="c", name="n", gpus_requested=2,
        mem_requested=123, shareable=True, log_file="/l", submitted_at="t",
    )
    assert msg["type"] == "register"
    assert msg["gpus_requested"] == 2
    assert msg["mem_requested"] == 123
    assert msg["shareable"] is True


def test_get_type_requires_string():
    with pytest.raises(ProtocolError):
        protocol.get_type({"no": "type"})
