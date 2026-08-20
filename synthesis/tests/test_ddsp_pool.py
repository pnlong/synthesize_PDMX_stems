"""Unit tests for DDSP GPU id parsing and persistent pool scheduling."""

from __future__ import annotations

import io
import json
import threading
from types import SimpleNamespace

import pytest

from synthesis.ddsp.env import parse_ddsp_gpu_ids
from synthesis.ddsp.pool import DdspWorkerPool, _ServeWorker


def test_parse_ddsp_gpu_ids_default():
    assert parse_ddsp_gpu_ids(force_cpu=False, cuda_visible=None) == ["0"]


def test_parse_ddsp_gpu_ids_force_cpu():
    assert parse_ddsp_gpu_ids(force_cpu=True) == ["-1"]


def test_hybrid_neural_song_workers_oneshot(monkeypatch):
    monkeypatch.setenv("SPDMX_DDSP_ONESHOT", "1")
    from synthesis.synthesize import _hybrid_neural_song_workers

    assert _hybrid_neural_song_workers() == 1
    assert parse_ddsp_gpu_ids(force_cpu=False, cuda_visible="2,3") == ["2", "3"]


class _FakeProc:
    def __init__(self):
        self._queue: list[dict] = []
        self._cond = threading.Condition()
        self.returncode = None
        self.stdin = self
        self.stdout = self
        self.stderr = io.StringIO("")
        self._alive = True
        self.written: list[dict] = []
        self.gpu_id = "?"

    def write(self, data: str) -> None:
        req = json.loads(data.strip())
        self.written.append(req)
        with self._cond:
            self._queue.append(
                {
                    "id": req.get("id"),
                    "ok": True,
                    "echo_command": req.get("command"),
                    "gpu": self.gpu_id,
                }
            )
            self._cond.notify_all()

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def readline(self) -> str:
        with self._cond:
            while not self._queue and self._alive:
                self._cond.wait(timeout=0.05)
            if not self._queue:
                return ""
            return json.dumps(self._queue.pop(0)) + "\n"

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        self._alive = False
        with self._cond:
            self._cond.notify_all()
        return 0

    def kill(self):
        self._alive = False
        with self._cond:
            self._cond.notify_all()


def _fake_worker(gpu_id: str) -> _ServeWorker:
    proc = _FakeProc()
    proc.gpu_id = gpu_id
    worker = _ServeWorker(gpu_id=gpu_id, proc=proc)
    worker.start_io()
    return worker


def test_pool_submit_exclusive_and_round_robin():
    w0 = _fake_worker("0")
    w1 = _fake_worker("1")
    pool = DdspWorkerPool([w0, w1])

    barrier = threading.Barrier(3)
    results: list[dict] = []
    lock = threading.Lock()

    def job():
        barrier.wait()
        status = pool.submit({"command": "ping"}, timeout_sec=2.0)
        with lock:
            results.append(status)

    threads = [threading.Thread(target=job) for _ in range(2)]
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()

    assert len(results) == 2
    assert all(r.get("ok") for r in results)
    used = {r.get("gpu") for r in results}
    assert used == {"0", "1"}

    pool.shutdown()
    assert pool._closed


def test_pool_error_propagation():
    w0 = _fake_worker("0")
    pool = DdspWorkerPool([w0])

    def boom_locked(payload, *, timeout_sec):
        return {"id": 1, "ok": False, "error": "boom"}

    w0._request_locked = boom_locked  # type: ignore[method-assign]
    status = pool.submit({"command": "midi_ddsp"}, timeout_sec=1.0)
    assert status["ok"] is False
    assert "boom" in status["error"]
    pool.shutdown()


def test_ddsp_oneshot_env(monkeypatch):
    from synthesis.ddsp import pool as pool_mod

    monkeypatch.setenv("SPDMX_DDSP_ONESHOT", "1")
    assert pool_mod.ddsp_oneshot_enabled()
    monkeypatch.delenv("SPDMX_DDSP_ONESHOT", raising=False)
    assert not pool_mod.ddsp_oneshot_enabled()


def test_worker_serve_ping_and_shutdown():
    from synthesis.ddsp.worker import _handle_serve_request

    assert _handle_serve_request({"id": 1, "command": "ping"}) == {
        "id": 1,
        "ok": True,
        "pong": True,
    }
    assert _handle_serve_request({"id": 2, "command": "shutdown"}) == {
        "id": 2,
        "ok": True,
        "shutdown": True,
    }


def test_worker_serve_unknown_command_error():
    from synthesis.ddsp.worker import _handle_serve_request

    status = _handle_serve_request({"id": 3, "command": "nope"})
    assert status["ok"] is False
    assert status["id"] == 3
    assert "unknown command" in status["error"]
