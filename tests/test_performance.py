import json
import logging
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import performance
from app.performance import PerformanceLog
from app.transcript import Transcript
from tests.test_access_token import _app

GB = 1024**3


def _entry(finished_at: float) -> dict:
    return {
        "finished_at": finished_at,
        "model": "org/model",
        "settings": "OpenVINO・GPU・int8",
        "audio_seconds": 180.0,
        "processing_seconds": 90.0,
        "process_memory_bytes": 3 * GB,
        "available_memory_bytes": 5 * GB,
        "on_battery": True,
    }


def test_records_come_back_newest_first_and_survive_a_restart(tmp_path):
    path = tmp_path / "performance.json"
    saved = PerformanceLog(path)
    saved.add(_entry(1.0))
    saved.add(_entry(2.0))
    assert [r["finished_at"] for r in saved.recent()] == [2.0, 1.0]
    assert [r["finished_at"] for r in PerformanceLog(path).recent()] == [2.0, 1.0]


def test_only_the_latest_records_are_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(performance, "MAX_RECORDS", 2)
    path = tmp_path / "performance.json"
    saved = PerformanceLog(path)
    for finished_at in (1.0, 2.0, 3.0):
        saved.add(_entry(finished_at))
    assert [r["finished_at"] for r in saved.recent()] == [3.0, 2.0]
    assert [r["finished_at"] for r in json.loads(path.read_text(encoding="utf-8"))] == [2.0, 3.0]


def test_unreadable_file_is_renamed_and_the_log_starts_empty(tmp_path, caplog):
    path = tmp_path / "performance.json"
    path.write_text('[{"model": "org/model"}]', encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger="app.performance"):
        saved = PerformanceLog(path)
    assert saved.recent() == []
    assert not path.exists()
    assert (tmp_path / "performance.bad").exists()
    assert any("已改名為 performance.bad" in r.getMessage() for r in caplog.records)
    saved.add(_entry(1.0))
    assert json.loads(path.read_text(encoding="utf-8")) == [_entry(1.0)]


def test_record_keeps_times_memory_and_power(monkeypatch):
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=3 * GB))
    monkeypatch.setattr(performance.psutil, "Process", lambda: process)
    monkeypatch.setattr(performance.psutil, "virtual_memory", lambda: SimpleNamespace(available=5 * GB, total=16 * GB))
    monkeypatch.setattr(performance, "on_battery", lambda: True)
    monkeypatch.setattr(performance.time, "time", lambda: 1.0)
    transcript = Transcript(
        text="", segments=[], audio_seconds=180.0, processing_seconds=90.0, speakers=None, speakers_error=None
    )
    assert performance.record("org/model", "OpenVINO・GPU・int8", transcript) == _entry(1.0)
    remote = performance.record("remote", None, transcript)
    assert (remote["settings"], remote["process_memory_bytes"]) == (None, None)


def test_api_lists_saved_runs_and_current_memory(tmp_path):
    PerformanceLog(tmp_path / "performance.json").add(_entry(1.0))
    body = TestClient(_app(tmp_path, None)).get("/api/performance").json()
    assert body["runs"] == [_entry(1.0)]
    assert body["memory"]["inference_bytes"] is None
    assert 0 < body["memory"]["available_bytes"] <= body["memory"]["total_bytes"]
