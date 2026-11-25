import httpx
import numpy as np

from app import backend_remote
from app.catalog import RemoteBackend

REMOTE = RemoteBackend(
    url="http://asr-service:9001",
    api_key_env="TEST_ASR_KEY",
    fields={},
    poll_seconds=0.01,
    timeout_seconds=5.0,
    job_timeout_seconds=5.0,
)


def _transcribe(monkeypatch, items: list[dict], audio_seconds: float):
    def reply(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"job_id": "j1"})
        return httpx.Response(200, json={"status": "done", "segments": items})

    def client(remote: RemoteBackend) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(reply), base_url=remote.url)

    monkeypatch.setattr(backend_remote, "_client", client)
    return backend_remote.transcribe(REMOTE, np.zeros(16000, dtype=np.float32), audio_seconds, lambda _: None)


def test_segment_without_end_time_ends_where_the_next_segment_starts(monkeypatch):
    items = [
        {"start_time": 0.0, "end_time": 26.96, "text": " 一 "},
        {"start_time": 26.96, "end_time": None, "text": "二"},
        {"start_time": 56.96, "end_time": 60.0, "text": " "},
        {"start_time": 60.0, "text": "三"},
    ]
    segments, text = _transcribe(monkeypatch, items, 100.0)
    assert [(s.text, s.start, s.end) for s in segments] == [
        ("一", 0.0, 26.96),
        ("二", 26.96, 56.96),
        ("三", 60.0, 100.0),
    ]
    assert text == "一\n二\n三"
