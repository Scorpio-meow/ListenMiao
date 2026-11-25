import threading
import time
from types import SimpleNamespace

import pytest

from app import inference as inference_module
from app.inference import InferenceClient, _Pending


def test_failed_profile_change_still_restarts_the_inference_process(caplog):
    client = InferenceClient(SimpleNamespace(), SimpleNamespace(models=[]), lambda _record: None)
    calls = []
    client._shutdown_process = lambda: calls.append("stop")
    client._spawn = lambda: calls.append("spawn")

    def change():
        raise OSError("磁碟已滿")

    with pytest.raises(OSError, match="磁碟已滿"):
        client._restart(change, {"state": "starting"})
    assert calls == ["stop", "spawn"]
    assert not client._restarting.is_set()
    assert any("變更硬體設定失敗" in r.getMessage() for r in caplog.records)


class _FakeProcess:
    pid = None
    exitcode = None

    def __init__(self, alive: bool) -> None:
        self.alive = alive

    def is_alive(self) -> bool:
        return self.alive

    def kill(self) -> None:
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        pass


def test_cancel_during_a_crash_restart_leaves_one_inference_process(monkeypatch, tmp_path):
    client = InferenceClient(SimpleNamespace(logs_dir=tmp_path), SimpleNamespace(models=[]), lambda _record: None)
    spawned = []

    def spawn():
        spawned.append(_FakeProcess(alive=True))
        client._process = spawned[-1]

    client._spawn = spawn
    client._process = _FakeProcess(alive=False)
    client._pending["job"] = _Pending(lambda _stage: None, lambda _progress: None)
    monkeypatch.setattr(inference_module, "_children", lambda _process: [])
    sleeping, release = threading.Event(), threading.Event()
    real_sleep = time.sleep

    def sleep(seconds):
        if threading.current_thread().name != "crash-restart":
            return real_sleep(seconds)
        sleeping.set()
        release.wait()

    monkeypatch.setattr(inference_module.time, "sleep", sleep)
    crash = threading.Thread(target=client._restart_after_crash, name="crash-restart")
    crash.start()
    assert sleeping.wait(5)
    cancel = threading.Thread(target=client.cancel, args=("job",))
    cancel.start()
    cancel.join(0.5)
    release.set()
    crash.join(5)
    cancel.join(5)
    assert sum(process.is_alive() for process in spawned) == 1
    assert client._pending["job"].error == "工作已取消"


def test_performance_record_is_passed_on_even_when_the_job_is_no_longer_pending():
    seen = []
    client = InferenceClient(SimpleNamespace(), SimpleNamespace(models=[]), seen.append)
    client._handle(("performance", {"model": "org/model"}))
    assert seen == [{"model": "org/model"}]


def test_performance_report_names_the_settings_only_for_local_models(monkeypatch):
    import queue

    monkeypatch.setattr(inference_module, "performance_record", lambda model_id, used, transcript: (model_id, used))
    specs = {"local": SimpleNamespace(remote=None), "remote": SimpleNamespace(remote=SimpleNamespace())}
    catalog = SimpleNamespace(get=specs.get)
    profile = SimpleNamespace(settings=SimpleNamespace(describe=lambda: "OpenVINO・GPU・int8"))
    events = queue.Queue()
    inference_module._report_performance(events, catalog, profile, "local", None)
    inference_module._report_performance(events, catalog, None, "remote", None)
    assert [events.get_nowait(), events.get_nowait()] == [
        ("performance", ("local", "OpenVINO・GPU・int8")),
        ("performance", ("remote", None)),
    ]


def test_memory_is_unknown_unless_the_inference_process_is_running(monkeypatch):
    client = InferenceClient(SimpleNamespace(), SimpleNamespace(models=[]), lambda _record: None)
    assert client.memory_bytes() is None
    client._process = SimpleNamespace(pid=4242)
    monkeypatch.setattr(
        inference_module.psutil, "Process", lambda pid: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=pid))
    )
    assert client.memory_bytes() == 4242

    def gone(pid):
        raise inference_module.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(inference_module.psutil, "Process", gone)
    assert client.memory_bytes() is None


def test_cpu_meter_restarts_when_transcription_begins():
    import queue

    restarts = []
    meter = SimpleNamespace(restart=lambda: restarts.append("restart"))
    events = queue.Queue()
    inference_module._enter_stage(events, meter, "abc", "loading")
    assert restarts == []
    inference_module._enter_stage(events, meter, "abc", "transcribing")
    assert restarts == ["restart"]
    assert [events.get_nowait(), events.get_nowait()] == [("stage", "abc", "loading"), ("stage", "abc", "transcribing")]
