import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.backend_openvino import OpenVinoEngine
from app.catalog import load_catalog

SPEC = load_catalog(Path(__file__).resolve().parent.parent / "models.json").models[0]


class _Pipeline:
    def __init__(self, metrics) -> None:
        self._metrics = metrics

    def generate(self, audio, **options):
        chunk = SimpleNamespace(text=" 你好 ", start_ts=0.0, end_ts=1.5)
        return SimpleNamespace(chunks=[chunk], texts=["你好"], perf_metrics=self._metrics)


def _transcribe(metrics):
    audio = np.zeros(32000, dtype=np.float32)
    return OpenVinoEngine(None, None).transcribe(_Pipeline(metrics), SPEC, audio, 2.0, "", lambda _: None)


def test_breakdown_reports_seconds_per_stage(caplog):
    caplog.set_level(logging.INFO, logger="app.backend_openvino")
    metrics = SimpleNamespace(
        whisper_raw_metrics=SimpleNamespace(
            encode_inference_durations=[2_900_000.0, 3_100_000.0],
            decode_inference_durations=[40_000.0] * 110,
        ),
        get_generate_duration=lambda: SimpleNamespace(mean=12_500.0),
        get_num_generated_tokens=lambda: 110,
    )
    _transcribe(metrics)
    assert "辨識分項：2 個視窗，encoder 6.0 秒，decoder 4.4 秒（110 個 token），其他 2.1 秒" in caplog.text


def test_unreadable_metrics_do_not_fail_the_transcription(caplog):
    segments, text = _transcribe(SimpleNamespace())
    assert [segment.text for segment in segments] == ["你好"]
    assert text == "你好"
    assert "辨識分項時間讀不出來" in caplog.text
