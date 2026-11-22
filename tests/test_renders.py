import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.exports import ExportContext, render
from app.jobs import JobStatus
from app.renders import _SUBTITLES_NAME, SOURCE_NAME, RenderQueue
from app.transcript import Segment, Transcript
from tests.test_access_token import _app

LONG_LINE = "這支影片的每一句字幕都很長而且中間完全沒有任何標點符號"
EDGE_PIXELS = 8
QUIET = ["-nostdin", "-hide_banner", "-loglevel", "error"]
PORTRAIT_CANVAS = "PlayResX: 1080\nPlayResY: 1920\n"
AUDIO_CANVAS = "PlayResX: 1920\nPlayResY: 1080\n"


def _clip(path: Path, width: int, height: int) -> None:
    source = f"color=black:size={width}x{height}:rate=10:duration=1"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            *QUIET,
            "-f",
            "lavfi",
            "-i",
            source,
            "-c:v",
            "libx264",
            "-f",
            "mp4",
            str(path),
        ],
        check=True,
    )


def _service_with_job(tmp_path: Path, monkeypatch, video_size: tuple[int, int] | None) -> TestClient:
    transcript = Transcript(LONG_LINE, [Segment(0.0, 1.0, LONG_LINE, None)], 1.0, 0.1, None, None)
    job = SimpleNamespace(
        id="abc",
        filename="影片.mp4",
        status=JobStatus.SUCCEEDED,
        transcript=transcript,
        audio_seconds=1.0,
        video_size=video_size,
    )

    class FakeQueue:
        def __init__(self, *_args) -> None:
            pass

        def get(self, job_id):
            return job if job_id == job.id else None

    monkeypatch.setattr(main_module, "JobQueue", FakeQueue)
    return TestClient(_app(tmp_path, None))


def _ffmpeg_names(exe: str, listing: str) -> set[str]:
    output = subprocess.run([exe, "-hide_banner", f"-{listing}"], capture_output=True, text=True, check=True).stdout
    return {fields[1] for fields in map(str.split, output.splitlines()) if len(fields) > 1}


def test_bundled_ffmpeg_has_the_filter_and_encoders_burning_uses(tmp_path):
    command = RenderQueue(tmp_path / "renders", tmp_path, 1, 60)._command()
    filters = _ffmpeg_names(command[0], "filters")
    encoders = _ffmpeg_names(command[0], "encoders")
    assert command[command.index("-vf") + 1].split("=")[0] in filters
    assert command[command.index("-c:v") + 1] in encoders
    assert command[command.index("-c:a") + 1] in encoders


def test_cancel_right_after_the_worker_takes_the_render(tmp_path, monkeypatch):
    queue = RenderQueue(tmp_path / "renders", tmp_path, 1, 60)
    taken, release = threading.Event(), threading.Event()

    def render(_item):
        taken.set()
        release.wait(5)

    monkeypatch.setattr(queue, "_render", render)
    queue.start()
    render_id, directory = queue.new_directory()
    queue.submit(render_id, directory, "job", "subtitles", "video.mp4", 1.0)
    assert taken.wait(5)
    try:
        assert queue.cancel(render_id)
        assert queue.describe(queue.get(render_id))["status"] == "cancelling"
    finally:
        release.set()


@pytest.mark.parametrize("size", [(720, 1280), (1280, 720)])
def test_burned_subtitles_stay_inside_the_frame(tmp_path, size):
    width, height = size
    queue = RenderQueue(tmp_path / "renders", tmp_path, 1, 60)
    queue.start()
    render_id, directory = queue.new_directory()
    _clip(directory / SOURCE_NAME, width, height)
    transcript = Transcript(LONG_LINE * 2, [Segment(0.0, 1.0, LONG_LINE * 2, None)], 1.0, 0.1, None, None)
    subtitles = render(transcript, "ass", ExportContext("影片.mp4", "Microsoft JhengHei", size))
    item = queue.submit(render_id, directory, "job", subtitles, "video.mp4", 1.0)
    deadline = time.monotonic() + 60
    while not item.done:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert item.status == "succeeded", item.error
    frame = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            *QUIET,
            "-i",
            str(item.output),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    lit = np.frombuffer(frame, dtype=np.uint8).reshape(height, width) > 128
    if not lit.any():
        pytest.skip("這台電腦的 libass 找不到能畫中文的字型")
    assert not lit[:, :EDGE_PIXELS].any()
    assert not lit[:, -EDGE_PIXELS:].any()


@pytest.mark.parametrize(("stored", "canvas"), [(None, AUDIO_CANVAS), ((72, 128), PORTRAIT_CANVAS)])
def test_ass_export_uses_the_size_recorded_with_the_job(tmp_path, monkeypatch, stored, canvas):
    response = _service_with_job(tmp_path, monkeypatch, stored).get("/api/jobs/abc/export/ass")

    assert response.status_code == 200
    assert canvas in response.text


def test_burning_uses_the_uploaded_video_even_when_the_job_has_no_size(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _clip(clip, 72, 128)
    service = _service_with_job(tmp_path, monkeypatch, None)

    response = service.post("/api/jobs/abc/renders", files={"file": ("影片.mp4", clip.read_bytes(), "video/mp4")})

    assert response.status_code == 202
    subtitles = (tmp_path / "renders" / response.json()["id"] / _SUBTITLES_NAME).read_text(encoding="utf-8")
    assert PORTRAIT_CANVAS in subtitles
