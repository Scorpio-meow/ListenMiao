import subprocess

import imageio_ffmpeg
import numpy as np
import pytest

from app.audio import probe_media, video_size, waveform

PORTRAIT = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'phone.mp4':
  Duration: 00:00:58.00, start: 0.000000, bitrate: 1499 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 720x1280, 1437 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
  Stream #0:1[0x2](und): Audio: aac (HE-AAC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 58 kb/s (default)
"""

ROTATED = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'rotated.mp4':
  Duration: 00:00:02.00, start: 0.000000, bitrate: 120 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1920x1080 [SAR 1:1 DAR 16:9], 44 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      handler_name    : VideoHandler
    Side data:
      {rotation}
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, mono, fltp, 69 kb/s (default)
"""

COVER_ART = """Input #0, mp3, from 'song.mp3':
  Duration: 00:00:02.04, start: 0.025057, bitrate: 72 kb/s
  Stream #0:0: Audio: mp3 (mp3float), 44100 Hz, mono, fltp, 64 kb/s, start 0.025057
  Stream #0:1: Video: mjpeg (Baseline), yuvj420p(pc, bt470bg/unknown/unknown), 300x300 [SAR 1:1 DAR 1:1], 90k tbr, 90k tbn, start 0.025056 (attached pic)
"""


def test_waveform_peaks_are_normalized_per_bucket(tmp_path):
    path = tmp_path / "audio.f32"
    np.array([0.1, -0.2, 0.0, 0.0, -0.5, 0.25], dtype=np.float32).tofile(path)

    assert waveform(path, 3) == [0.4, 0.0, 1.0]


def test_waveform_of_short_audio_leaves_empty_buckets(tmp_path):
    path = tmp_path / "audio.f32"
    np.array([0.5], dtype=np.float32).tofile(path)

    assert waveform(path, 2) == [0.0, 1.0]


def test_video_size_reads_the_frame_size_not_the_codec_tag():
    assert video_size(PORTRAIT) == (720, 1280)


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        ("Display Matrix: rotation of 90.00 degrees", (1080, 1920)),
        ("displaymatrix: rotation of -90.00 degrees", (1080, 1920)),
        ("Display Matrix: rotation of 270.00 degrees", (1080, 1920)),
        ("Display Matrix: rotation of -180.00 degrees", (1920, 1080)),
    ],
)
def test_video_size_follows_the_display_rotation(rotation, expected):
    assert video_size(ROTATED.format(rotation=rotation)) == expected


def test_cover_art_is_not_a_video():
    assert video_size(COVER_ART) is None
    assert video_size(COVER_ART + PORTRAIT.split("\n", 2)[2]) == (720, 1280)


def test_probe_reports_the_size_the_subtitle_filter_sees(tmp_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    stored = tmp_path / "stored.mp4"
    rotated = tmp_path / "rotated.mp4"
    quiet = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
    subprocess.run(
        [*quiet, "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:duration=1", "-c:v", "libx264", str(stored)],
        check=True,
    )
    subprocess.run([*quiet, "-display_rotation", "90", "-i", str(stored), "-c", "copy", str(rotated)], check=True)
    shown = subprocess.run(
        [*quiet, "-i", str(rotated), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
        check=True,
    ).stdout

    assert probe_media(stored).video_size == (640, 360)
    width, height = probe_media(rotated).video_size
    assert (width, height) == (360, 640)
    assert len(shown) == width * height
