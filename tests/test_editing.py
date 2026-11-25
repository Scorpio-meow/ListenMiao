import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.glossary import parse
from app.jobs import JobStatus, RevisionConflict, TranscriptInvalid
from app.speakers import NO_SPEECH
from app.transcript import Segment, Speaker
from tests.test_access_token import _app
from tests.test_jobs import FakeInference, _queue, _submit, _wait


def _finished(tmp_path, speakers=False):
    queue = _queue(tmp_path, FakeInference(finish=True))
    job = _submit(queue, tmp_path)
    _wait(lambda: job.status is JobStatus.SUCCEEDED)
    if speakers:
        job.transcript = replace(
            job.transcript,
            segments=[Segment(0.0, 0.4, "你", "S1"), Segment(0.4, 1.0, "好", "S2")],
            speakers=[
                Speaker("S1", "小明", 0.4, "v1", 0.9),
                Speaker("S2", "說話者 2", 0.6, None, None),
            ],
        )
    return queue, job


def test_replacing_segments_bumps_revision(tmp_path):
    queue, job = _finished(tmp_path)
    segments = [Segment(0.0, 0.5, "你", None), Segment(0.5, 1.0, "好", None)]
    updated = queue.replace_transcript(job.id, 0, segments, None)
    assert updated.revision == 1
    assert updated.transcript.text == "你\n好"
    assert queue.describe(updated)["result"]["revision"] == 1
    with pytest.raises(RevisionConflict):
        queue.replace_transcript(job.id, 0, segments, None)


def test_merging_speakers_drops_the_unused_one_and_recounts_time(tmp_path):
    queue, job = _finished(tmp_path, speakers=True)
    speakers = job.transcript.speakers
    segments = [replace(s, speaker="S1") for s in job.transcript.segments]
    updated = queue.replace_transcript(job.id, 0, segments, speakers).transcript
    assert [(s.id, s.seconds, s.voice_id) for s in updated.speakers] == [("S1", 1.0, "v1")]


def test_new_speaker_can_be_added(tmp_path):
    queue, job = _finished(tmp_path, speakers=True)
    speakers = [*job.transcript.speakers, Speaker("S3", "陳美玲", 0.0, None, None)]
    segments = [job.transcript.segments[0], replace(job.transcript.segments[1], speaker="S3")]
    updated = queue.replace_transcript(job.id, 0, segments, speakers).transcript
    assert [s.id for s in updated.speakers] == ["S1", "S3"]


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ([Segment(0.5, 0.4, "x", None)], "結束時間"),
        ([Segment(0.0, 1.5, "x", None)], "超出音訊範圍"),
        ([Segment(0.5, 0.8, "x", None), Segment(0.1, 0.2, "y", None)], "排序"),
        ([Segment(0.0, 0.5, "x", "S1")], "speakers 是 null"),
        ([Segment(float("nan"), 0.5, "x", None)], "不是數字"),
    ],
)
def test_invalid_segments_are_rejected(tmp_path, segments, message):
    queue, job = _finished(tmp_path)
    with pytest.raises(TranscriptInvalid, match=message):
        queue.replace_transcript(job.id, 0, segments, None)
    assert job.revision == 0


def test_zero_length_segment_from_the_model_can_be_saved(tmp_path):
    queue, job = _finished(tmp_path)
    updated = queue.replace_transcript(job.id, 0, [Segment(0.5, 0.5, "好", None)], None)
    assert updated.revision == 1


def test_speakers_can_be_added_by_hand_and_removed_again(tmp_path):
    queue, job = _finished(tmp_path)
    job.transcript = replace(job.transcript, speakers_error=NO_SPEECH)
    plain = job.transcript.segments
    labelled = [replace(s, speaker="S1") for s in plain]
    added = queue.replace_transcript(job.id, 0, labelled, [Speaker("S1", "說話者 1", 0.0, None, None)]).transcript
    assert [(s.id, s.seconds) for s in added.speakers] == [("S1", round(sum(s.end - s.start for s in plain), 1))]
    assert all(s.speaker == "S1" for s in added.segments)
    assert added.speakers_error == NO_SPEECH
    removed = queue.replace_transcript(job.id, 1, plain, None).transcript
    assert removed.speakers is None


def test_deleting_every_segment_drops_the_speakers(tmp_path):
    queue, job = _finished(tmp_path)
    labelled = [replace(s, speaker="S1") for s in job.transcript.segments]
    queue.replace_transcript(job.id, 0, labelled, [Speaker("S1", "說話者 1", 0.0, None, None)])
    emptied = queue.replace_transcript(job.id, 1, [], [Speaker("S1", "說話者 1", 0.0, None, None)]).transcript
    assert emptied.speakers is None
    restored = _queue(tmp_path, FakeInference(finish=True)).get(job.id).transcript
    assert (restored.speakers, restored.speakers_error) == (None, None)


def test_speakers_must_match_the_result(tmp_path):
    queue, job = _finished(tmp_path, speakers=True)
    segments = job.transcript.segments
    with pytest.raises(TranscriptInvalid, match="不在說話者清單"):
        queue.replace_transcript(job.id, 0, segments, job.transcript.speakers[:1])
    bad = [replace(job.transcript.speakers[0], id="X1"), job.transcript.speakers[1]]
    with pytest.raises(TranscriptInvalid, match="S 加數字"):
        queue.replace_transcript(job.id, 0, segments, bad)


def test_glossary_replacements_apply_to_new_results(tmp_path):
    queue = _queue(tmp_path, FakeInference(finish=True))
    queue._glossary.save(parse({"hotwords": "", "replacements": [{"find": "你好", "replace": "您好"}]}))
    job = _submit(queue, tmp_path)
    _wait(lambda: job.status is JobStatus.SUCCEEDED)
    assert job.transcript.segments[0].text == "您好"


def test_summaries_list_newest_first(tmp_path):
    queue, first = _finished(tmp_path)
    second = _submit(queue, tmp_path)
    _wait(lambda: second.status is JobStatus.SUCCEEDED)
    summaries = queue.summaries()
    assert [s["id"] for s in summaries] == [second.id, first.id]
    assert summaries[0]["segments"] == 1


def test_edits_survive_restart(tmp_path):
    queue, job = _finished(tmp_path)
    queue.replace_transcript(job.id, 0, [Segment(0.0, 1.0, "改過", None)], None)
    restored = _queue(tmp_path, FakeInference(finish=True)).get(job.id)
    assert restored.transcript.text == "改過"
    assert restored.revision == 1


def test_voiceprint_speakers_stay_listed_after_a_merge(tmp_path):
    import numpy as np

    from app.voices import Voiceprint

    queue, job = _finished(tmp_path, speakers=True)
    job.speaker_voiceprints = {"S1": Voiceprint(np.ones((1, 2), dtype=np.float32), np.ones(1, dtype=np.float32))}
    assert queue.describe(job)["result"]["voiceprint_speakers"] == ["S1"]
    segments = [replace(s, speaker="S2") for s in job.transcript.segments]
    updated = queue.replace_transcript(job.id, 0, segments, job.transcript.speakers)
    result = queue.describe(updated)["result"]
    assert [s["id"] for s in result["speakers"]] == ["S2"]
    assert result["voiceprint_speakers"] == ["S1"]


def test_remembering_a_voice_that_sounds_like_someone_else_needs_confirmation(tmp_path, monkeypatch):
    import numpy as np

    from app.catalog import load_catalog
    from app.voices import VoiceLibrary, Voiceprint
    from tests.test_access_token import ROOT

    def voice(*values):
        return Voiceprint(np.asarray([values], dtype=np.float32), np.ones(1, dtype=np.float32))

    slug = load_catalog(ROOT / "models.json").speakers.embedding.slug
    library = VoiceLibrary(tmp_path / "voices" / f"{slug}.json")
    library.remember("小明", voice(1, 0))
    library.remember("美玲", voice(0, 1))
    sounds_like_meiling = voice(0.6, 0.8)

    class FakeQueue:
        def __init__(self, *_args) -> None:
            pass

        def speaker_voice(self, job_id, speaker_id):
            return "小明", sounds_like_meiling

        def recording_voiceprints(self, job_id):
            return [sounds_like_meiling]

        def link_voice(self, job_id, speaker_id, voice_id):
            return job_id

        def describe(self, job_id):
            return {"id": job_id}

    monkeypatch.setattr(main_module, "JobQueue", FakeQueue)
    client = TestClient(_app(tmp_path, None))
    body = {"job_id": "abc", "speaker_id": "S1"}
    refused = client.post("/api/voices", json=body)
    assert refused.status_code == 409
    assert (
        refused.json()["detail"]
        == "這段聲音比較像語者庫裡的「美玲」（相似度 0.800），和「小明」原有的聲音只有 0.600，還沒有記住"
    )
    assert [v["samples"] for v in client.get("/api/voices").json()["voices"]] == [1, 1]
    accepted = client.post("/api/voices", json=body | {"confirm": True})
    assert accepted.status_code == 201
    assert accepted.json()["voice"]["samples"] == 2


def test_recording_voiceprints_cover_every_speaker_of_the_job(tmp_path):
    import numpy as np

    from app.voices import Voiceprint

    queue, job = _finished(tmp_path, speakers=True)
    first = Voiceprint(np.ones((1, 2), dtype=np.float32), np.ones(1, dtype=np.float32))
    second = Voiceprint(np.zeros((1, 2), dtype=np.float32), np.ones(1, dtype=np.float32))
    job.speaker_voiceprints = {"S1": first, "S2": second}
    assert queue.recording_voiceprints(job.id) == [first, second]


def test_ranking_is_logged_with_the_runner_up_and_the_gap():
    from app.jobs import _describe_ranking
    from app.voices import Match

    ranked = [Match("v1", "小明", 0.506), Match("v2", "美玲", 0.183)]
    assert _describe_ranking(ranked) == "第一名 v1 0.506，第二名 v2 0.183，差距 0.323"
    assert _describe_ranking(ranked[:1]) == "第一名 v1 0.506"


def test_editing_keeps_the_result_past_its_original_expiry(tmp_path):
    queue, job = _finished(tmp_path)
    finished = time.time() - 3600 - 1
    job.finished_at = finished
    queue.replace_transcript(job.id, 0, [Segment(0.0, 1.0, "改過", None)], None)
    with queue._cond:
        queue._purge_expired()
    assert queue.get(job.id) is job
    restored = _queue(tmp_path, FakeInference(finish=True)).get(job.id)
    assert restored.transcript.text == "改過"
    assert restored.finished_at == finished


def test_speaker_with_only_a_zero_length_segment_is_kept(tmp_path):
    queue, job = _finished(tmp_path, speakers=True)
    speakers = [*job.transcript.speakers, Speaker("S3", "說話者 3", 0.0, None, None)]
    segments = [Segment(0.0, 0.4, "你", "S1"), Segment(0.4, 0.4, "嗯", "S3"), Segment(0.4, 1.0, "好", "S2")]
    updated = queue.replace_transcript(job.id, 0, segments, speakers)
    assert [s.id for s in updated.transcript.speakers] == ["S1", "S2", "S3"]
    queue.replace_transcript(job.id, updated.revision, segments, updated.transcript.speakers)


def test_saved_result_with_an_empty_speaker_list_becomes_editable(tmp_path):
    queue, job = _finished(tmp_path)
    job.transcript = replace(job.transcript, speakers=[])
    with queue._cond:
        queue._persist(job)
    restored = _queue(tmp_path, FakeInference(finish=True)).get(job.id)
    assert restored.transcript.speakers is None
    assert restored.transcript.speakers_error == NO_SPEECH


def test_saving_keeps_a_deleted_voice_as_unlinked_instead_of_rejecting(tmp_path, monkeypatch):
    received = {}

    class RecordingQueue:
        def __init__(self, *_args) -> None:
            pass

        def replace_transcript(self, job_id, revision, segments, speakers):
            received.update(segments=segments, speakers=speakers)
            return job_id

        def describe(self, job_id):
            return {"id": job_id}

    monkeypatch.setattr(main_module, "JobQueue", RecordingQueue)
    body = {
        "revision": 0,
        "segments": [{"start": 0.123, "end": 1.0, "text": " 你好\n  世界 ", "speaker": "S1"}],
        "speakers": [{"id": "S1", "name": "小明", "voice_id": "deleted-voice", "similarity": 0.9}],
    }
    response = TestClient(_app(tmp_path, None)).put("/api/jobs/abc/transcript", json=body)
    assert response.status_code == 200
    assert received["segments"] == [Segment(0.12, 1.0, "你好 世界", "S1")]
    assert received["speakers"] == [Speaker("S1", "小明", 0.0, None, None)]
