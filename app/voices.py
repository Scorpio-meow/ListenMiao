from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_MAX_VOICE_MEMBERS = 128
_VECTOR_DECIMALS = 4


def _vector_keys(vectors: np.ndarray) -> list[bytes]:
    return [row.tobytes() for row in np.rint(vectors * 10**_VECTOR_DECIMALS).astype(np.int32)]


@dataclass(frozen=True)
class Voiceprint:
    vectors: np.ndarray
    weights: np.ndarray

    def similarity(self, other: Voiceprint) -> float:
        pairwise = self.vectors @ other.vectors.T
        return float(self.weights @ pairwise @ other.weights / (self.weights.sum() * other.weights.sum()))

    def keep_last(self, limit: int) -> Voiceprint:
        return Voiceprint(vectors=self.vectors[-limit:], weights=self.weights[-limit:])

    def without(self, others: Iterable[Voiceprint]) -> Voiceprint | None:
        seen = {key for other in others for key in _vector_keys(other.vectors)}
        keep = np.array([key not in seen for key in _vector_keys(self.vectors)])
        if not keep.any():
            return None
        return Voiceprint(vectors=self.vectors[keep], weights=self.weights[keep])

    def __add__(self, other: Voiceprint) -> Voiceprint:
        return Voiceprint(
            vectors=np.concatenate([self.vectors, other.vectors]),
            weights=np.concatenate([self.weights, other.weights]),
        )

    def to_json(self) -> dict:
        return {
            "vectors": np.round(self.vectors, _VECTOR_DECIMALS).tolist(),
            "weights": np.round(self.weights, 2).tolist(),
        }

    @classmethod
    def from_json(cls, data: dict) -> Voiceprint:
        return cls(
            vectors=np.asarray(data["vectors"], dtype=np.float32),
            weights=np.asarray(data["weights"], dtype=np.float32),
        )


@dataclass(frozen=True)
class Match:
    voice_id: str
    name: str
    similarity: float


def _outdated(voice: dict) -> bool:
    return "voiceprint" not in voice


class VoiceLibrary:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._voices: list[dict] = json.loads(path.read_text(encoding="utf-8"))["voices"] if path.exists() else []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        partial = self._path.with_suffix(".partial")
        partial.write_text(json.dumps({"voices": self._voices}, ensure_ascii=False), encoding="utf-8")
        os.replace(partial, self._path)

    def entries(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "id": v["id"],
                    "name": v["name"],
                    "samples": 0 if _outdated(v) else v["samples"],
                    "outdated": _outdated(v),
                    "created_at": v["created_at"],
                    "updated_at": v["updated_at"],
                }
                for v in self._voices
            ]

    def _known(self) -> list[tuple[str, str, Voiceprint]]:
        with self._lock:
            return [
                (v["id"], v["name"], Voiceprint.from_json(v["voiceprint"])) for v in self._voices if not _outdated(v)
            ]

    def ranking(self, voiceprints: dict[str, Voiceprint]) -> dict[str, list[Match]]:
        known = self._known()
        return {
            speaker_id: sorted(
                (
                    Match(voice_id=voice_id, name=name, similarity=voiceprint.similarity(stored))
                    for voice_id, name, stored in known
                ),
                key=lambda match: match.similarity,
                reverse=True,
            )
            for speaker_id, voiceprint in voiceprints.items()
        }

    def closer_voice(
        self, name: str, voiceprint: Voiceprint, same_recording: Iterable[Voiceprint]
    ) -> tuple[Match, float] | None:
        known = self._known()
        own = next(((voice_id, stored) for voice_id, known_name, stored in known if known_name == name), None)
        if own is None:
            return None
        same_recording = list(same_recording)
        others = [
            Match(voice_id=voice_id, name=other_name, similarity=voiceprint.similarity(rest))
            for voice_id, other_name, stored in known
            if other_name != name and (rest := stored.without(same_recording)) is not None
        ]
        if not others:
            return None
        closest = max(others, key=lambda match: match.similarity)
        own_similarity = voiceprint.similarity(own[1])
        if closest.similarity <= own_similarity:
            return None
        log.info(
            "要記進語者庫 %s 的聲紋比較像 %s（相似度 %.3f 對 %.3f）",
            own[0],
            closest.voice_id,
            closest.similarity,
            own_similarity,
        )
        return closest, own_similarity

    @staticmethod
    def match(ranking: dict[str, list[Match]], threshold: float) -> dict[str, Match]:
        candidates = [
            (match.similarity, speaker_id, match.voice_id, match.name)
            for speaker_id, ranked in ranking.items()
            for match in ranked
            if match.similarity >= threshold
        ]
        matches: dict[str, Match] = {}
        taken: set[str] = set()
        for similarity, speaker_id, voice_id, name in sorted(candidates, reverse=True):
            if speaker_id in matches or voice_id in taken:
                continue
            matches[speaker_id] = Match(voice_id=voice_id, name=name, similarity=similarity)
            taken.add(voice_id)
        return matches

    def remember(self, name: str, voiceprint: Voiceprint) -> dict:
        now = time.time()
        with self._lock:
            voice = next((v for v in self._voices if v["name"] == name), None)
            if voice is None:
                voice = {"id": secrets.token_hex(8), "name": name, "samples": 0, "created_at": now}
                self._voices.append(voice)
            if _outdated(voice):
                voice["samples"] = 0
                combined = voiceprint
            else:
                combined = Voiceprint.from_json(voice["voiceprint"]) + voiceprint
            voice["voiceprint"] = combined.keep_last(_MAX_VOICE_MEMBERS).to_json()
            voice["samples"] += 1
            voice["updated_at"] = now
            self._save()
            log.info("語者庫記住聲紋 %s（第 %d 次）", voice["id"], voice["samples"])
            return {"id": voice["id"], "name": voice["name"], "samples": voice["samples"]}

    def delete(self, voice_id: str) -> bool:
        with self._lock:
            remaining = [v for v in self._voices if v["id"] != voice_id]
            if len(remaining) == len(self._voices):
                return False
            self._voices = remaining
            self._save()
            log.info("語者庫刪除聲紋 %s", voice_id)
            return True
