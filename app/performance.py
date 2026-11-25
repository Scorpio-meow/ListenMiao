from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import psutil

from .hardware import on_battery
from .transcript import Transcript

log = logging.getLogger(__name__)

MAX_RECORDS = 200
_FIELDS = frozenset(
    {
        "finished_at",
        "model",
        "settings",
        "audio_seconds",
        "processing_seconds",
        "process_memory_bytes",
        "available_memory_bytes",
        "on_battery",
    }
)


def memory_now() -> dict:
    memory = psutil.virtual_memory()
    return {"available_bytes": memory.available, "total_bytes": memory.total}


def record(model_id: str, settings: str | None, transcript: Transcript) -> dict:
    return {
        "finished_at": time.time(),
        "model": model_id,
        "settings": settings,
        "audio_seconds": transcript.audio_seconds,
        "processing_seconds": transcript.processing_seconds,
        "process_memory_bytes": psutil.Process().memory_info().rss if settings is not None else None,
        "available_memory_bytes": psutil.virtual_memory().available,
        "on_battery": on_battery(),
    }


class PerformanceLog:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._records = self._load()

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            records = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(records, list) or not all(isinstance(r, dict) and r.keys() >= _FIELDS for r in records):
                raise ValueError("效能紀錄必須是陣列，每筆都要有全部欄位")
        except (OSError, ValueError):
            bad = self._path.with_suffix(".bad")
            try:
                self._path.replace(bad)
            except OSError as exc:
                log.exception("無法讀取效能紀錄 %s，也無法改名（%s），這次從空的紀錄開始", self._path.name, exc)
            else:
                log.exception("無法讀取效能紀錄 %s，已改名為 %s，這次從空的紀錄開始", self._path.name, bad.name)
            return []
        return records[-MAX_RECORDS:]

    def add(self, entry: dict) -> None:
        with self._lock:
            self._records = [*self._records, entry][-MAX_RECORDS:]
            self._path.parent.mkdir(parents=True, exist_ok=True)
            partial = self._path.with_suffix(".partial")
            partial.write_text(json.dumps(self._records, ensure_ascii=False), encoding="utf-8")
            os.replace(partial, self._path)

    def recent(self) -> list[dict]:
        with self._lock:
            return self._records[::-1]
