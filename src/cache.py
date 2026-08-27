from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import read_json, stable_hash, write_json_atomic


class JsonCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.values: dict[str, Any] = read_json(self.path) if self.path.exists() else {}
        self.hits = 0
        self.misses = 0
        self.dirty = False

    @staticmethod
    def make_key(
        stage_name: str,
        model_name: str,
        model_version: str,
        prompt_template_version: str,
        input_text: str,
    ) -> str:
        return stable_hash((stage_name, model_name, model_version, prompt_template_version, input_text))

    def get(self, key: str) -> Any | None:
        if key in self.values:
            self.hits += 1
            return self.values[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value
        self.dirty = True

    def save(self) -> None:
        if self.dirty:
            write_json_atomic(self.path, self.values)
            self.dirty = False

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

