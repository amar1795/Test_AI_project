from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.normalize import NormalizedMention, normalize_mention
from src.utils import read_json, write_json_atomic

try:
    from rapidfuzz.fuzz import token_sort_ratio as _rapid_token_sort_ratio
except ImportError:  # dependency-free fallback
    _rapid_token_sort_ratio = None


def token_sort_ratio(left: str, right: str) -> float:
    if _rapid_token_sort_ratio:
        return float(_rapid_token_sort_ratio(left, right))
    a, b = " ".join(sorted(left.split())), " ".join(sorted(right.split()))
    return 100.0 * SequenceMatcher(None, a, b).ratio()


@dataclass(frozen=True)
class ResolutionResult:
    status: str
    entity_id: str | None
    canonical: str | None
    label: str
    method: str
    confidence: float
    candidate_entity_ids: tuple[str, ...] = ()


class EntityRegistry:
    def __init__(self, path: str | Path, config: dict[str, Any]):
        self.path = Path(path)
        self.config = config
        self.entities: dict[str, dict[str, Any]] = read_json(self.path) if self.path.exists() else {}
        if not isinstance(self.entities, dict):
            raise ValueError("registry must be a JSON object keyed by entity_id")

    def _normalized(self, value: str, label: str) -> NormalizedMention:
        return normalize_mention(
            value,
            label,
            self.config.get("honorifics", ()),
            self.config.get("corporate_suffixes", ()),
        )

    @staticmethod
    def _mention_key(document: dict[str, Any], mention: dict[str, Any]) -> str:
        return f"{document['doc_id']}:{mention['start']}:{mention['end']}:{mention['label']}"

    def _next_id(self) -> str:
        numbers = [int(match.group(1)) for key in self.entities if (match := re.fullmatch(r"ent_(\d+)", key))]
        return f"ent_{(max(numbers, default=0) + 1):05d}"

    def _existing_record(self, mention_key: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        for entity in self.entities.values():
            record = entity.get("resolution_records", {}).get(mention_key)
            if record:
                return entity, record
        return None

    def _blocked(self, normalized: NormalizedMention, label: str) -> list[str]:
        blocked: list[str] = []
        for entity_id, entity in self.entities.items():
            if entity["label"] != label:
                continue
            keys = {self._normalized(alias, label).key for alias in entity.get("blocked_aliases", [])}
            if normalized.key in keys:
                blocked.append(entity_id)
        return sorted(blocked)

    def _rank_candidates(self, normalized: NormalizedMention, label: str) -> list[tuple[float, dict[str, Any]]]:
        ranked: list[tuple[float, dict[str, Any]]] = []
        for entity in self.entities.values():
            if entity["label"] != label:
                continue
            forms = [entity["canonical"], *entity.get("aliases", [])]
            scores: list[float] = []
            for form in forms:
                candidate = self._normalized(form, label)
                scores.append(token_sort_ratio(normalized.key, candidate.key))
                if label in {"PERSON", "ORG"}:
                    scores.append(token_sort_ratio(normalized.initials_key, candidate.initials_key))
            ranked.append((max(scores, default=0.0), entity))
        return sorted(ranked, key=lambda pair: (-pair[0], pair[1]["entity_id"]))

    def _record(
        self,
        entity: dict[str, Any],
        document: dict[str, Any],
        mention: dict[str, Any],
        method: str,
        confidence: float,
    ) -> None:
        mention_key = self._mention_key(document, mention)
        records = entity.setdefault("resolution_records", {})
        if mention_key in records:
            return
        if mention["text"] not in entity["aliases"]:
            entity["aliases"].append(mention["text"])
            entity["aliases"].sort(key=lambda item: (item.casefold(), item))
        entity.setdefault("mention_keys", []).append(mention_key)
        entity["mention_keys"].sort()
        entity.setdefault("document_ids", [])
        if document["doc_id"] not in entity["document_ids"]:
            entity["document_ids"].append(document["doc_id"])
            entity["document_ids"].sort()
        entity["doc_count"] = len(entity["document_ids"])
        counts = entity.setdefault("surface_counts", {})
        counts[mention["text"]] = counts.get(mention["text"], 0) + 1
        records[mention_key] = {"method": method, "confidence": confidence}

    def _create(self, normalized: NormalizedMention, document: dict[str, Any], mention: dict[str, Any]) -> dict[str, Any]:
        entity_id = self._next_id()
        date = str(document.get("published_at", ""))[:10]
        entity = {
            "entity_id": entity_id,
            "canonical": normalized.canonical or mention["text"],
            "label": mention["label"],
            "aliases": [],
            "blocked_aliases": [],
            "first_seen": date,
            "doc_count": 0,
            "document_ids": [],
            "mention_keys": [],
            "resolution_records": {},
            "surface_counts": {},
        }
        self.entities[entity_id] = entity
        self._record(entity, document, mention, "new_entity", 1.0)
        return entity

    def resolve(self, mention: dict[str, Any], document: dict[str, Any]) -> ResolutionResult:
        label = mention["label"]
        mention_key = self._mention_key(document, mention)
        existing = self._existing_record(mention_key)
        if existing:
            entity, record = existing
            return ResolutionResult(
                "resolved", entity["entity_id"], entity["canonical"], label,
                record["method"], float(record["confidence"]),
            )

        normalized = self._normalized(mention["text"], label)
        blocked = self._blocked(normalized, label)
        if blocked:
            return ResolutionResult("ambiguous", None, None, label, "blocked_alias", 0.0, tuple(blocked[:3]))

        canonical_matches: list[dict[str, Any]] = []
        alias_matches: list[dict[str, Any]] = []
        initials_matches: list[dict[str, Any]] = []
        for entity in self.entities.values():
            if entity["label"] != label:
                continue
            canonical = self._normalized(entity["canonical"], label)
            if canonical.key == normalized.key:
                canonical_matches.append(entity)
                continue
            aliases = [self._normalized(alias, label) for alias in entity.get("aliases", [])]
            if any(alias.key == normalized.key for alias in aliases):
                alias_matches.append(entity)
            elif label in {"PERSON", "ORG"} and normalized.initials_key == canonical.initials_key:
                initials_matches.append(entity)

        for matches, method, confidence in (
            (canonical_matches, "registry_exact", 1.0),
            (alias_matches, "registry_alias", 0.95),
            (initials_matches, "registry_alias", 0.95),
        ):
            if len(matches) == 1:
                entity = matches[0]
                self._record(entity, document, mention, method, confidence)
                return ResolutionResult("resolved", entity["entity_id"], entity["canonical"], label, method, confidence)
            if len(matches) > 1:
                ids = tuple(sorted(entity["entity_id"] for entity in matches)[:3])
                return ResolutionResult("ambiguous", None, None, label, "multiple_alias_matches", 0.0, ids)

        ranked = self._rank_candidates(normalized, label)
        if ranked:
            best_score, best_entity = ranked[0]
            if best_score >= float(self.config.get("fuzzy_accept_threshold", 92.0)):
                confidence = round(best_score / 100.0, 4)
                self._record(best_entity, document, mention, "fuzzy_high", confidence)
                return ResolutionResult(
                    "resolved", best_entity["entity_id"], best_entity["canonical"], label, "fuzzy_high", confidence
                )
            if best_score >= float(self.config.get("fuzzy_ambiguous_threshold", 80.0)):
                ids = tuple(entity["entity_id"] for score, entity in ranked[:3] if score >= 80.0)
                return ResolutionResult("ambiguous", None, None, label, "fuzzy_ambiguous", best_score / 100.0, ids)

        entity = self._create(normalized, document, mention)
        return ResolutionResult("resolved", entity["entity_id"], entity["canonical"], label, "new_entity", 1.0)

    def accept_disambiguation(
        self,
        entity_id: str,
        mention: dict[str, Any],
        document: dict[str, Any],
        confidence: float,
    ) -> ResolutionResult:
        entity = self.entities[entity_id]
        if entity["label"] != mention["label"]:
            raise ValueError("LLM candidate label does not match mention label")
        self._record(entity, document, mention, "llm_disambiguation", confidence)
        return ResolutionResult(
            "resolved", entity_id, entity["canonical"], entity["label"], "llm_disambiguation", confidence
        )

    def save(self) -> None:
        write_json_atomic(self.path, {key: self.entities[key] for key in sorted(self.entities)})
