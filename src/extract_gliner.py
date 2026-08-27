from __future__ import annotations

from typing import Any, Iterable

from src.segment import find_sentence_id


def load_model(model_name: str):
    try:
        from gliner import GLiNER
    except ImportError as error:
        raise RuntimeError("GLiNER is not installed. Install gliner or choose another extractor.") from error
    return GLiNER.from_pretrained(model_name)


def extract_mentions(
    text: str,
    sentences: list[dict[str, Any]],
    model_name: str,
    labels: list[str],
    label_map: dict[str, str],
    threshold: float,
    model=None,
) -> list[dict[str, Any]]:
    model = model or load_model(model_name)
    predictions = model.predict_entities(text, labels, threshold=threshold)
    mentions: list[dict[str, Any]] = []
    for prediction in predictions:
        start, end = int(prediction["start"]), int(prediction["end"])
        raw_label = str(prediction["label"]).casefold()
        mapped = label_map.get(raw_label, raw_label.upper().replace(" ", "_"))
        mentions.append(
            {
                "id": "",
                "text": text[start:end],
                "start": start,
                "end": end,
                "label": mapped,
                "source": "gliner",
                "confidence": float(prediction.get("score", threshold)),
                "sent_id": find_sentence_id(sentences, start, end),
            }
        )
    return mentions


def merge_mentions(
    mentions: Iterable[dict[str, Any]],
    source_priority: tuple[str, ...] = ("spacy", "gliner", "keyword"),
) -> list[dict[str, Any]]:
    """Resolve duplicates and overlaps with a deterministic longest-span policy."""
    priority = {source: index for index, source in enumerate(source_priority)}
    candidates = sorted(
        mentions,
        key=lambda item: (
            item["start"],
            -(item["end"] - item["start"]),
            priority.get(item["source"], len(priority)),
            -item["confidence"],
            item["label"],
        ),
    )
    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        exact = next(
            (
                existing for existing in kept
                if existing["start"] == candidate["start"]
                and existing["end"] == candidate["end"]
                and existing["label"] == candidate["label"]
            ),
            None,
        )
        if exact:
            exact_rank = priority.get(exact["source"], len(priority))
            candidate_rank = priority.get(candidate["source"], len(priority))
            if (candidate_rank, -candidate["confidence"]) < (exact_rank, -exact["confidence"]):
                kept[kept.index(exact)] = candidate
            continue

        overlap = next(
            (existing for existing in kept if candidate["start"] < existing["end"] and existing["start"] < candidate["end"]),
            None,
        )
        if not overlap:
            kept.append(candidate)
            continue
        candidate_length = candidate["end"] - candidate["start"]
        overlap_length = overlap["end"] - overlap["start"]
        if candidate_length > overlap_length:
            kept[kept.index(overlap)] = candidate

    kept.sort(key=lambda item: (item["start"], item["end"], item["label"], item["source"]))
    for index, mention in enumerate(kept):
        mention["id"] = f"m{index}"
    return kept

