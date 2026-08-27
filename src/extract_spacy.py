from __future__ import annotations

from typing import Any, Iterable

from src.segment import find_sentence_id


def load_model(model_name: str):
    try:
        import spacy
    except ImportError as error:
        raise RuntimeError("spaCy is not installed; install requirements.txt or use --extractor keyword") from error
    try:
        return spacy.load(model_name)
    except OSError as error:
        raise RuntimeError(
            f"spaCy model {model_name!r} is not installed. Run: python -m spacy download {model_name}"
        ) from error


def extract_mentions(
    text: str,
    sentences: list[dict[str, Any]],
    model_name: str,
    label_map: dict[str, str],
    nlp=None,
) -> list[dict[str, Any]]:
    nlp = nlp or load_model(model_name)
    doc = nlp(text)
    mentions: list[dict[str, Any]] = []
    for entity in doc.ents:
        mapped = label_map.get(entity.label_)
        if not mapped:
            continue
        mentions.append(
            {
                "id": "",
                "text": text[entity.start_char:entity.end_char],
                "start": entity.start_char,
                "end": entity.end_char,
                "label": mapped,
                "source": "spacy",
                "confidence": 1.0,
                "sent_id": find_sentence_id(sentences, entity.start_char, entity.end_char),
            }
        )
    return mentions


def extract_batch(
    texts: Iterable[str],
    model_name: str,
    label_map: dict[str, str],
    batch_size: int = 16,
):
    """Yield raw spaCy entities in batches for benchmark and production callers."""
    nlp = load_model(model_name)
    for doc in nlp.pipe(texts, batch_size=batch_size):
        yield [
            {
                "text": entity.text,
                "start": entity.start_char,
                "end": entity.end_char,
                "label": label_map[entity.label_],
            }
            for entity in doc.ents
            if entity.label_ in label_map
        ]

