from __future__ import annotations

import re
from typing import Any


SENTENCE_END = re.compile(r"[^.!?]*(?:[.!?]+(?=\s|$)|$)", re.DOTALL)


def regex_segment(text: str) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    for match in SENTENCE_END.finditer(text):
        raw_start, raw_end = match.span()
        if raw_start == raw_end:
            continue
        chunk = match.group(0)
        left = len(chunk) - len(chunk.lstrip())
        right = len(chunk.rstrip())
        start, end = raw_start + left, raw_start + right
        if start < end:
            sentences.append({"id": len(sentences), "start": start, "end": end, "text": text[start:end]})
    return sentences


def spacy_segment(text: str, model_name: str = "en_core_web_sm") -> list[dict[str, Any]]:
    try:
        import spacy
    except ImportError as error:
        raise RuntimeError("spaCy is not installed; install requirements.txt or use --segmenter regex") from error
    nlp = spacy.load(model_name, disable=["ner"])
    doc = nlp(text)
    return [
        {"id": index, "start": sent.start_char, "end": sent.end_char, "text": text[sent.start_char:sent.end_char]}
        for index, sent in enumerate(doc.sents)
    ]


def find_sentence_id(sentences: list[dict[str, Any]], start: int, end: int) -> int | None:
    for sentence in sentences:
        if sentence["start"] <= start and end <= sentence["end"]:
            return sentence["id"]
    return None

