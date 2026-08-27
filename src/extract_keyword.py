from __future__ import annotations

import re
from typing import Any

from src.segment import find_sentence_id


def extract_mentions(
    text: str,
    sentences: list[dict[str, Any]],
    keyword_entities: dict[str, list[str]],
) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for label, terms in keyword_entities.items():
        for term in sorted(set(terms), key=lambda item: (-len(item), item.casefold())):
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(text):
                mentions.append(
                    {
                        "id": "",
                        "text": text[match.start():match.end()],
                        "start": match.start(),
                        "end": match.end(),
                        "label": label,
                        "source": "keyword",
                        "confidence": 1.0,
                        "sent_id": find_sentence_id(sentences, match.start(), match.end()),
                    }
                )
    return mentions

